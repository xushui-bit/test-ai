"""
BGE Embedding Service - 为本地模型提供嵌入服务
支持多种API格式：自定义API、OpenAI兼容API、LangChain兼容API
"""
import os
import logging
import hashlib
import time
import numpy as np
import torch
import uvicorn
from typing import List, Optional, Union
from functools import wraps
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator

# 设置离线模式以使用本地模型
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# 尝试导入sentence-transformers库
try:
    from sentence_transformers import SentenceTransformer
    import torch.nn.functional as F
except Exception:
    SentenceTransformer = None
    F = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bge_service")

app = FastAPI(title="bge-embed-service")

class Settings:
    """服务配置类"""
    model_name: str = os.environ.get("BGE_MODEL", r"D:\ai\models\baai\bge-small-zh-v1.5")
    cache_dir: str = os.environ.get("BGE_CACHE_DIR", r"d:\ai\models\baai\bge-small-zh-v1.5-cache")
    port: int = int(os.environ.get("BGE_PORT", "7860"))
    host: str = os.environ.get("BGE_HOST", "127.0.0.1")
    device: str = os.environ.get("BGE_DEVICE", "gpu")
    batch_size: int = int(os.environ.get("BGE_BATCH_SIZE", "32"))
    max_seq_length: int = int(os.environ.get("BGE_MAX_SEQ_LENGTH", "512"))

config = Settings()
os.makedirs(config.cache_dir, exist_ok=True)

class DummyEmbedder:
    """
    确定性虚拟嵌入器，用于离线测试
    从输入文本的SHA256生成固定大小的向量，确保结果稳定
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True, batch_size=None):
        """
        对文本列表进行编码

        Args:
            texts: 文本列表
            show_progress_bar: 是否显示进度条
            convert_to_numpy: 是否转换为numpy数组
            batch_size: 批处理大小

        Returns:
            编码后的嵌入向量数组
        """
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode('utf-8')).digest()
            arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            vals = (arr / 255.0) * 2 - 1
            if vals.size < self.dim:
                vals = np.tile(vals, int(np.ceil(self.dim / vals.size)))[:self.dim]
            else:
                vals = vals[:self.dim]
            out.append(vals.astype(np.float32))
        return np.stack(out, axis=0)

class EnhancedReq(BaseModel):
    """增强型请求模型"""
    texts: List[str]
    normalize: bool = True
    batch_size: Optional[int] = None
    show_progress: bool = False

    @validator('texts')
    def texts_non_empty(cls, v):
        """验证文本列表非空"""
        if not v or len(v) == 0:
            raise ValueError('texts must be non-empty')
        if len(v) > 1000:  # 添加限制
            raise ValueError('too many texts, maximum 1000 per request')
        return v

class EnhancedResp(BaseModel):
    """增强型响应模型"""
    embeddings: List[List[float]]
    model_info: dict
    normalized: bool
    count: int

class EmbeddingCache:
    """嵌入缓存类，提高重复请求的响应速度"""
    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size

    def get_key(self, text):
        """生成文本的缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text):
        """获取缓存的嵌入向量"""
        return self.cache.get(self.get_key(text))

    def set(self, text, embedding):
        """设置缓存的嵌入向量"""
        if len(self.cache) >= self.max_size:
            # 简单的LRU策略：删除第一个项目
            self.cache.pop(next(iter(self.cache)))
        self.cache[self.get_key(text)] = embedding

embedding_cache = EmbeddingCache()

class ModelManager:
    """模型管理器，负责加载和管理嵌入模型"""
    def __init__(self):
        self.model = None
        self.model_info = {}

    def load_model(self):
        """
        加载嵌入模型
        根据配置自动选择设备（CPU/GPU）
        """
        global config

        if self.model is not None:
            return self.model

        # 自动检测设备
        if config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = config.device

        log.info(f"Using device: {device}")

        try:
            if SentenceTransformer is None:
                # sentence-transformers not available in this environment
                log.warning("sentence-transformers not installed or failed to import; using DummyEmbedder")
                self.model = DummyEmbedder(dim=384)
                self.model_info = {
                    "source": "dummy",
                    "device": "cpu",
                    "embedding_dim": 384,
                    "max_seq_length": 512
                }
                return self.model

            # load actual model
            self.model = SentenceTransformer(
                config.model_name,
                device=device,
                cache_folder=config.cache_dir
            )

            # 配置模型参数
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = config.max_seq_length

            self.model_info = {
                "source": config.model_name,
                "device": device,
                "embedding_dim": self.model.get_sentence_embedding_dimension(),
                "max_seq_length": getattr(self.model, 'max_seq_length', 512)
            }

            log.info(f"Model loaded: {self.model_info}")

        except Exception as e:
            log.warning(f"Failed to load model: {e}, using dummy embedder")
            self.model = DummyEmbedder(dim=384)
            self.model_info = {
                "source": "dummy",
                "device": "cpu",
                "embedding_dim": 384,
                "max_seq_length": 512
            }

        return self.model

    def normalize_embeddings(self, embeddings):
        """
        归一化嵌入向量
        将向量的L2范数归一化为1
        """
        if isinstance(embeddings, np.ndarray):
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / norms
        else:
            # 对于PyTorch tensor
            return F.normalize(embeddings, p=2, dim=1)

model_manager = ModelManager()

def timing_decorator(func):
    """
    计时装饰器
    记录函数执行时间
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        log.info(f"Function {func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@app.on_event("startup")
def startup_event():
    """应用启动时预加载模型"""
    log.info("Starting up embedding service...")
    model_manager.load_model()

@app.get("/health")
def health():
    """健康检查端点"""
    return {
        "status": "ok",
        "model_loaded": model_manager.model is not None and not isinstance(model_manager.model, DummyEmbedder),
        "model_info": model_manager.model_info
    }

@app.get("/model-info")
def get_model_info():
    """获取模型信息"""
    return model_manager.model_info

@app.post("/embed", response_model=EnhancedResp)
@timing_decorator
def embed(req: EnhancedReq):
    """
    主要的嵌入端点
    处理文本列表并返回嵌入向量
    """
    model = model_manager.model

    if model is None:
        raise HTTPException(status_code=503, detail="Embedding model not available")

    try:
        # 检查缓存
        cached_embeddings = []
        texts_to_process = []
        cache_indices = []

        for i, text in enumerate(req.texts):
            cached = embedding_cache.get(text)
            if cached is not None:
                cached_embeddings.append(cached)
                cache_indices.append(i)
            else:
                texts_to_process.append(text)

        # 处理未缓存的文本
        if texts_to_process:
            new_embeddings = model.encode(
                texts_to_process,
                show_progress_bar=req.show_progress,
                convert_to_numpy=True,
                batch_size=req.batch_size or config.batch_size
            )

            # 缓存新结果
            for text, emb in zip(texts_to_process, new_embeddings):
                embedding_cache.set(text, emb)
        else:
            new_embeddings = np.array([])

        # 合并结果
        final_batch_size = len(req.texts)
        if cached_embeddings and new_embeddings.size > 0:
            embedding_dim = new_embeddings.shape[1]
            all_embeddings = np.zeros((final_batch_size, embedding_dim))
            all_embeddings[cache_indices] = cached_embeddings
            process_indices = [i for i in range(final_batch_size) if i not in cache_indices]
            all_embeddings[process_indices] = new_embeddings
        elif cached_embeddings:
            all_embeddings = np.array(cached_embeddings)
        else:
            all_embeddings = new_embeddings

        # 归一化处理
        if req.normalize and not isinstance(model, DummyEmbedder):
            all_embeddings = model_manager.normalize_embeddings(all_embeddings)

        return EnhancedResp(
            embeddings=all_embeddings.tolist(),
            model_info=model_manager.model_info,
            normalized=req.normalize,
            count=len(req.texts)
        )

    except Exception as e:
        log.exception("Failed to compute embeddings: %s", e)
        raise HTTPException(status_code=500, detail="embedding failed")

# 兼容旧版API
class Req(BaseModel):
    """旧版API请求模型"""
    texts: List[str]

class Resp(BaseModel):
    """旧版API响应模型"""
    embeddings: List[List[float]]

@app.post("/embed_legacy", response_model=Resp)
def embed_legacy(req: Req):
    """兼容旧版API"""
    enhanced_req = EnhancedReq(texts=req.texts, normalize=True)
    enhanced_resp = embed(enhanced_req)
    return Resp(embeddings=enhanced_resp.embeddings)


# OpenAI-style request/response compatibility
class OpenAIEmbReq(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]
    user: Optional[str] = None  # OpenAI API also supports user field

@app.post("/v1/embeddings")
@timing_decorator
def openai_embeddings(req: OpenAIEmbReq):
    """
    OpenAI兼容的嵌入端点
    接收OpenAI格式的请求并返回OpenAI格式的响应
    """
    try:
        # 将输入标准化为字符串列表
        texts = [req.input] if isinstance(req.input, str) else req.input
        if not texts or len(texts) == 0:
            raise HTTPException(status_code=400, detail="input must be a non-empty string or list of strings")

        # 确保模型已加载
        model_manager.load_model()
        model = model_manager.model
        if model is None:
            raise HTTPException(status_code=503, detail="Embedding model not available")

        # 检查缓存
        cached_embeddings = []
        texts_to_process = []
        cache_indices = []

        for i, text in enumerate(texts):
            cached = embedding_cache.get(text)
            if cached is not None:
                cached_embeddings.append(cached)
                cache_indices.append(i)
            else:
                texts_to_process.append(text)

        # 处理未缓存的文本
        if texts_to_process:
            new_embeddings = model.encode(
                texts_to_process,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=config.batch_size
            )

            # 缓存新结果
            for text, emb in zip(texts_to_process, new_embeddings):
                embedding_cache.set(text, emb)
        else:
            new_embeddings = np.array([])

        # 合并结果
        final_batch_size = len(texts)
        if cached_embeddings and new_embeddings.size > 0:
            embedding_dim = new_embeddings.shape[1] if new_embeddings.size > 0 else cached_embeddings[0].shape[0]
            all_embeddings = np.zeros((final_batch_size, embedding_dim))
            all_embeddings[cache_indices] = cached_embeddings
            process_indices = [i for i in range(final_batch_size) if i not in cache_indices]
            all_embeddings[process_indices] = new_embeddings
        elif cached_embeddings:
            all_embeddings = np.array(cached_embeddings)
        else:
            all_embeddings = new_embeddings

        # 归一化处理
        if not isinstance(model, DummyEmbedder):
            all_embeddings = model_manager.normalize_embeddings(all_embeddings)

        # 构建OpenAI格式的响应
        data = []
        for i, embedding in enumerate(all_embeddings):
            data.append({
                "object": "embedding",
                "embedding": embedding.tolist(),
                "index": i
            })

        # 计算token使用情况（近似）
        total_tokens = sum(len(text.split()) for text in texts)

        return {
            "object": "list",
            "data": data,
            "model": req.model or model_manager.model_info.get("source", "bge-local"),
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        log.exception("OpenAI-style embedding failed: %s", e)
        raise HTTPException(status_code=500, detail="embedding failed")

# LangChain兼容路由 - 修复潜在的环路问题
class LangChainEmbReq(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = "bge-local"

@app.post("/embeddings")
@timing_decorator
def langchain_embeddings(req: LangChainEmbReq):
    """
    LangChain兼容的嵌入端点
    为LangChain和其他库提供兼容的embeddings接口
    """
    try:
        # 将输入标准化为字符串列表
        input_texts = [req.input] if isinstance(req.input, str) else req.input
        if not input_texts:
            raise HTTPException(status_code=400, detail="input must be a non-empty string or list of strings")

        # 确保模型已加载
        model_manager.load_model()
        model = model_manager.model

        if model is None:
            raise HTTPException(status_code=503, detail="Model not available")

        # 直接处理嵌入，避免循环调用
        # 检查缓存
        cached_embeddings = []
        texts_to_process = []
        cache_indices = []

        for i, text in enumerate(input_texts):
            cached = embedding_cache.get(text)
            if cached is not None:
                cached_embeddings.append(cached)
                cache_indices.append(i)
            else:
                texts_to_process.append(text)

        # 处理未缓存的文本
        if texts_to_process:
            new_embeddings = model.encode(
                texts_to_process,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=config.batch_size
            )

            # 缓存新结果
            for text, emb in zip(texts_to_process, new_embeddings):
                embedding_cache.set(text, emb)
        else:
            new_embeddings = np.array([])

        # 合并结果
        final_batch_size = len(input_texts)
        if cached_embeddings and new_embeddings.size > 0:
            embedding_dim = new_embeddings.shape[1] if new_embeddings.size > 0 else cached_embeddings[0].shape[0]
            all_embeddings = np.zeros((final_batch_size, embedding_dim))
            all_embeddings[cache_indices] = cached_embeddings
            process_indices = [i for i in range(final_batch_size) if i not in cache_indices]
            all_embeddings[process_indices] = new_embeddings
        elif cached_embeddings:
            all_embeddings = np.array(cached_embeddings)
        else:
            all_embeddings = new_embeddings

        # 归一化处理
        if not isinstance(model, DummyEmbedder):
            all_embeddings = model_manager.normalize_embeddings(all_embeddings)

        # 构建响应数据
        data = []
        for i, embedding in enumerate(all_embeddings):
            data.append({
                "embedding": embedding.tolist(),
                "index": i,
                "object": "embedding"
            })

        # 计算token使用情况（近似）
        total_tokens = sum(len(text.split()) for text in input_texts)

        return {
            "object": "list",
            "data": data,
            "model": req.model,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("LangChain embeddings failed: %s", e)
        raise HTTPException(status_code=500, detail="embedding failed")


if __name__ == "__main__":
    # 预加载模型
    model_manager.load_model()
    
    # 直接将app传入uvicorn
    uvicorn.run(
        app,  # 直接传入 app 对象，而不是模块名字符串
        host="0.0.0.0", 
        port=config.port, 
        log_level="info",
        workers=1  # 由于模型在内存中，建议单worker
    )
