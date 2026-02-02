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
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

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
    device: str = os.environ.get("BGE_DEVICE", "cuda")
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

    @field_validator('texts')
    @classmethod
    def texts_non_empty(cls, v):
        """验证文本列表非空"""
        if not v or len(v) == 0:
            raise ValueError('文本列表不能为空')
        if len(v) > 1000:  # 添加限制
            raise ValueError('文本数量过多，每次请求最多1000个')
        return v

class EnhancedResp(BaseModel):
    """增强型响应模型"""
    embeddings: List[List[float]]
    model_info: dict
    normalized: bool
    count: int

import pickle
from collections import OrderedDict
from datetime import datetime, timedelta
import threading

class EmbeddingCache:
    """
    嵌入缓存类，提高重复请求的响应速度
    支持持久化存储、LRU策略、过期机制和统计信息
    """
    def __init__(self, max_size=10000, cache_dir=None, ttl_hours=24):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.cache_dir = cache_dir
        self.ttl = timedelta(hours=ttl_hours)
        self.lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0
        }
        
        # 加载持久化缓存
        if cache_dir:
            self.cache_file = os.path.join(cache_dir, "embedding_cache.pkl")
            self._load_cache()

    def get_key(self, text):
        """生成文本的缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text):
        """获取缓存的嵌入向量"""
        key = self.get_key(text)
        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None
            
            entry = self.cache[key]
            
            # 检查是否过期
            if self._is_expired(entry):
                del self.cache[key]
                self.stats["expired"] += 1
                self.stats["misses"] += 1
                return None
            
            # LRU: 移动到末尾
            self.cache.move_to_end(key)
            self.stats["hits"] += 1
            return entry["embedding"]

    def set(self, text, embedding):
        """设置缓存的嵌入向量"""
        key = self.get_key(text)
        with self.lock:
            # 如果已存在，更新并移动到末尾
            if key in self.cache:
                self.cache.move_to_end(key)
            
            # 检查是否需要淘汰
            if len(self.cache) >= self.max_size and key not in self.cache:
                # 淘汰最久未使用的项目
                self.cache.popitem(last=False)
                self.stats["evictions"] += 1
            
            # 存储缓存条目（包含时间戳）
            self.cache[key] = {
                "embedding": embedding,
                "timestamp": datetime.now(),
                "text": text
            }

    def get_batch(self, texts):
        """批量获取缓存的嵌入向量"""
        results = {}
        with self.lock:
            for text in texts:
                key = self.get_key(text)
                if key in self.cache:
                    entry = self.cache[key]
                    if not self._is_expired(entry):
                        self.cache.move_to_end(key)
                        self.stats["hits"] += 1
                        results[key] = entry["embedding"]
                    else:
                        del self.cache[key]
                        self.stats["expired"] += 1
                        self.stats["misses"] += 1
                else:
                    self.stats["misses"] += 1
        return results

    def set_batch(self, texts, embeddings):
        """批量设置缓存的嵌入向量"""
        with self.lock:
            for text, emb in zip(texts, embeddings):
                self.set(text, emb)

    def _is_expired(self, entry):
        """检查缓存条目是否过期"""
        if self.ttl is None:
            return False
        return datetime.now() - entry["timestamp"] > self.ttl

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "expired": 0
            }

    def get_stats(self):
        """获取缓存统计信息"""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"],
                "expired": self.stats["expired"],
                "hit_rate": hit_rate,
                "ttl_hours": self.ttl.total_seconds() / 3600 if self.ttl else None
            }

    def _load_cache(self):
        """从文件加载缓存"""
        if not os.path.exists(self.cache_file):
            return
        
        try:
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
                # 加载缓存数据
                self.cache = OrderedDict(data.get("cache", {}))
                # 加载统计信息
                self.stats = data.get("stats", {
                    "hits": 0,
                    "misses": 0,
                    "evictions": 0,
                    "expired": 0
                })
                log.info(f"Loaded {len(self.cache)} cached embeddings from {self.cache_file}")
        except Exception as e:
            log.warning(f"Failed to load cache from {self.cache_file}: {e}")

    def save_cache(self):
        """保存缓存到文件"""
        if not self.cache_dir:
            return
        
        try:
            data = {
                "cache": dict(self.cache),
                "stats": self.stats,
                "saved_at": datetime.now().isoformat()
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
            log.info(f"Saved {len(self.cache)} cached embeddings to {self.cache_file}")
        except Exception as e:
            log.warning(f"Failed to save cache to {self.cache_file}: {e}")

    def remove_expired(self):
        """移除所有过期的缓存条目"""
        with self.lock:
            expired_keys = [k for k, v in self.cache.items() if self._is_expired(v)]
            for key in expired_keys:
                del self.cache[key]
            return len(expired_keys)

embedding_cache = EmbeddingCache(max_size=10000, cache_dir=config.cache_dir, ttl_hours=24)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期上下文管理器，用于处理启动和关闭事件"""
    # 启动阶段
    log.info("Starting up embedding service...")
    model_manager.load_model()
    yield
    # 关闭阶段
    log.info("Shutting down embedding service...")
    embedding_cache.save_cache()

app = FastAPI(title="bge-embed-service", lifespan=lifespan)

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

        # 验证CUDA是否可用
        if device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"

        log.info(f"Using device: {device}")

        try:
            if SentenceTransformer is None:
                # sentence-transformers库在此环境中不可用
                log.warning("sentence-transformers未安装或导入失败，使用虚拟嵌入器")
                self.model = DummyEmbedder(dim=384)
                self.model_info = {
                    "source": "dummy",
                    "device": "cpu",
                    "embedding_dim": 384,
                    "max_seq_length": 512
                }
                return self.model

            # 加载实际模型
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
        raise HTTPException(status_code=503, detail="嵌入模型不可用")

    try:
        # 批量检查缓存
        cached_results = embedding_cache.get_batch(req.texts)
        texts_to_process = []
        cache_indices = []
        cached_embeddings_map = {}

        for i, text in enumerate(req.texts):
            key = embedding_cache.get_key(text)
            if key in cached_results:
                cache_indices.append(i)
                cached_embeddings_map[i] = cached_results[key]
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

            # 批量缓存新结果
            embedding_cache.set_batch(texts_to_process, new_embeddings)
        else:
            new_embeddings = np.array([])

        # 合并结果
        final_batch_size = len(req.texts)
        if cached_embeddings_map and new_embeddings.size > 0:
            embedding_dim = new_embeddings.shape[1]
            all_embeddings = np.zeros((final_batch_size, embedding_dim))
            for i, emb in cached_embeddings_map.items():
                all_embeddings[i] = emb
            process_indices = [i for i in range(final_batch_size) if i not in cache_indices]
            all_embeddings[process_indices] = new_embeddings
        elif cached_embeddings_map:
            all_embeddings = np.zeros((final_batch_size, next(iter(cached_embeddings_map.values())).shape[0]))
            for i, emb in cached_embeddings_map.items():
                all_embeddings[i] = emb
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
        raise HTTPException(status_code=500, detail="嵌入计算失败")

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
            raise HTTPException(status_code=400, detail="输入必须是非空字符串或字符串列表")

        # 确保模型已加载
        model_manager.load_model()
        model = model_manager.model
        if model is None:
            raise HTTPException(status_code=503, detail="嵌入模型不可用")

        # 批量检查缓存
        cached_results = embedding_cache.get_batch(texts)
        texts_to_process = []
        cache_indices = []
        cached_embeddings_map = {}

        for i, text in enumerate(texts):
            key = embedding_cache.get_key(text)
            if key in cached_results:
                cache_indices.append(i)
                cached_embeddings_map[i] = cached_results[key]
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

            # 批量缓存新结果
            embedding_cache.set_batch(texts_to_process, new_embeddings)
        else:
            new_embeddings = np.array([])

        # 合并结果
        final_batch_size = len(texts)
        if cached_embeddings_map and new_embeddings.size > 0:
            embedding_dim = new_embeddings.shape[1]
            all_embeddings = np.zeros((final_batch_size, embedding_dim))
            for i, emb in cached_embeddings_map.items():
                all_embeddings[i] = emb
            process_indices = [i for i in range(final_batch_size) if i not in cache_indices]
            all_embeddings[process_indices] = new_embeddings
        elif cached_embeddings_map:
            all_embeddings = np.zeros((final_batch_size, next(iter(cached_embeddings_map.values())).shape[0]))
            for i, emb in cached_embeddings_map.items():
                all_embeddings[i] = emb
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
        raise HTTPException(status_code=500, detail="嵌入计算失败")

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
            raise HTTPException(status_code=400, detail="输入必须是非空字符串或字符串列表")

        # 确保模型已加载
        model_manager.load_model()
        model = model_manager.model

        if model is None:
            raise HTTPException(status_code=503, detail="模型不可用")

        # 批量检查缓存
        cached_results = embedding_cache.get_batch(input_texts)
        texts_to_process = []
        cache_indices = []
        cached_embeddings_map = {}

        for i, text in enumerate(input_texts):
            key = embedding_cache.get_key(text)
            if key in cached_results:
                cache_indices.append(i)
                cached_embeddings_map[i] = cached_results[key]
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

            # 批量缓存新结果
            embedding_cache.set_batch(texts_to_process, new_embeddings)
        else:
            new_embeddings = np.array([])

        # 合并结果
        final_batch_size = len(input_texts)
        if cached_embeddings_map and new_embeddings.size > 0:
            embedding_dim = new_embeddings.shape[1]
            all_embeddings = np.zeros((final_batch_size, embedding_dim))
            for i, emb in cached_embeddings_map.items():
                all_embeddings[i] = emb
            process_indices = [i for i in range(final_batch_size) if i not in cache_indices]
            all_embeddings[process_indices] = new_embeddings
        elif cached_embeddings_map:
            all_embeddings = np.zeros((final_batch_size, next(iter(cached_embeddings_map.values())).shape[0]))
            for i, emb in cached_embeddings_map.items():
                all_embeddings[i] = emb
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
        raise HTTPException(status_code=500, detail="嵌入计算失败")


# 缓存管理 API
@app.get("/cache/stats")
def get_cache_stats():
    """获取缓存统计信息"""
    return embedding_cache.get_stats()

@app.post("/cache/clear")
def clear_cache():
    """清空缓存"""
    embedding_cache.clear()
    return {"status": "ok", "message": "缓存已清空"}

@app.post("/cache/save")
def save_cache():
    """保存缓存到文件"""
    embedding_cache.save_cache()
    return {"status": "ok", "message": "缓存已保存"}

@app.post("/cache/remove-expired")
def remove_expired_cache():
    """移除过期的缓存条目"""
    removed_count = embedding_cache.remove_expired()
    return {
        "status": "ok",
        "message": f"已移除 {removed_count} 个过期条目",
        "removed_count": removed_count
    }


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
