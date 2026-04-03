# BGE Embedding Service

基于 FastAPI 的本地 BGE 中文嵌入服务，为本地模型提供文本嵌入功能。

## 模型信息

### 默认模型

| 属性 | 值 |
|------|-----|
| **模型名称** | BAAI/bge-small-zh-v1.5 |
| **模型路径** | `D:\ai\models\baai\bge-small-zh-v1.5` |
| **嵌入维度** | 512 |
| **最大序列长度** | 512 tokens |
| **语言** | 中文 |
| **框架** | Sentence Transformers |

### 模型说明

- **开发者**：北京智源人工智能研究院 (BAAI)
- **版本**：v1.5
- **规模**：small (约 24M 参数)
- **优化方向**：针对中文文本的语义理解和嵌入生成进行了优化

### 支持的模型配置

通过环境变量可以自定义模型配置：

| 环境变量 | 默认值 | 描述 |
|----------|--------|------|
| BGE_MODEL | `D:\ai\models\baai\bge-small-zh-v1.5` | 模型文件路径 |
| BGE_CACHE_DIR | `D:\ai\models\baai\bge-small-zh-v1.5-cache` | 缓存目录 |
| BGE_DEVICE | cuda | 运行设备 (cpu/gpu/auto) |
| BGE_BATCH_SIZE | 32 | 批处理大小 |
| BGE_MAX_SEQ_LENGTH | 512 | 最大序列长度 |

### 设备支持

- **CPU 模式**：默认模式，适用于无 GPU 环境
- **GPU 模式**：需要 NVIDIA GPU 和 CUDA 支持，性能提升约 5-10 倍
- **自动检测**：设置 `BGE_DEVICE=auto` 可自动选择最佳设备

### 替换模型

如需使用其他 BGE 模型（如 `bge-base-zh-v1.5` 或 `bge-large-zh-v1.5`）：

1. 下载模型到本地目录
2. 设置环境变量：
   ```bash
   set BGE_MODEL=D:\ai\models\baai\bge-base-zh-v1.5
   ```
3. 重启服务

## 特性

- **本地部署**：完全离线运行，保护数据隐私
- **多 API 兼容**：支持自定义 API、OpenAI 兼容 API、LangChain 兼容 API
- **嵌入缓存**：LRU 缓存策略，支持持久化和过期机制
- **模型管理**：自动设备检测（CPU/GPU），支持批量处理
- **中文优化**：使用 bge-small-zh-v1.5 模型，针对中文文本嵌入优化

## 适用场景

- 语义搜索
- 文本相似度计算
- RAG（检索增强生成）
- 知识库问答
- 文本聚类

## 前置要求

| 要求 | 最低配置 | 推荐配置 |
|------|----------|----------|
| Python | 3.11+ | 3.11+ |
| RAM | 4GB | 8GB+ |
| GPU | 可选 | NVIDIA GPU + 4GB VRAM |
| 磁盘 | 500MB | 1GB+ |

## 快速开始

### 1. 克隆项目

```bash
git clone <仓库地址>
cd <项目名称>
```

### 2. 创建虚拟环境

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 启动服务

```bash
python cherry_bge_service.py
```

服务启动后访问 http://127.0.0.1:7860

## 使用示例

### 基础 API 调用

```bash
curl -X POST http://127.0.0.1:7860/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["你好世界", "人工智能是未来的方向"]}'
```

**响应示例：**
```json
{
  "embeddings": [[0.123, -0.456, ...], [0.789, -0.012, ...]],
  "model_info": {
    "source": "D:\\ai\\models\\baai\\bge-small-zh-v1.5",
    "device": "cpu",
    "embedding_dim": 512,
    "max_seq_length": 512
  },
  "normalized": true,
  "count": 2
}
```

### OpenAI 兼容调用

安装 OpenAI SDK：
```bash
pip install openai
```

Python 代码：
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:7860/v1",
    api_key="dummy"  # 本地服务不需要真实 key
)

response = client.embeddings.create(
    input="你好世界",
    model="bge-local"
)

print(response.data[0].embedding)
```

### LangChain 集成

安装 LangChain：
```bash
pip install langchain langchain-community
```

Python 代码：
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="models/baai/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'}
)

# 单文本嵌入
query_embedding = embeddings.embed_query("你好世界")
print(len(query_embedding))  # 512

# 多文本嵌入
documents = ["第一个文档", "第二个文档", "第三个文档"]
doc_embeddings = embeddings.embed_documents(documents)
print(len(doc_embeddings))  # 3
```

### 在 RAG 中使用

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 加载文档
loader = TextLoader("your_document.txt")
documents = loader.load()

# 分割文本
splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = splitter.split_documents(documents)

# 创建向量数据库
embeddings = HuggingFaceEmbeddings(model_name="models/baai/bge-small-zh-v1.5")
vectorstore = FAISS.from_documents(docs, embeddings)

# 相似度搜索
query = "你的问题"
docs = vectorstore.similarity_search(query)
```

## API 端点

### 基础 API

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/model-info` | 获取模型信息 |
| POST | `/embed` | 主要嵌入端点 |
| POST | `/embed_legacy` | 旧版 API |
| POST | `/v1/embeddings` | OpenAI 兼容 API |
| POST | `/embeddings` | LangChain 兼容 API |

### 缓存管理 API

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/cache/stats` | 获取缓存统计 |
| POST | `/cache/clear` | 清空缓存 |
| POST | `/cache/save` | 保存缓存到文件 |
| POST | `/cache/remove-expired` | 移除过期条目 |

### 请求/响应格式

#### POST /embed

**请求：**
```json
{
  "texts": ["文本1", "文本2"],
  "normalize": true,
  "batch_size": 32,
  "show_progress": false
}
```

**响应：**
```json
{
  "embeddings": [[...], [...]],
  "model_info": {...},
  "normalized": true,
  "count": 2
}
```

## 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| BGE_MODEL | `D:\ai\models\baai\bge-small-zh-v1.5` | 模型路径 |
| BGE_CACHE_DIR | `D:\ai\models\baai\bge-small-zh-v1.5-cache` | 缓存目录 |
| BGE_PORT | 7860 | 服务端口 |
| BGE_HOST | 127.0.0.1 | 服务主机 |
| BGE_DEVICE | cuda | 设备 (cpu/gpu/auto) |
| BGE_BATCH_SIZE | 32 | 批处理大小 |
| BGE_MAX_SEQ_LENGTH | 512 | 最大序列长度 |

**切换到 GPU 模式：**
```bash
# Windows
set BGE_DEVICE=gpu
python cherry_bge_service.py

# Linux/Mac
export BGE_DEVICE=gpu
python cherry_bge_service.py
```

## Docker 部署

### 使用 Docker Compose（推荐）

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 使用 Docker

```bash
# 构建镜像
docker build -t bge-embed-service .

# 运行容器
docker run -d -p 7860:7860 -v ./models:/app/models bge-embed-service
```

访问 http://localhost:7860

## 项目结构

```
.
├── cherry_bge_service.py    # 主服务入口
├── requirements.txt         # 依赖
├── Dockerfile              # Docker 配置
├── docker-compose.yml      # Docker Compose 配置
├── pytest.ini              # pytest 配置
├── README.md               # 项目说明
├── AGENTS.md              # Agent 编码规范
├── .gitignore             # Git 忽略配置
├── docs/                  # 文档
│   ├── COMMANDS.md
│   ├── QWEN.md
│   └── IFLOW.md
├── models/                 # 模型文件
│   └── baai/
│       └── bge-small-zh-v1.5/
├── tests/                 # 测试文件
│   ├── test_api.py
│   └── test_performance.py
└── scripts/               # 启动脚本
    ├── start_service.ps1
    └── ...
```

## 测试

### 运行所有测试

```bash
pytest
```

### 运行单个测试

```bash
pytest tests/test_api.py::TestHealthAndModelInfo::test_health_endpoint -v
```

### 运行性能测试

```bash
python tests/test_performance.py
```

## 性能参考

| 场景 | CPU 模式 | GPU 模式 |
|------|----------|----------|
| 单文本 (100字) | ~50ms | ~10ms |
| 批处理 (32条) | ~500ms | ~50ms |
| 批处理 (100条) | ~1.5s | ~100ms |

实际性能取决于硬件配置和文本长度。

## 常见问题

### 1. 服务启动失败

检查是否端口被占用：
```bash
netstat -ano | findstr :7860
```

### 2. 模型加载失败

确认模型路径正确：
```bash
ls models/baai/bge-small-zh-v1.5/
```

### 3. GPU 不可用

确认 CUDA 是否正确安装：
```python
import torch
print(torch.cuda.is_available())
```

### 4. 内存不足

减少批处理大小：
```bash
set BGE_BATCH_SIZE=16
```

## 相关文档

- [COMMANDS.md](docs/COMMANDS.md) - 详细命令说明
- [QWEN.md](docs/QWEN.md) - Qwen 集成说明
- [IFLOW.md](docs/IFLOW.md) - Flow 集成说明
- [LICENSES.md](LICENSES.md) - 许可证信息

## 许可证

本项目采用 MIT License 开源。使用的模型和依赖均有各自的许可证：

- **BGE 模型**: Apache-2.0
- **FastAPI**: MIT
- **PyTorch**: BSD-3-Clause
- **sentence-transformers**: Apache-2.0

详见 [LICENSES.md](LICENSES.md) 获取完整的许可证信息。
