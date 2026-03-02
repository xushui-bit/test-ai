# 项目上下文：BGE嵌入服务

## 项目类型
这是一个基于Python的本地嵌入服务项目，使用FastAPI框架为BGE（BAAI General Embedding）模型提供API接口，支持多种API格式。

## 项目概述
本项目是一个本地BGE（BAAI General Embedding）嵌入服务，为本地模型提供嵌入功能。服务支持多种API格式，包括自定义API、OpenAI兼容API和LangChain兼容API。项目旨在提供一个离线可用的文本嵌入服务，支持中文文本处理。

## 核心功能
- **多格式API支持**：支持自定义API、OpenAI兼容API和LangChain兼容API
- **本地模型**：使用本地存储的BGE模型，支持离线运行
- **模型缓存**：内置嵌入缓存机制，提高重复请求的响应速度
- **设备自动检测**：支持CPU/GPU自动检测和配置
- **批处理支持**：支持批量文本嵌入处理
- **归一化处理**：支持嵌入向量的归一化

## 主要文件
- `cherry_bge_service.py`：主服务文件，包含FastAPI应用和所有API端点
- `requirements.txt`：项目依赖列表
- `models/baai/bge-small-zh-v1.5/`：本地BGE模型文件目录

## 技术栈
- **后端框架**：FastAPI
- **模型库**：sentence-transformers, torch
- **Web服务器**：uvicorn
- **其他依赖**：numpy, transformers, safetensors, huggingface_hub

## 模型信息
- **模型名称**：bge-small-zh-v1.5（中文模型）
- **模型路径**：D:\ai\models\baai\bge-small-zh-v1.5
- **缓存目录**：D:\ai\models\baai\bge-small-zh-v1.5-cache

## 配置选项
服务支持通过环境变量进行配置：
- `BGE_MODEL`：模型路径（默认：D:\ai\models\baai\bge-small-zh-v1.5）
- `BGE_CACHE_DIR`：缓存目录（默认：D:\ai\models\baai\bge-small-zh-v1.5-cache）
- `BGE_PORT`：服务端口（默认：7860）
- `BGE_HOST`：服务主机（默认：127.0.0.1）
- `BGE_DEVICE`：设备选择（默认：gpu，可选：cpu, gpu, auto）
- `BGE_BATCH_SIZE`：批处理大小（默认：32）
- `BGE_MAX_SEQ_LENGTH`：最大序列长度（默认：512）

## API端点
- `GET /health`：健康检查
- `GET /model-info`：获取模型信息
- `POST /embed`：主要嵌入端点（EnhancedReq/EnhancedResp格式）
- `POST /embed_legacy`：兼容旧版API
- `POST /v1/embeddings`：OpenAI兼容API
- `POST /embeddings`：LangChain兼容API

## 启动和运行
要启动服务，可以运行以下命令：
```
python cherry_bge_service.py
```

或者使用uvicorn：
```
uvicorn cherry_bge_service:app --host 0.0.0.0 --port 7860
```

## 开发惯例
- 使用Pydantic进行请求/响应模型验证
- 包含计时装饰器来监控函数执行时间
- 实现了模型管理器来处理模型加载和管理
- 使用日志记录系统来跟踪服务运行状态
- 包含错误处理和异常捕获机制