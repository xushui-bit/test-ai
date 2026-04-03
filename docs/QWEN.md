# BGE Embedding Service 项目上下文

## 项目概述
这是一个基于Python的本地嵌入服务项目，使用FastAPI框架为BGE（BAAI General Embedding）模型提供API接口，支持多种API格式。项目旨在提供一个离线可用的文本嵌入服务，特别针对中文文本处理进行了优化。

### 核心功能
- **多格式API支持**：支持自定义API、OpenAI兼容API和LangChain兼容API
- **本地模型**：使用本地存储的BGE模型，支持离线运行
- **模型缓存**：内置嵌入缓存机制，提高重复请求的响应速度
- **设备自动检测**：支持CPU/GPU自动检测和配置
- **批处理支持**：支持批量文本嵌入处理
- **归一化处理**：支持嵌入向量的归一化

### 技术栈
- **后端框架**：FastAPI
- **模型库**：sentence-transformers, torch
- **Web服务器**：uvicorn
- **其他依赖**：numpy, transformers, safetensors, huggingface_hub

## 项目结构
```
D:\ai\
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

## 配置选项
服务支持通过环境变量进行配置：
- `BGE_MODEL`：模型路径（默认：D:\\ai\\models\\baai\\bge-small-zh-v1.5）
- `BGE_CACHE_DIR`：缓存目录（默认：D:\\ai\\models\\baai\\bge-small-zh-v1.5-cache）
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

## 依赖管理
项目依赖定义在`requirements.txt`文件中：
```
fastapi
uvicorn[standard]
sentence-transformers
torch
transformers
safetensors
huggingface_hub
requests
python-dotenv
```

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
- 包含确定性虚拟嵌入器，用于离线测试

## 模型信息
- **模型名称**：bge-small-zh-v1.5（中文模型）
- **模型路径**：D:\\ai\\models\\baai\\bge-small-zh-v1.5
- **缓存目录**：D:\\ai\\models\\baai\\bge-small-zh-v1.5-cache
- **嵌入维度**：384（根据模型配置）
- **最大序列长度**：512

## 特殊功能
- **离线模式**：通过设置环境变量`TRANSFORMERS_OFFLINE=1`和`HF_HUB_OFFLINE=1`，确保使用本地模型
- **缓存机制**：使用MD5哈希作为缓存键，提高重复请求的响应速度
- **虚拟嵌入器**：在无法加载实际模型时，使用基于SHA256哈希的确定性虚拟嵌入器
- **自动设备检测**：当设置为"auto"时，自动检测并使用可用的GPU或CPU