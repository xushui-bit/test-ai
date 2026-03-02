# BGE Embedding Service

基于 FastAPI 的 BGE 嵌入服务，为本地模型提供嵌入功能。

## 功能特性

- **多API兼容**: 支持自定义API、OpenAI兼容API、LangChain兼容API
- **嵌入缓存**: LRU缓存策略，支持持久化和过期机制
- **模型管理**: 自动设备检测（CPU/GPU），支持批量处理

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
python cherry_bge_service.py
```

或使用脚本:

```powershell
.\scripts\start_service.ps1
```

## API端点

### 基础API

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/model-info` | 获取模型信息 |
| POST | `/embed` | 主要嵌入端点 |
| POST | `/embed_legacy` | 旧版API |
| POST | `/v1/embeddings` | OpenAI兼容API |
| POST | `/embeddings` | LangChain兼容API |

### 缓存管理API

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/cache/stats` | 获取缓存统计信息 |
| POST | `/cache/clear` | 清空缓存 |
| POST | `/cache/save` | 保存缓存到文件 |
| POST | `/cache/remove-expired` | 移除过期条目 |

## 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| BGE_MODEL | `models/baai/bge-small-zh-v1.5` | 模型路径 |
| BGE_CACHE_DIR | `models/baai/bge-small-zh-v1.5-cache` | 缓存目录 |
| BGE_PORT | 7860 | 服务端口 |
| BGE_HOST | 127.0.0.1 | 服务主机 |
| BGE_DEVICE | cuda | 设备 (cpu/gpu/auto) |
| BGE_BATCH_SIZE | 32 | 批处理大小 |
| BGE_MAX_SEQ_LENGTH | 512 | 最大序列长度 |

## 项目结构

```
.
├── cherry_bge_service.py    # 主服务入口
├── requirements.txt         # 依赖
├── pytest.ini              # pytest配置
├── setup.cfg               # coverage配置
├── AGENTS.md              # Agent编码规范
├── .vscode/               # VS Code配置
├── docs/                  # 文档
│   ├── README.md
│   ├── COMMANDS.md
│   ├── QWEN.md           # Qwen使用说明
│   └── IFLOW.md          # Flow使用说明
├── models/                # 模型文件
│   └── baai/
│       └── bge-small-zh-v1.5/
├── tests/                 # 测试文件
│   ├── test_api.py
│   └── test_performance.py
└── scripts/               # 启动脚本
    ├── start_service.ps1
    ├── quick_start.bat
    ├── run_service_and_test.bat
    └── test_cache_api.bat
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
pytest tests/test_performance.py -v
```

## 相关文档

- [COMMANDS.md](docs/COMMANDS.md) - 详细命令说明
- [QWEN.md](docs/QWEN.md) - Qwen集成说明
- [IFLOW.md](docs/IFLOW.md) - Flow集成说明
