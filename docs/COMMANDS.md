# BGE嵌入服务 - 终端命令说明

## 快速启动

### Windows批处理脚本

#### 1. 快速启动服务
```batch
# 双击运行或在命令行执行
quick_start.bat
```

#### 2. 启动服务并运行测试
```batch
# 双击运行或在命令行执行
run_service_and_test.bat
```

#### 3. 测试缓存管理API
```batch
# 双击运行或在命令行执行
test_cache_api.bat
```

### PowerShell脚本

#### 基本启动
```powershell
.\start_service.ps1
```

#### 启动并运行测试
```powershell
.\start_service.ps1 -test
```

#### 启动并测试缓存API
```powershell
.\start_service.ps1 -cache
```

#### 指定端口启动
```powershell
.\start_service.ps1 -port 8080
```

#### 查看帮助
```powershell
.\start_service.ps1 -help
```

## 手动命令

### 1. 激活虚拟环境
```batch
.venv\Scripts\activate.bat
```

### 2. 启动服务
```batch
python cherry_bge_service.py
```

### 3. 运行测试
```batch
pytest
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

## 使用curl测试API

### 健康检查
```bash
curl http://127.0.0.1:7860/health
```

### 获取模型信息
```bash
curl http://127.0.0.1:7860/model-info
```

### 嵌入文本
```bash
curl -X POST http://127.0.0.1:7860/embed -H "Content-Type: application/json" -d "{\"texts\": [\"测试文本\"], \"normalize\": true}"
```

### 获取缓存统计
```bash
curl http://127.0.0.1:7860/cache/stats
```

### 清空缓存
```bash
curl -X POST http://127.0.0.1:7860/cache/clear
```

### 保存缓存
```bash
curl -X POST http://127.0.0.1:7860/cache/save
```

## 环境变量配置

| 变量名 | 默认值 | 描述 |
|--------|--------|------|
| BGE_MODEL | `D:\ai\models\baai\bge-small-zh-v1.5` | 模型路径 |
| BGE_CACHE_DIR | `D:\ai\models\baai\bge-small-zh-v1.5-cache` | 缓存目录 |
| BGE_PORT | 7860 | 服务端口 |
| BGE_HOST | 127.0.0.1 | 服务主机 |
| BGE_DEVICE | cuda | 设备选择 (cpu/gpu/auto) |
| BGE_BATCH_SIZE | 32 | 批处理大小 |
| BGE_MAX_SEQ_LENGTH | 512 | 最大序列长度 |

### 设置环境变量（Windows CMD）
```batch
set BGE_PORT=8080
set BGE_DEVICE=cpu
```

### 设置环境变量（PowerShell）
```powershell
$env:BGE_PORT = "8080"
$env:BGE_DEVICE = "cpu"
```

## 缓存功能说明

### 缓存特性
- **持久化存储**：缓存自动保存到文件，重启后可恢复
- **LRU策略**：使用最近最少使用算法淘汰旧数据
- **过期机制**：默认24小时过期，可配置
- **线程安全**：支持并发访问
- **批量操作**：支持批量读写提高性能

### 缓存统计信息
```json
{
  "size": 100,              // 当前缓存条目数
  "max_size": 10000,        // 最大缓存容量
  "hits": 1500,             // 缓存命中次数
  "misses": 500,            // 缓存未命中次数
  "evictions": 10,          // 淘汰次数
  "expired": 5,             // 过期次数
  "hit_rate": 0.75,         // 命中率
  "ttl_hours": 24           // 过期时间（小时）
}
```

## 文件说明

| 文件 | 描述 |
|------|------|
| `cherry_bge_service.py` | 主服务文件 |
| `test_bge_service.py` | 测试脚本 |
| `quick_start.bat` | 快速启动脚本 |
| `run_service_and_test.bat` | 启动并测试脚本 |
| `test_cache_api.bat` | 缓存API测试脚本 |
| `start_service.ps1` | PowerShell启动脚本 |
| `requirements.txt` | 依赖列表 |