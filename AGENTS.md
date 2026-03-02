# Agent Coding Guidelines

## Project Overview

This is a BGE Embedding Service built with FastAPI. It provides embedding services for local models with support for multiple API formats: custom API, OpenAI-compatible API, and LangChain-compatible API.

## Build/Lint/Test Commands

### Running Tests

Run all tests:
```bash
pytest
```

Run a single test file:
```bash
pytest tests/test_api.py
```

Run a single test function:
```bash
pytest tests/test_api.py::TestHealthAndModelInfo::test_health_endpoint -v
```

Run tests with coverage:
```bash
pytest --cov=cherry_bge_service --cov-report=html:coverage_report --cov-report=term-missing
```

Run tests without coverage (faster):
```bash
pytest -v --no-cov
```

### Starting the Service

Start the service:
```bash
python cherry_bge_service.py
```

Or use PowerShell script:
```powershell
.\start_service.ps1
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| BGE_MODEL | `D:\ai\models\baai\bge-small-zh-v1.5` | Model path |
| BGE_CACHE_DIR | `D:\ai\models\baai\bge-small-zh-v1.5-cache` | Cache directory |
| BGE_PORT | 7860 | Service port |
| BGE_HOST | 127.0.0.1 | Service host |
| BGE_DEVICE | cuda | Device (cpu/gpu/auto) |
| BGE_BATCH_SIZE | 32 | Batch size |
| BGE_MAX_SEQ_LENGTH | 512 | Max sequence length |

## Code Style Guidelines

### Naming Conventions

- **Classes**: PascalCase (e.g., `EmbeddingCache`, `ModelManager`)
- **Functions/Variables**: snake_case (e.g., `load_model`, `cache_dir`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_SIZE`)
- **Private methods**: Prefix with underscore (e.g., `_load_cache`)

### Type Hints

Always use type hints for function parameters and return types:

```python
def encode(self, texts, show_progress_bar=False, convert_to_numpy=True, batch_size=None) -> np.ndarray:
    # ...
```

Use `typing` module for complex types:
```python
from typing import List, Optional, Union, Dict, Any

def process(texts: List[str], options: Optional[Dict[str, Any]] = None) -> List[float]:
    # ...
```

### Imports

Organize imports in the following order:
1. Standard library (os, logging, time, etc.)
2. Third-party libraries (numpy, torch, fastapi, etc.)
3. Local project imports

```python
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
```

### Error Handling

- Use `try/except` blocks with specific exception types
- Re-raise HTTP exceptions in FastAPI routes
- Log exceptions with appropriate levels
- Return meaningful error messages to clients

```python
try:
    result = model.encode(texts)
except Exception as e:
    log.exception("Failed to compute embeddings: %s", e)
    raise HTTPException(status_code=500, detail="嵌入计算失败")
```

### Pydantic Models

Use Pydantic v2 with `field_validator` for validation:

```python
class EnhancedReq(BaseModel):
    texts: List[str]
    normalize: bool = True
    batch_size: Optional[int] = None
    show_progress: bool = False

    @field_validator('texts')
    @classmethod
    def texts_non_empty(cls, v):
        if not v or len(v) == 0:
            raise ValueError('文本列表不能为空')
        if len(v) > 1000:
            raise ValueError('文本数量过多，每次请求最多1000个')
        return v
```

### FastAPI Routes

- Use `response_model` for type-safe responses
- Add docstrings for all endpoints
- Use appropriate HTTP methods (GET for queries, POST for mutations)
- Handle async with `async def` or sync with `def`

```python
@app.post("/embed", response_model=EnhancedResp)
@timing_decorator
def embed(req: EnhancedReq):
    """
    主要的嵌入端点
    处理文本列表并返回嵌入向量
    """
    # implementation
```

### Thread Safety

Use threading locks for shared state:

```python
class EmbeddingCache:
    def __init__(self, max_size=10000, cache_dir=None, ttl_hours=24):
        # ...
        self.lock = threading.RLock()
    
    def get(self, text):
        with self.lock:
            # thread-safe operations
```

### Decorators

Use `@wraps` from functools to preserve function metadata:

```python
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        log.info(f"Function {func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper
```

### Logging

Configure logging and use appropriate levels:

```python
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bge_service")

log.info("Starting up embedding service...")
log.warning("CUDA requested but not available, falling back to CPU")
log.exception("Failed to compute embeddings: %s", e)
```

### File Organization

- Main service logic in `cherry_bge_service.py`
- Tests in `tests/test_*.py`
- Configuration files: `setup.cfg`, `pytest.ini`
- Scripts: `.bat` for Windows batch, `.ps1` for PowerShell

### Testing Guidelines

- Use pytest classes to group related tests
- Use descriptive test names: `test_<what_is_being_tested>`
- Test both success and failure cases
- Use fixtures for shared setup
- Include performance tests where appropriate

```python
class TestEmbedEndpoint:
    """测试主要嵌入端点"""

    def test_embed_single_chinese_text(self):
        """测试单个中文文本嵌入"""
        payload = {"texts": ["测试文本"], "normalize": True}
        response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        assert response.status_code == 200
```

### Project Structure

```
d:\ai\
├── cherry_bge_service.py    # Main service
├── tests/                   # Test directory
│   ├── test_api.py         # API tests
│   └── test_performance.py # Performance tests
├── models/                  # Model files
│   └── baai/              # BGE model
├── requirements.txt         # Dependencies
├── pytest.ini              # Pytest config
├── setup.cfg               # Coverage config
└── .vscode/settings.json  # VS Code settings
```

### Running the Application

Before running tests, ensure the service is running:

```bash
python cherry_bge_service.py
```

Then in another terminal:

```bash
pytest
```
