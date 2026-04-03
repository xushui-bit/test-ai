# 许可证信息

## 项目许可证

本项目采用 **MIT License** 开源。详见根目录 `LICENSE` 文件。

## 模型许可证

| 模型 | 开发者 | 许可证 | 链接 |
|------|--------|--------|------|
| BAAI/bge-small-zh-v1.5 | 北京智源人工智能研究院 (BAAI) | Apache-2.0 | [Hugging Face](https://huggingface.co/BAAI/bge-small-zh-v1.5) |

## 依赖许可证

### 核心依赖

| 依赖 | 许可证 | 类型 | 链接 |
|------|--------|------|------|
| FastAPI | MIT | Web 框架 | [GitHub](https://github.com/tiangolo/fastapi) |
| uvicorn | BSD-3-Clause | ASGI 服务器 | [GitHub](https://github.com/encode/uvicorn) |
| sentence-transformers | Apache-2.0 | 嵌入框架 | [GitHub](https://github.com/UKPLab/sentence-transformers) |
| PyTorch (torch) | BSD-3-Clause | 深度学习框架 | [GitHub](https://github.com/pytorch/pytorch) |
| safetensors | Apache-2.0 | 张量存储 | [GitHub](https://github.com/huggingface/safetensors) |
| huggingface_hub | Apache-2.0 | 模型库客户端 | [GitHub](https://github.com/huggingface/huggingface_hub) |

### 开发依赖

| 依赖 | 许可证 | 类型 | 链接 |
|------|--------|------|------|
| pytest | MIT | 测试框架 | [GitHub](https://github.com/pytest-dev/pytest) |
| pytest-cov | MIT | 覆盖率插件 | [GitHub](https://github.com/pytest-dev/pytest-cov) |

### 间接依赖

| 依赖 | 许可证 | 类型 |
|------|--------|------|
| numpy | BSD-3-Clause | 数值计算 |
| transformers | Apache-2.0 | NLP 框架 |
| tokenizers | Apache-2.0 | 分词器 |
| scipy | BSD-3-Clause | 科学计算 |
| scikit-learn | BSD-3-Clause | 机器学习 |
| pillow | HPND | 图像处理 |

### 运行时

| 组件 | 许可证 | 说明 |
|------|--------|------|
| Python 3.11 | PSF License v2 | Python 运行时 |
| python:3.11-slim (Docker) | 多种开源许可证 | Debian slim 基础镜像 |

## 许可证兼容性

本项目使用的所有许可证均为**宽松型（permissive）许可证**，包括：

- **MIT License** - 最宽松，仅需保留版权声明
- **Apache-2.0** - 宽松，包含专利授权条款
- **BSD-3-Clause** - 宽松，需保留版权声明和免责声明

这些许可证彼此兼容，并且与商业使用完全兼容。

## 使用说明

### 商业使用

本项目可以用于商业用途，但需遵守以下条件：

1. 保留所有依赖的版权声明
2. 遵守各依赖的许可证条款
3. BGE 模型使用需遵守 Apache-2.0 许可证

### 修改和分发

- 可以修改和重新分发本项目代码
- 需保留原始版权声明和许可证文件
- 修改后的代码需明确标注

## 第三方许可证

本项目依赖的第三方库的完整许可证文本，请参阅各项目的官方仓库：

- [FastAPI LICENSE](https://github.com/tiangolo/fastapi/blob/master/LICENSE)
- [uvicorn LICENSE](https://github.com/encode/uvicorn/blob/master/LICENSE.md)
- [PyTorch LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE)
- [sentence-transformers LICENSE](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE)
- [safetensors LICENSE](https://github.com/huggingface/safetensors/blob/main/LICENSE)
- [pytest LICENSE](https://github.com/pytest-dev/pytest/blob/main/LICENSE)

## 免责声明

本文件提供的许可证信息仅供参考。如有法律疑问，请咨询专业法律人士。
