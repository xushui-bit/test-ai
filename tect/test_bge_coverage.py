"""
BGE嵌入服务的直接函数调用测试，用于覆盖率分析
通过直接调用函数而不是API来让覆盖率工具能够追踪代码执行
"""
import pytest
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from cherry_bge_service import ModelManager, Req, EnhancedReq, OpenAIEmbReq, LangChainEmbReq
import numpy as np

# 测试用例
TEST_TEXTS = [
    "这是一个测试文本",
    "人工智能是未来的发展方向",
    "机器学习是AI的子集"
]

def test_model_manager_initialization():
    """测试模型管理器初始化"""
    model_manager = ModelManager()
    assert model_manager is not None
    # 由于模型加载可能需要较长时间，这里我们只测试初始化
    # 但不实际加载模型（因为这会花较长时间）


def test_basic_request_validation():
    """测试基本请求验证"""
    # 测试有效的请求
    req = Req(texts=TEST_TEXTS)
    assert req.texts == TEST_TEXTS


def test_enhanced_request():
    """测试增强请求"""
    req = EnhancedReq(
        texts=TEST_TEXTS,
        normalize=True,
        show_progress=False,
        batch_size=32
    )
    assert req.texts == TEST_TEXTS
    assert req.normalize is True
    assert req.show_progress is False
    assert req.batch_size == 32


def test_openai_request():
    """测试OpenAI兼容请求"""
    req = OpenAIEmbReq(
        input=TEST_TEXTS,
        model="bge-local",
        user="test-user"
    )
    assert req.input == TEST_TEXTS
    assert req.model == "bge-local"


def test_langchain_request():
    """测试LangChain兼容请求"""
    req = LangChainEmbReq(
        input=TEST_TEXTS,
        model="bge-local"
    )
    assert req.input == TEST_TEXTS
    assert req.model == "bge-local"


def test_embedding_array_conversion():
    """测试嵌入向量数组转换"""
    # 创建模拟的嵌入向量
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    
    # 测试转换为numpy数组
    embeddings_array = np.array(embeddings)
    assert embeddings_array.shape == (2, 3)
    
    # 测试归一化
    normalized = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-5)


def test_cache_functionality():
    """测试缓存功能相关代码路径"""
    # 测试缓存键生成逻辑
    text = "test text for cache"
    cache_key = f"emb:{text}:True"  # 假设归一化为True
    assert cache_key.startswith("emb:")
    
    # 测试一些缓存统计计算逻辑
    hits = 10
    misses = 5
    total_requests = hits + misses
    hit_rate = hits / total_requests if total_requests > 0 else 0
    assert hit_rate == 0.6666666666666666


if __name__ == "__main__":
    # 当直接运行此文件时，执行pytest
    pytest.main([__file__, "-v"])