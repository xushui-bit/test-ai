"""
BGE嵌入服务API测试
测试所有API端点的功能
"""
import pytest
import requests
import json
import time
import numpy as np
from typing import List, Dict, Any

BASE_URL = "http://127.0.0.1:7860"

TEST_CASES = {
    "single_chinese": "这是一个测试文本",
    "multiple_chinese": [
        "人工智能是未来的发展方向",
        "机器学习是AI的子集",
        "深度学习使用神经网络"
    ],
    "single_english": "This is a test text in English",
    "multiple_english": [
        "The quick brown fox jumps over the lazy dog",
        "Python is a popular programming language",
        "FastAPI is a modern web framework"
    ]
}


class TestHealthAndModelInfo:
    """测试健康检查和模型信息端点"""

    def test_health_endpoint(self):
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "ok"

    def test_model_info_endpoint(self):
        response = requests.get(f"{BASE_URL}/model-info", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "source" in data
        assert "device" in data
        assert "embedding_dim" in data
        assert "max_seq_length" in data


class TestEmbedEndpoint:
    """测试主要嵌入端点"""

    def test_embed_single_chinese_text(self):
        payload = {"texts": [TEST_CASES["single_chinese"]], "normalize": True}
        response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == 1
        assert data["count"] == 1

    def test_embed_multiple_chinese_text(self):
        payload = {"texts": TEST_CASES["multiple_chinese"], "normalize": True}
        response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        assert response.status_code == 200
        data = response.json()
        assert len(data["embeddings"]) == len(TEST_CASES["multiple_chinese"])

    def test_embed_single_english_text(self):
        payload = {"texts": [TEST_CASES["single_english"]], "normalize": True}
        response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        assert response.status_code == 200

    def test_embed_multiple_english_text(self):
        payload = {"texts": TEST_CASES["multiple_english"], "normalize": True}
        response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        assert response.status_code == 200
        data = response.json()
        assert len(data["embeddings"]) == len(TEST_CASES["multiple_english"])

    def test_embed_without_normalization(self):
        payload = {"texts": [TEST_CASES["single_chinese"]], "normalize": False}
        response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        assert response.status_code == 200
        assert response.json()["normalized"] == False

    def test_embed_with_batch_size(self):
        payload = {"texts": TEST_CASES["multiple_chinese"], "normalize": True, "batch_size": 2}
        response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        assert response.status_code == 200
        data = response.json()
        assert len(data["embeddings"]) == len(TEST_CASES["multiple_chinese"])

    def test_embed_performance(self):
        payload = {"texts": [TEST_CASES["single_chinese"]], "normalize": True}
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        elapsed_time = time.time() - start_time
        assert response.status_code == 200
        assert elapsed_time < 5.0


class TestLegacyAPI:
    """测试旧版API兼容性"""

    def test_embed_legacy(self):
        payload = {"texts": TEST_CASES["multiple_chinese"]}
        response = requests.post(f"{BASE_URL}/embed_legacy", json=payload, timeout=30)
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data


class TestOpenAICompatibility:
    """测试OpenAI兼容性API"""

    def test_openai_embeddings_multiple(self):
        payload = {"input": TEST_CASES["multiple_chinese"], "model": "bge-local"}
        response = requests.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=30)
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) == len(TEST_CASES["multiple_chinese"])

    def test_openai_embeddings_single(self):
        payload = {"input": TEST_CASES["single_chinese"], "model": "bge-local"}
        response = requests.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=30)
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data["data"][0]


class TestLangChainCompatibility:
    """测试LangChain兼容性API"""

    def test_langchain_embeddings_multiple(self):
        payload = {"input": TEST_CASES["multiple_chinese"], "model": "bge-local"}
        response = requests.post(f"{BASE_URL}/embeddings", json=payload, timeout=30)
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) == len(TEST_CASES["multiple_chinese"])


class TestCacheAPI:
    """测试缓存管理API"""

    def test_cache_stats(self):
        response = requests.get(f"{BASE_URL}/cache/stats", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "size" in data
        assert "hit_rate" in data

    def test_cache_clear(self):
        response = requests.post(f"{BASE_URL}/cache/clear", timeout=10)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_cache_save(self):
        response = requests.post(f"{BASE_URL}/cache/save", timeout=10)
        assert response.status_code == 200

    def test_cache_remove_expired(self):
        response = requests.post(f"{BASE_URL}/cache/remove-expired", timeout=10)
        assert response.status_code == 200


class TestErrorHandling:
    """测试错误处理"""

    def test_empty_texts_error(self):
        payload = {"texts": [], "normalize": True}
        response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        assert response.status_code == 422

    def test_invalid_json(self):
        response = requests.post(
            f"{BASE_URL}/embed",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        assert response.status_code in [400, 422]

    def test_too_many_texts_error(self):
        payload = {"texts": ["测试文本"] * 1001, "normalize": True}
        response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        assert response.status_code == 422


class TestEmbeddingQuality:
    """测试嵌入质量"""

    def test_embeddings_are_arrays(self):
        payload = {"texts": [TEST_CASES["single_chinese"]], "normalize": True}
        response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        assert response.status_code == 200
        embedding = response.json()["embeddings"][0]
        assert isinstance(embedding, list)
        assert len(embedding) > 0

    def test_normalized_embeddings_have_unit_norm(self):
        payload = {"texts": [TEST_CASES["single_chinese"]], "normalize": True}
        response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        assert response.status_code == 200
        embedding = np.array(response.json()["embeddings"][0])
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01


class TestCacheEffectiveness:
    """测试缓存效果"""

    def test_cache_consistency(self):
        test_text = "测试缓存一致性的文本"
        payload = {"texts": [test_text], "normalize": True}
        
        response1 = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        assert response1.status_code == 200
        
        response2 = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        assert response2.status_code == 200
        
        emb1 = np.array(response1.json()["embeddings"][0])
        emb2 = np.array(response2.json()["embeddings"][0])
        assert np.allclose(emb1, emb2)


class TestBatchProcessing:
    """测试批量处理"""

    @pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
    def test_different_batch_sizes(self, batch_size):
        texts = [f"这是第{i}个测试文本" for i in range(min(batch_size, 10))]
        payload = {"texts": texts, "normalize": True, "batch_size": batch_size}
        response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        assert response.status_code == 200
        assert len(response.json()["embeddings"]) == len(texts)


@pytest.fixture(scope="session")
def service_available():
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def test_service_available(service_available):
    assert service_available, f"Service not available at {BASE_URL}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
