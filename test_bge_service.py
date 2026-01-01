"""
BGE嵌入服务测试脚本
测试所有API端点的功能
"""
import requests
import json
import time
import numpy as np
from typing import List, Dict, Any

# 服务配置
BASE_URL = "http://127.0.0.1:7860"

# 测试用例
TEST_CASES = {
    "单个中文文本": "这是一个测试文本",
    "多个中文文本": [
        "人工智能是未来的发展方向",
        "机器学习是AI的子集",
        "深度学习使用神经网络"
    ],
    "单个英文文本": "This is a test text in English",
    "多个英文文本": [
        "The quick brown fox jumps over the lazy dog",
        "Python is a popular programming language",
        "FastAPI is a modern web framework"
    ]
}


def print_section(title: str):
    """打印测试章节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_success(message: str):
    """打印成功消息"""
    print(f"✓ {message}")


def print_error(message: str):
    """打印错误消息"""
    print(f"✗ {message}")


def print_info(message: str):
    """打印信息消息"""
    print(f"  {message}")


def print_response(response: Dict[str, Any]):
    """打印响应内容"""
    print(json.dumps(response, indent=2, ensure_ascii=False))


def test_health():
    """测试健康检查端点"""
    print_section("测试健康检查端点 (GET /health)")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        
        print_success(f"健康检查成功")
        print_info(f"状态: {data.get('status')}")
        print_info(f"模型已加载: {data.get('model_loaded')}")
        print_info(f"模型信息: {json.dumps(data.get('model_info'), indent=4, ensure_ascii=False)}")
        
        return True
    except Exception as e:
        print_error(f"健康检查失败: {e}")
        return False


def test_model_info():
    """测试模型信息端点"""
    print_section("测试模型信息端点 (GET /model-info)")
    
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        response.raise_for_status()
        data = response.json()
        
        print_success(f"获取模型信息成功")
        print_response(data)
        
        return True
    except Exception as e:
        print_error(f"获取模型信息失败: {e}")
        return False


def test_embed_endpoint(texts: List[str], normalize: bool = True):
    """测试主要嵌入端点"""
    print_section(f"测试主要嵌入端点 (POST /embed)")
    
    payload = {
        "texts": texts,
        "normalize": normalize,
        "show_progress": False
    }
    
    print_info(f"请求数据: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/embed", json=payload)
        elapsed_time = time.time() - start_time
        
        response.raise_for_status()
        data = response.json()
        
        print_success(f"嵌入计算成功 (耗时: {elapsed_time:.4f}秒)")
        print_info(f"嵌入维度: {len(data['embeddings'][0])}")
        print_info(f"文本数量: {data['count']}")
        print_info(f"是否归一化: {data['normalized']}")
        print_info(f"模型信息: {json.dumps(data['model_info'], indent=4, ensure_ascii=False)}")
        
        # 打印前几个嵌入向量的前5个值
        print_info(f"第一个文本的嵌入向量(前5个值): {data['embeddings'][0][:5]}")
        
        # 验证嵌入向量
        if normalize:
            # 验证归一化
            for i, emb in enumerate(data['embeddings']):
                norm = np.linalg.norm(emb)
                print_info(f"文本{i}的L2范数: {norm:.6f}")
                if abs(norm - 1.0) > 0.01:
                    print_error(f"文本{i}的归一化失败，范数为: {norm}")
        
        return True
    except Exception as e:
        print_error(f"嵌入计算失败: {e}")
        if hasattr(e, 'response'):
            print_error(f"响应内容: {e.response.text}")
        return False


def test_embed_legacy(texts: List[str]):
    """测试兼容旧版API"""
    print_section("测试兼容旧版API (POST /embed_legacy)")
    
    payload = {
        "texts": texts
    }
    
    print_info(f"请求数据: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/embed_legacy", json=payload)
        elapsed_time = time.time() - start_time
        
        response.raise_for_status()
        data = response.json()
        
        print_success(f"旧版API嵌入计算成功 (耗时: {elapsed_time:.4f}秒)")
        print_info(f"嵌入维度: {len(data['embeddings'][0])}")
        print_info(f"文本数量: {len(data['embeddings'])}")
        print_info(f"第一个文本的嵌入向量(前5个值): {data['embeddings'][0][:5]}")
        
        return True
    except Exception as e:
        print_error(f"旧版API嵌入计算失败: {e}")
        if hasattr(e, 'response'):
            print_error(f"响应内容: {e.response.text}")
        return False


def test_openai_embeddings(texts: List[str]):
    """测试OpenAI兼容API"""
    print_section("测试OpenAI兼容API (POST /v1/embeddings)")
    
    payload = {
        "input": texts,
        "model": "bge-local"
    }
    
    print_info(f"请求数据: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/v1/embeddings", json=payload)
        elapsed_time = time.time() - start_time
        
        response.raise_for_status()
        data = response.json()
        
        print_success(f"OpenAI兼容API嵌入计算成功 (耗时: {elapsed_time:.4f}秒)")
        print_info(f"响应对象类型: {data.get('object')}")
        print_info(f"使用的模型: {data.get('model')}")
        print_info(f"Token使用情况: {json.dumps(data.get('usage'), indent=4)}")
        print_info(f"嵌入数据数量: {len(data.get('data', []))}")
        
        # 打印第一个嵌入信息
        if data.get('data'):
            first_emb = data['data'][0]
            print_info(f"第一个嵌入对象: {json.dumps(first_emb, indent=4, ensure_ascii=False)}")
            print_info(f"第一个嵌入向量(前5个值): {first_emb['embedding'][:5]}")
        
        return True
    except Exception as e:
        print_error(f"OpenAI兼容API嵌入计算失败: {e}")
        if hasattr(e, 'response'):
            print_error(f"响应内容: {e.response.text}")
        return False


def test_langchain_embeddings(texts: List[str]):
    """测试LangChain兼容API"""
    print_section("测试LangChain兼容API (POST /embeddings)")
    
    payload = {
        "input": texts,
        "model": "bge-local"
    }
    
    print_info(f"请求数据: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/embeddings", json=payload)
        elapsed_time = time.time() - start_time
        
        response.raise_for_status()
        data = response.json()
        
        print_success(f"LangChain兼容API嵌入计算成功 (耗时: {elapsed_time:.4f}秒)")
        print_info(f"响应对象类型: {data.get('object')}")
        print_info(f"使用的模型: {data.get('model')}")
        print_info(f"Token使用情况: {json.dumps(data.get('usage'), indent=4)}")
        print_info(f"嵌入数据数量: {len(data.get('data', []))}")
        
        # 打印第一个嵌入信息
        if data.get('data'):
            first_emb = data['data'][0]
            print_info(f"第一个嵌入对象: {json.dumps(first_emb, indent=4, ensure_ascii=False)}")
            print_info(f"第一个嵌入向量(前5个值): {first_emb['embedding'][:5]}")
        
        return True
    except Exception as e:
        print_error(f"LangChain兼容API嵌入计算失败: {e}")
        if hasattr(e, 'response'):
            print_error(f"响应内容: {e.response.text}")
        return False


def test_single_text_openai():
    """测试OpenAI API的单个文本输入"""
    print_section("测试OpenAI API单个文本输入")
    
    payload = {
        "input": "这是一个单独的测试文本",
        "model": "bge-local"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/v1/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()
        
        print_success(f"单个文本嵌入成功")
        print_info(f"嵌入数据数量: {len(data.get('data', []))}")
        print_info(f"第一个嵌入向量(前5个值): {data['data'][0]['embedding'][:5]}")
        
        return True
    except Exception as e:
        print_error(f"单个文本嵌入失败: {e}")
        return False


def test_cache_effectiveness():
    """测试缓存效果"""
    print_section("测试缓存效果")
    
    test_text = "测试缓存效果的文本"
    
    # 第一次请求（无缓存）
    payload = {
        "texts": [test_text],
        "normalize": True
    }
    
    print_info("第一次请求（无缓存）...")
    start_time = time.time()
    response1 = requests.post(f"{BASE_URL}/embed", json=payload)
    time1 = time.time() - start_time
    response1.raise_for_status()
    
    print_success(f"第一次请求完成 (耗时: {time1:.4f}秒)")
    
    # 第二次请求（有缓存）
    print_info("第二次请求（有缓存）...")
    start_time = time.time()
    response2 = requests.post(f"{BASE_URL}/embed", json=payload)
    time2 = time.time() - start_time
    response2.raise_for_status()
    
    print_success(f"第二次请求完成 (耗时: {time2:.4f}秒)")
    
    # 比较结果
    emb1 = response1.json()['embeddings'][0]
    emb2 = response2.json()['embeddings'][0]
    
    if np.allclose(emb1, emb2):
        print_success("两次请求的嵌入向量完全一致")
        print_info(f"性能提升: {(time1/time2):.2f}倍")
    else:
        print_error("两次请求的嵌入向量不一致")
        return False
    
    return True


def test_batch_size():
    """测试不同批量大小的性能"""
    print_section("测试不同批量大小的性能")
    
    batch_sizes = [1, 5, 10]
    test_texts = ["这是第{}个测试文本".format(i) for i in range(10)]
    
    for batch_size in batch_sizes:
        payload = {
            "texts": test_texts[:batch_size],
            "normalize": True,
            "batch_size": batch_size
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/embed", json=payload)
            elapsed_time = time.time() - start_time
            response.raise_for_status()
            
            print_success(f"批量大小={batch_size}: {elapsed_time:.4f}秒, 每个文本平均{elapsed_time/batch_size:.4f}秒")
        except Exception as e:
            print_error(f"批量大小={batch_size} 测试失败: {e}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("  BGE嵌入服务测试套件")
    print("="*60)
    
    results = []
    
    # 基础测试
    results.append(("健康检查", test_health()))
    results.append(("模型信息", test_model_info()))
    
    # 主要嵌入端点测试
    results.append(("主要端点 - 单个中文文本", test_embed_endpoint([TEST_CASES["单个中文文本"]])))
    results.append(("主要端点 - 多个中文文本", test_embed_endpoint(TEST_CASES["多个中文文本"])))
    results.append(("主要端点 - 单个英文文本", test_embed_endpoint([TEST_CASES["单个英文文本"]])))
    results.append(("主要端点 - 多个英文文本", test_embed_endpoint(TEST_CASES["多个英文文本"])))
    
    # 旧版API测试
    results.append(("旧版API", test_embed_legacy(TEST_CASES["多个中文文本"])))
    
    # OpenAI兼容API测试
    results.append(("OpenAI API - 多个文本", test_openai_embeddings(TEST_CASES["多个中文文本"])))
    results.append(("OpenAI API - 单个文本", test_single_text_openai()))
    
    # LangChain兼容API测试
    results.append(("LangChain API", test_langchain_embeddings(TEST_CASES["多个中文文本"])))
    
    # 性能测试
    results.append(("缓存效果", test_cache_effectiveness()))
    results.append(("批量大小性能", test_batch_size()))
    
    # 打印测试总结
    print_section("测试总结")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print_success("所有测试通过！")
    else:
        print_error(f"有 {total - passed} 个测试失败")


def main():
    for i in range(2):
            run_all_tests()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()