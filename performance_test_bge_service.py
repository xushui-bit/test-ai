"""
BGE嵌入服务性能测试脚本
测试服务在各种负载下的性能表现
"""

import requests
import time
import json
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import statistics
import os

# 服务配置
BASE_URL = os.environ.get("BGE_PERF_TEST_URL", "http://127.0.0.1:7860")
PORT = os.environ.get("BGE_PORT", "7860")

# 测试用例
TEST_CASES = {
    "短中文文本": ["这是一个测试文本"] * 10,
    "长中文文本": ["人工智能是未来的发展方向，机器学习是AI的重要子集，深度学习使用神经网络进行复杂模式识别和预测。"] * 5,
    "短英文文本": ["This is a test text in English"] * 10,
    "长英文文本": ["Artificial intelligence is the future direction of development, machine learning is an important subset of AI, and deep learning uses neural networks for complex pattern recognition and prediction."] * 5,
    "混合文本": [
        "人工智能是未来的发展方向",
        "Machine learning is an important subset of AI",
        "这是另一个测试文本",
        "Deep learning uses neural networks"
    ] * 3
}

class PerformanceTester:
    """性能测试器类"""
    
    def __init__(self):
        self.results = {}
    
    def print_section(self, title: str):
        """打印测试章节标题"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
    
    def print_success(self, message: str):
        """打印成功消息"""
        print(f"✓ {message}")
    
    def print_error(self, message: str):
        """打印错误消息"""
        print(f"✗ {message}")
    
    def print_info(self, message: str):
        """打印信息消息"""
        print(f"  {message}")
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('status') == 'ok'
        except Exception as e:
            self.print_error(f"服务健康检查失败: {e}")
            return False
    
    def test_single_request(self, texts: List[str], api_type: str = "custom") -> Tuple[float, Dict[str, Any]]:
        """测试单个请求的性能"""
        start_time = time.time()
        
        if api_type == "custom":
            # 自定义API
            payload = {
                "texts": texts,
                "normalize": True,
                "show_progress": False
            }
            response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        elif api_type == "openai":
            # OpenAI兼容API
            payload = {
                "input": texts,
                "model": "bge-local"
            }
            response = requests.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=30)
        elif api_type == "langchain":
            # LangChain兼容API
            payload = {
                "input": texts,
                "model": "bge-local"
            }
            response = requests.post(f"{BASE_URL}/embeddings", json=payload, timeout=30)
        
        response.raise_for_status()
        response_time = time.time() - start_time
        response_data = response.json()
        
        return response_time, response_data
    
    def test_concurrent_requests(self, texts: List[str], num_requests: int, api_type: str = "custom") -> List[float]:
        """测试并发请求性能"""
        response_times = []
        
        def make_request():
            try:
                start_time = time.time()
                
                if api_type == "custom":
                    payload = {
                        "texts": texts,
                        "normalize": True,
                        "show_progress": False
                    }
                    response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
                elif api_type == "openai":
                    payload = {
                        "input": texts,
                        "model": "bge-local"
                    }
                    response = requests.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=30)
                elif api_type == "langchain":
                    payload = {
                        "input": texts,
                        "model": "bge-local"
                    }
                    response = requests.post(f"{BASE_URL}/embeddings", json=payload, timeout=30)
                
                response.raise_for_status()
                response_time = time.time() - start_time
                response_times.append(response_time)
            except Exception as e:
                self.print_error(f"并发请求失败: {e}")
        
        # 创建线程池并发出并发请求
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            for future in as_completed(futures):
                future.result()  # 等待所有请求完成
        
        return response_times
    
    def test_throughput(self, texts: List[str], duration: int = 30, api_type: str = "custom") -> Dict[str, Any]:
        """测试吞吐量"""
        start_time = time.time()
        response_times = []
        completed_requests = 0
        
        def make_request():
            try:
                start = time.time()
                
                if api_type == "custom":
                    payload = {
                        "texts": texts,
                        "normalize": True,
                        "show_progress": False
                    }
                    response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
                elif api_type == "openai":
                    payload = {
                        "input": texts,
                        "model": "bge-local"
                    }
                    response = requests.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=30)
                elif api_type == "langchain":
                    payload = {
                        "input": texts,
                        "model": "bge-local"
                    }
                    response = requests.post(f"{BASE_URL}/embeddings", json=payload, timeout=30)
                
                response.raise_for_status()
                response_time = time.time() - start
                return response_time
            except Exception as e:
                self.print_error(f"吞吐量测试请求失败: {e}")
                return None
        
        # 在指定持续时间内持续发送请求
        while time.time() - start_time < duration:
            response_time = make_request()
            if response_time is not None:
                response_times.append(response_time)
                completed_requests += 1
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
            
            throughput = completed_requests / duration  # 请求/秒
            
            return {
                "completed_requests": completed_requests,
                "duration": duration,
                "throughput_rps": throughput,
                "avg_response_time": avg_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time,
                "p95_response_time": p95_response_time,
                "p99_response_time": p99_response_time,
                "response_times": response_times
            }
        else:
            return {
                "completed_requests": 0,
                "duration": duration,
                "throughput_rps": 0,
                "avg_response_time": 0,
                "min_response_time": 0,
                "max_response_time": 0,
                "p95_response_time": 0,
                "p99_response_time": 0,
                "response_times": []
            }
    
    def test_cache_performance(self, texts: List[str]) -> Dict[str, Any]:
        """测试缓存性能"""
        # 第一次请求（无缓存）
        start_time = time.time()
        payload = {
            "texts": texts,
            "normalize": True
        }
        response1 = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
        response1.raise_for_status()
        uncached_time = time.time() - start_time
        
        # 多次请求（有缓存）
        cached_times = []
        for _ in range(5):
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
            response.raise_for_status()
            cached_times.append(time.time() - start_time)
        
        avg_cached_time = statistics.mean(cached_times)
        cache_improvement = (uncached_time - avg_cached_time) / uncached_time * 100
        
        return {
            "uncached_time": uncached_time,
            "avg_cached_time": avg_cached_time,
            "cache_improvement": cache_improvement,
            "cache_hits": len(cached_times)
        }
    
    def test_batch_size_performance(self, base_texts: List[str]) -> Dict[str, Any]:
        """测试不同批量大小的性能"""
        results = {}
        
        batch_sizes = [1, 5, 10, 20, 32]  # 使用默认的批处理大小32作为上限
        
        for batch_size in batch_sizes:
            if batch_size > len(base_texts):
                continue
                
            texts = base_texts[:batch_size]
            start_time = time.time()
            
            payload = {
                "texts": texts,
                "normalize": True
            }
            
            response = requests.post(f"{BASE_URL}/embed", json=payload, timeout=30)
            response.raise_for_status()
            
            response_time = time.time() - start_time
            avg_per_text = response_time / batch_size
            
            results[batch_size] = {
                "total_time": response_time,
                "avg_per_text": avg_per_text,
                "batch_size": batch_size
            }
        
        return results
    
    def run_single_request_tests(self):
        """运行单个请求性能测试"""
        self.print_section("单个请求性能测试")
        
        for test_name, texts in TEST_CASES.items():
            try:
                # 测试自定义API
                response_time, response_data = self.test_single_request(texts, "custom")
                self.print_success(f"自定义API - {test_name}: {response_time:.4f}秒, {len(texts)}个文本")
                
                # 测试OpenAI兼容API
                response_time, response_data = self.test_single_request(texts, "openai")
                self.print_success(f"OpenAI兼容API - {test_name}: {response_time:.4f}秒, {len(texts)}个文本")
                
                # 测试LangChain兼容API
                response_time, response_data = self.test_single_request(texts, "langchain")
                self.print_success(f"LangChain兼容API - {test_name}: {response_time:.4f}秒, {len(texts)}个文本")
                
            except Exception as e:
                self.print_error(f"单个请求测试失败 {test_name}: {e}")
    
    def run_concurrent_tests(self):
        """运行并发请求性能测试"""
        self.print_section("并发请求性能测试")
        
        texts = TEST_CASES["短中文文本"][:5]  # 使用5个短文本
        
        for num_requests in [5, 10, 20]:
            for api_type in ["custom", "openai", "langchain"]:
                try:
                    response_times = self.test_concurrent_requests(texts, num_requests, api_type)
                    
                    if response_times:
                        avg_time = statistics.mean(response_times)
                        min_time = min(response_times)
                        max_time = max(response_times)
                        
                        self.print_success(
                            f"{api_type} API - {num_requests}并发请求: "
                            f"平均{avg_time:.4f}秒, "
                            f"最小{min_time:.4f}秒, "
                            f"最大{max_time:.4f}秒"
                        )
                    else:
                        self.print_error(f"{api_type} API - {num_requests}并发请求: 无响应时间数据")
                        
                except Exception as e:
                    self.print_error(f"并发测试失败 {api_type}, {num_requests}请求: {e}")
    
    def run_throughput_tests(self):
        """运行吞吐量测试"""
        self.print_section("吞吐量测试")
        
        texts = TEST_CASES["短中文文本"][:3]  # 使用3个短文本
        
        for api_type in ["custom", "openai", "langchain"]:
            try:
                result = self.test_throughput(texts, duration=20, api_type=api_type)
                
                self.print_success(
                    f"{api_type} API吞吐量测试结果: "
                    f"完成{result['completed_requests']}个请求, "
                    f"吞吐量{result['throughput_rps']:.2f} RPS, "
                    f"平均响应{result['avg_response_time']:.4f}秒"
                )
                
            except Exception as e:
                self.print_error(f"吞吐量测试失败 {api_type}: {e}")
    
    def run_cache_performance_test(self):
        """运行缓存性能测试"""
        self.print_section("缓存性能测试")
        
        texts = ["测试缓存效果的文本"]
        
        try:
            result = self.test_cache_performance(texts)
            
            self.print_success(
                f"缓存性能: 未缓存{result['uncached_time']:.4f}秒, "
                f"缓存后{result['avg_cached_time']:.4f}秒, "
                f"性能提升{result['cache_improvement']:.2f}%"
            )
            
        except Exception as e:
            self.print_error(f"缓存性能测试失败: {e}")
    
    def run_batch_size_test(self):
        """运行批量大小性能测试"""
        self.print_section("批量大小性能测试")
        
        base_texts = TEST_CASES["短中文文本"]
        
        try:
            results = self.test_batch_size_performance(base_texts)
            
            for batch_size, data in results.items():
                self.print_success(
                    f"批量大小{batch_size}: 总时间{data['total_time']:.4f}秒, "
                    f"每文本平均{data['avg_per_text']:.4f}秒"
                )
                
        except Exception as e:
            self.print_error(f"批量大小性能测试失败: {e}")
    
    def generate_performance_report(self):
        """生成性能报告"""
        self.print_section("性能测试报告")
        
        # 获取缓存统计信息
        try:
            cache_stats_response = requests.get(f"{BASE_URL}/cache/stats", timeout=10)
            cache_stats = cache_stats_response.json()
            
            self.print_info("缓存统计信息:")
            self.print_info(f"  - 缓存大小: {cache_stats.get('size', 0)}/{cache_stats.get('max_size', 0)}")
            self.print_info(f"  - 命中率: {cache_stats.get('hit_rate', 0):.2%}")
            self.print_info(f"  - 命中次数: {cache_stats.get('hits', 0)}")
            self.print_info(f"  - 未命中次数: {cache_stats.get('misses', 0)}")
        except Exception as e:
            self.print_error(f"获取缓存统计信息失败: {e}")
    
    def run_all_tests(self):
        """运行所有性能测试"""
        self.print_section("BGE嵌入服务性能测试套件")
        
        if not self.health_check():
            self.print_error("服务健康检查失败，无法继续性能测试")
            return
        
        print("开始性能测试...")
        
        # 运行各个性能测试
        self.run_single_request_tests()
        self.run_concurrent_tests()
        self.run_throughput_tests()
        self.run_cache_performance_test()
        self.run_batch_size_test()
        self.generate_performance_report()
        
        self.print_section("性能测试完成")


def main():
    tester = PerformanceTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()