"""
运行所有测试并生成覆盖率报告的脚本
"""
import subprocess
import sys
import os

def run_coverage_tests():
    """运行所有测试并生成覆盖率报告"""
    print("正在运行覆盖率测试...")
    
    # 首先运行直接函数调用的测试来生成覆盖率数据
    print("\n1. 运行直接函数调用测试以生成覆盖率数据...")
    result = subprocess.run([
        sys.executable,
        "-m",
        "pytest",
        "tect/test_bge_coverage.py",
        "--cov=cherry_bge_service",
        "--cov-report=html:coverage_report",
        "--cov-report=term-missing",
        "-q"
    ])
    
    if result.returncode != 0:
        print("直接函数调用测试失败")
        return result.returncode
    
    print("\n2. API测试跳过（需要先启动服务）")
    print("提示：要运行API测试，请先启动服务：")
    print("  python cherry_bge_service.py")
    print("  或者在另一个终端中运行: uvicorn cherry_bge_service:app --reload")
    
    print("\n覆盖率报告已生成在: coverage_report/index.html")
    print("覆盖率测试完成！")
    
    return result.returncode

if __name__ == "__main__":
    exit_code = run_coverage_tests()
    sys.exit(exit_code)
