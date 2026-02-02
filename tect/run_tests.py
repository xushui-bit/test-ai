"""
BGE嵌入服务测试入口点
运行所有pytest测试
"""
import subprocess
import sys
import os


def run_tests():
    """运行所有测试"""
    print("正在运行BGE嵌入服务的pytest测试...")

    # 运行所有测试
    result = subprocess.run([
        sys.executable,
        "-m",
        "pytest",
        "test_bge_service_pytest.py",
        "-v",
        "--tb=short"
    ], cwd=os.path.dirname(__file__))

    if result.returncode == 0:
        print("\n✓ 所有测试通过！")
    else:
        print(f"\n✗ 测试失败，返回码: {result.returncode}")

    return result.returncode


def run_tests_quiet():
    """运行测试，输出更简洁"""
    print("正在运行BGE嵌入服务的pytest测试...")

    # 运行所有测试，输出更简洁
    result = subprocess.run([
        sys.executable,
        "-m",
        "pytest",
        "test_bge_service_pytest.py",
        "--tb=no",
        "-q"
    ], cwd=os.path.dirname(__file__))

    if result.returncode == 0:
        print("\n✓ 所有测试通过！")
        print(f"运行了 {get_test_count()} 个测试")
    else:
        print(f"\n✗ 测试失败，返回码: {result.returncode}")

    return result.returncode


def get_test_count():
    """获取测试总数"""
    try:
        result = subprocess.run([
            sys.executable,
            "-m",
            "pytest",
            "test_bge_service_pytest.py",
            "--collect-only",
            "-q"
        ], cwd=os.path.dirname(__file__), capture_output=True, text=True)

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'collected' in line and 'items' in line:
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            return int(part)
        return 0
    except:
        return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quiet":
        exit_code = run_tests_quiet()
    else:
        exit_code = run_tests()

    sys.exit(exit_code)
