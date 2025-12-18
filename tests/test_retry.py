"""
retry.py 功能快速测试
运行: python tests/test_retry_simple.py
"""
import sys
sys.path.insert(0, "/Users/bisuv/Developer/PycharmProjects/smart-search-assistant")

from src.utils import retry_with_backoff,safe_call,CircuitBreaker


print("=" * 40)
print("测试1: retry_with_backoff")


call_count = 0

@retry_with_backoff(max_retries=2, base_delay=0.5)
def unstable_api():
    global call_count
    call_count += 1
    print(f"  第{call_count}次调用...")
    if call_count < 2:
        raise Exception("模拟失败")
    return "成功!"
result = unstable_api()


# ===== 测试2: safe_call =====
print("=" * 40)
print("测试2: safe_call")
def risky_func():
    raise Exception("出错了")
result = safe_call(risky_func, default="默认值")
print(f"出错时返回: {result}\n")

# ===== 测试3: 熔断器 =====
print("=" * 40)
print("测试3: CircuitBreaker")
breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=5)
@breaker
def bad_service():
    raise Exception("服务挂了")
for i in range(3):
    try:
        bad_service()
    except Exception as e:
        print(f"  调用{i+1}: {type(e).__name__} - {e}")
print(f"熔断器状态: {breaker.state}")