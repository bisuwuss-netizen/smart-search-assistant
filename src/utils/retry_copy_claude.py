import time
from typing import Callable


class CircuitBreakerOpen(Exception):
    pass


class CircuitBreaker():
    def __init__(
            self,
            failure_threshold: int = 3,
            recovery_timeout: float = 2.5
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.last_failure_time = 0
        self.failure_count = 0
        self.state = "closed"

    def __call__(self, fun: Callable):
        def wrapper(*args, **kwargs):
            print(f"[调用前] 状态: {self.state}, 失败次数: {self.failure_count}")

            if self.state == "open":
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = "half-open"
                    print("  🔄 熔断器进入半开状态，尝试恢复...")
                else:
                    raise CircuitBreakerOpen(
                        f"熔断器开启中，请等待 {self.recovery_timeout - (time.time() - self.last_failure_time):.0f}s"
                    )

            try:
                result = fun(*args, **kwargs)
                print("  ✅ 函数执行成功")
                self.last_failure_time = 0
                self.failure_count = 0
                self.state = "closed"
                return result
            except Exception as e:
                print(f"  ❌ 函数执行失败: {type(e).__name__}")
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold: #当前失败次数大于等于阈值
                    self.state = "open"
                    print(f"  🔴 熔断器开启！连续失败 {self.failure_count} 次")
                raise e
                # ⚠️ 注意：这里没有 raise！

        return wrapper


breaker = CircuitBreaker(recovery_timeout=2)


@breaker
def say_hello(name="zhansan"):
    print(f"    执行 say_hello({name})")
    num = 1 / 0


for i in range(5):
    print(f"\n===== 第 {i + 1} 次调用 =====")
    try:
        say_hello()
    except Exception as e:
        print(f"💥 外层捕获异常: {type(e).__name__}")

    if i == 3:
        print("😴 睡眠 2.5 秒...")
        time.sleep(2.5)