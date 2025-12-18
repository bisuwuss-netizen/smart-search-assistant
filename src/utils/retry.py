"""
重试和容错机制

提供装饰器和工具函数，用于处理：
1. API 调用失败重试
2. 超时处理
3. 降级策略
"""
import time
import functools
from typing import Callable, Any, Optional, Type, Tuple


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    带指数退避的重试装饰器

    Args:
        max_retries: 最大重试次数
        base_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
        exponential_base: 指数基数
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数

    使用示例：
        @retry_with_backoff(max_retries=3, exceptions=(APIError, TimeoutError))
        def call_api():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        print(f"  ⚠️ 重试 {attempt + 1}/{max_retries}，等待 {delay:.1f}s...")

                        if on_retry:
                            on_retry(e, attempt + 1)

                        time.sleep(delay)
                    else:
                        print(f"  ❌ 达到最大重试次数 ({max_retries})，放弃")

            raise last_exception

        return wrapper
    return decorator


def safe_call(
    func: Callable,
    *args,
    default: Any = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_error: Optional[Callable[[Exception], None]] = None,
    **kwargs
) -> Any:
    """
    安全调用函数，出错时返回默认值

    Args:
        func: 要调用的函数
        default: 出错时的默认返回值
        exceptions: 要捕获的异常类型
        on_error: 出错时的回调函数

    使用示例：
        result = safe_call(risky_function, default=[], on_error=log_error)
    """
    try:
        return func(*args, **kwargs)
    except exceptions as e:
        if on_error:
            on_error(e)
        return default


class CircuitBreaker:
    """
    熔断器模式

    当连续失败次数达到阈值时，暂时停止调用，避免雪崩效应。

    使用示例：
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        @breaker
        def call_external_api():
            ...
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions

        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 检查熔断器状态
            if self.state == "open":
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = "half-open"
                    print("  🔄 熔断器进入半开状态，尝试恢复...")
                else:
                    raise CircuitBreakerOpen(
                        f"熔断器开启中，请等待 {self.recovery_timeout - (time.time() - self.last_failure_time):.0f}s"
                    )

            try:
                result = func(*args, **kwargs)
                # 调用成功，重置计数器
                self.last_failure_time = 0
                self.failure_count = 0
                self.state = "closed"
                return result

            except self.expected_exceptions as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    print(f"  🔴 熔断器开启！连续失败 {self.failure_count} 次")

                raise e

        return wrapper

    def reset(self):
        """手动重置熔断器"""
        self.failure_count = 0
        self.state = "closed"


class CircuitBreakerOpen(Exception):
    """熔断器开启时抛出的异常"""
    pass


# ============ 预配置的重试策略 ============

# LLM API 调用重试
llm_retry = retry_with_backoff(
    max_retries=3,
    base_delay=2.0,
    exceptions=(Exception,),  # 可以替换为具体的 API 异常
    on_retry=lambda e, n: print(f"  ⚠️ LLM 调用失败: {e}")
)

# 搜索 API 重试
search_retry = retry_with_backoff(
    max_retries=2,
    base_delay=1.0,
    max_delay=10.0,
    exceptions=(Exception,),
    on_retry=lambda e, n: print(f"  ⚠️ 搜索失败: {e}")
)

# 向量库操作重试
vector_retry = retry_with_backoff(
    max_retries=2,
    base_delay=0.5,
    exceptions=(Exception,),
    on_retry=lambda e, n: print(f"  ⚠️ 向量操作失败: {e}")
)
