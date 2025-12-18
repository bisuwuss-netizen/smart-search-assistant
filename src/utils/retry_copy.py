import time
from typing import Callable, Any

from src.utils import CircuitBreakerOpen


# def simple_retry(max_retry=3,base_delay=1):
#
#     def decorator(fun:Callable)->Callable:
#
#         def wrapper():

# def decorator(fun:Callable):
#     def wrapper(*args,**kwargs):
#         return fun(*args,**kwargs)
#     return wrapper
#
#
# @decorator
# def my_fun(name="zhangsna",age = 18):
#     print(f"name={name},age={age}")
#
# my_fun()


def my_retry(
        max_retry:int=3,
        base_delay:float=1.0,
        max_delay:float = 2.5,
        exponential_base:float = 2.0,
        max_retries:int = 3
):
    def decorator(fun):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retry + 1):
                try:
                    return fun(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retry:
                        delay = min(max_delay,base_delay * (exponential_base ** attempt))
                        time.sleep(delay)
                    else:
                        print(f"  ❌ 达到最大重试次数 ({max_retries})，放弃")
            raise last_exception
        return wrapper
    return decorator

# @my_retry()



def safe_call(
        fun:Callable,
        *args,
        default:Any = None,
        **kwargs
):
    try:
        return fun(*args,**kwargs)
    except Exception as e:
        return default

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

    def __call__(self, fun:Callable):
        def wrapper(*args,**kwargs):
            # if self.state is "open":

            if self.state == "open":
                #尝试关闭熔断器
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = "half-open"
                    print("  🔄 熔断器进入半开状态，尝试恢复...")
                else:
                    raise CircuitBreakerOpen(
                        f"熔断器开启中，请等待 {self.recovery_timeout - (time.time() - self.last_failure_time):.0f}s"
                    )

            try:
                result = fun(*args,**kwargs)
                self.last_failure_time = 0
                self.failure_count = 0
                self.state = "closed"
                return result
            except Exception as e:
                self.failure_count+=1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    #熔断器断开
                    self.state = "open"
                    print(f"  🔴 熔断器开启！连续失败 {self.failure_count} 次,达到最高上限")
                    raise e
        return wrapper


breaker = CircuitBreaker(recovery_timeout = 2)

@breaker
def say_hello(name = "zhansan"):
    print(name+" over")
    num = 1/0

for i in range(5):
    try:
        say_hello()
    except Exception as e:
        print("出错了")

    if i == 3:
        time.sleep(2.5)

# say_hello = safe_call(say_hello,default="出错了，请检查")
# print(say_hello)