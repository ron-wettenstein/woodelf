import time
from functools import wraps

def time_accumulator(func):
    total_time = 0.0
    call_count = 0

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal total_time, call_count
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        total_time += elapsed
        call_count += 1
        return result

    # expose stats
    wrapper.total_time = lambda: total_time
    wrapper.call_count = lambda: call_count
    wrapper.avg_time = lambda: total_time / call_count if call_count else 0.0
    wrapper.reset = lambda: _reset()

    def _reset():
        nonlocal total_time, call_count
        total_time = 0.0
        call_count = 0

    return wrapper