import time
from functools import wraps
from typing import TypeVar
import os

T = TypeVar('T')


def timing(fn: T) -> T:
    """Time a function."""
    assert callable(fn)

    _avtivated = os.environ.get("TRITON_TIMING", "0") == "1"

    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        results = fn(*args, **kwargs)
        end = time.time()
        elapsed_time = end - start
        if _avtivated:
            print(f'  -- {fn.__name__} : {elapsed_time=}, sec')
        return results

    return wrapper


class TimeRegion:
    _avtivated = os.environ.get("TRITON_TIMING", "0") == "1"

    def __init__(self, annotation):
        self._annotation = annotation
        self._start = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        elapsed_time = time.time() - self._start
        if TimeRegion._avtivated:
            print(f'  -- {self._annotation} : {elapsed_time=}, sec')
