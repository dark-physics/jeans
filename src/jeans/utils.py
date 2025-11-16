import time
from functools import wraps


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        filename = func.__code__.co_filename
        print(f"{func.__name__} (in {filename}) took {end - start:.3f} seconds.")
        return result

    return wrapper
