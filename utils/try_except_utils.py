# utils/try_except_utils.py

def try_except(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred in function {func.__name__}: {e}")

    return wrapper
