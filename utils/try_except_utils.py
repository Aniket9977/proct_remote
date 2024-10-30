# utils/try_except_utils.py

def try_except(func):
    """Decorator to handle exceptions for a function."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred in function {func.__name__}: {e}")
            # You can also log the error if needed.
    return wrapper
