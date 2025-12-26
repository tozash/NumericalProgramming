import sys

def log(step, message):
    """Prints a formatted log message."""
    print(f"[{step}] {message}")
    sys.stdout.flush()
