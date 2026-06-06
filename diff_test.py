import time

def greet(name):
    """Return a time-stamped greeting."""
    now = time.strftime("%H:%M:%S")
    return f"[{now}] Hello, {name}! Welcome to the Codex diff test."

if __name__ == "__main__":
    name = input("Enter your name: ") or "World"
    print(greet(name))
    print(greet("Codex"))
