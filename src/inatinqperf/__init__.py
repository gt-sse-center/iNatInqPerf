def hello() -> str:
    return "Hello from inatinqperf!"


# noqa: D104

from importlib.metadata import version  # noqa: E402


__version__ = version("inatinqperf")
