"""__init__ for inatinqperf/utils."""


def hello() -> str:
    """Hello function which is a sample function to show that this package works."""
    return "Hello from inatinqperf!"


from importlib.metadata import version  # noqa: E402

__version__ = version("inatinqperf")
