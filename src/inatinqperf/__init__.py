"""__init__.py for inatinqperf package."""

try:
    from importlib.metadata import version

    __version__ = version("inatinqperf")
except Exception:
    __version__ = "unknown"
# -----------------------
