"""Utils module."""

from typing import Any

from tabulate import tabulate

from inatinqperf.utils.profiler import Profiler


def get_table(data: dict[str, Any]) -> str:
    """Return input dict as a nicely formatted table."""
    # Convert values to lists so that `tablulate` works better.
    return tabulate({k: [v] for k, v in data.items()}, headers="keys", tablefmt="github")


__all__ = ["Profiler", "get_table"]
