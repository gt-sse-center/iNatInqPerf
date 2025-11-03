"""Utils module."""

from typing import Any

from tabulate import tabulate

from inatinqperf.utils.dataio import export_images, load_huggingface_dataset
from inatinqperf.utils.embed import (
    ImageDatasetWithEmbeddings,
    embed_images,
    embed_text,
    to_huggingface_dataset,
)
from inatinqperf.utils.profiler import Profiler


def get_table(data: dict[str, Any]) -> str:
    """Return input dict as a nicely formatted table."""
    # Convert values to lists so that `tablulate` works better.
    return tabulate({k: [v] for k, v in data.items()}, headers="keys", tablefmt="github")


__all__ = [
    "ImageDatasetWithEmbeddings",
    "Profiler",
    "embed_images",
    "embed_text",
    "export_images",
    "get_table",
    "load_huggingface_dataset",
    "to_huggingface_dataset",
]
