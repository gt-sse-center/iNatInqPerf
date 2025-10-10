"""Enum of similarity search metrics."""

import enum


class Metric(str, enum.Enum):
    """Enum for metrics used to compute vector similarity.

    Inherit from `str` so we can get human-readable metric names.

    More details about FAISS metrics can be found here: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
    """

    INNER_PRODUCT = "ip"  # inner product
    COSINE = "cosine"  # Cosine distance
    L2 = "l2"  # Euclidean L2 distance
    MANHATTAN = "l1"  # Taxicab/L1 distance
