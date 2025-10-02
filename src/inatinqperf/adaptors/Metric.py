"""Metrics."""

import enum


class Metric(enum.Enum):
    """Enum for metrics used to compute vector similarity.

    More details about metrics can be found here: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
    """

    INNER_PRODUCT = "ip"  # maximum inner product search
    COSINE = "cos"  # Cosine distance
    L2 = "l2"  # Euclidean L2 distance
