import enum

class Metric(enum.Enum):
    """Enum for metrics used to compute vector similarity.

    More details about metrics can be found here: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
    """

    INNER_PRODUCT = 0  # maximum inner product search
    COSINE = 1  # Cosine distance
    L2 = 2  # Euclidean L2 distance