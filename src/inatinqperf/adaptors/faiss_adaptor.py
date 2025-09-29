"""FAISS vector database adaptor."""

import enum
from collections.abc import Sequence

import faiss
import numpy as np
from loguru import logger

from inatinqperf.adaptors.base import VectorDatabase


class Metric(enum.Enum):
    """Enum for metrics used to compute vector similarity.

    More details about FAISS metrics can be found here: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
    """

    INNER_PRODUCT = 0  # maximum inner product search
    COSINE = 1  # Cosine distance
    L2 = 2  # Euclidean L2 distance


class Faiss(VectorDatabase):
    """Base class for FAISS vector database."""

    def __init__(self, dim: int, metric: str = "ip") -> None:
        super().__init__()

        self.dim = dim
        self.metric = metric.lower()

        self.index = None

    def upsert(self, ids: np.ndarray, x: np.ndarray) -> None:
        """Upsert vectors with given IDs."""
        self.index.remove_ids(faiss.IDSelectorArray(ids.astype(np.int64)))
        self.index.add_with_ids(x.astype(np.float32), ids.astype(np.int64))

    def delete(self, ids: Sequence[int]) -> None:
        """Delete vectors with given IDs."""
        arr = np.asarray(list(ids), dtype=np.int64)
        self.index.remove_ids(faiss.IDSelectorArray(arr))

    def drop_index(self) -> None:
        """Drop the index."""
        self.index = None


class FaissFlat(Faiss):
    """FAISS vector database with Flat index."""

    def __init__(self, dim: int, metric: str = "ip", **params) -> None:  # noqa: ARG002
        """Initialize FAISS Flat index."""
        super().__init__(dim, metric)

        base = faiss.IndexFlatIP(dim) if self.metric in ("ip", "cosine") else faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIDMap2(base)

    def train_index(self, x_train: np.ndarray) -> None:
        """Train the index with given vectors. No-op for Flat index."""

    def search(self, q: np.ndarray, topk: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Search for top-k nearest neighbors."""
        kwargs.pop("nprobe", None)  # not used
        return self.index.search(q.astype(np.float32), topk)

    def stats(self) -> dict[str, object]:
        """Return index statistics."""
        return {
            "ntotal": int(self.index.ntotal),
            "kind": "flat",
            "metric": self.metric,
        }


def _unwrap_to_ivf(base: faiss.Index) -> faiss.Index | None:
    """Return the IVF index inside a composite FAISS index, or None if not found.

    Works across FAISS builds with/without extract_index_ivf.
    """
    # Try the official helper first
    if hasattr(faiss, "extract_index_ivf"):
        try:
            ivf = faiss.extract_index_ivf(base)
            if ivf is not None:
                return ivf
        except Exception:
            logger.warning("[FAISS] Warning: extract_index_ivf failed")

    # Fallback: walk .index fields until we find .nlist
    node = base
    visited = 0
    while node is not None and visited < 5:  # noqa: PLR2004
        if hasattr(node, "nlist"):  # IVF layer
            return node
        node = getattr(node, "index", None)
        visited += 1
    return None


class FaissIVFPQ(Faiss):
    """FAISS vector database with IVF-PQ index."""

    def __init__(
        self,
        dim: int,
        metric: str = "ip",
        nlist: int = 32768,
        m: int = 64,
        nbits: int = 8,
        nprobe: int = 32,
    ) -> None:
        """Initialize FAISS IVF-PQ index."""
        super().__init__(dim, metric)

        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe

        # Build a robust composite index via index_factory
        desc = f"OPQ{self.m},IVF{nlist},PQ{self.m}x{self.nbits}"
        self.metric_type = faiss.METRIC_INNER_PRODUCT if self.metric in ("ip", "cosine") else faiss.METRIC_L2
        base = faiss.index_factory(dim, desc, self.metric_type)
        self.index = faiss.IndexIDMap2(base)

    def train_index(self, x_train: np.ndarray) -> None:
        """Train the index with given vectors."""
        x_train = x_train.astype(np.float32, copy=False)
        n = x_train.shape[0]

        # If dataset is smaller than nlist, rebuild with reduced nlist
        ivf = _unwrap_to_ivf(self.index.index)
        if ivf is not None and hasattr(ivf, "nlist"):
            current_nlist = ivf.nlist

            # Since FAISS hardcodes the minimum number
            # of clustering points to 39, we make sure
            # to set the effective nlist accordingly.
            effective_nlist = max(1, min(current_nlist, int(np.floor(n / 39))))

            if effective_nlist != current_nlist:
                # Recreate with smaller nlist to avoid training failures

                # NOTE: OPQ always uses 2^8 centroids, as this is hardcoded in FAISS (https://github.com/facebookresearch/faiss/blob/3b14dad6d9ac48a0764e5ba01a45bca1d7d738ee/faiss/VectorTransform.cpp#L1068)
                # Hence we check for the effective nlist against 2^8 to decide whether to use OPQ.
                if effective_nlist < 2**8:
                    desc = f"IVF{effective_nlist},PQ{self.m}x{self.nbits}"
                else:
                    desc = f"OPQ{self.m},IVF{effective_nlist},PQ{self.m}x{self.nbits}"

                base = faiss.index_factory(self.dim, desc, self.metric_type)
                self.index = faiss.IndexIDMap2(base)
                ivf = _unwrap_to_ivf(self.index.index)

        # Train if needed
        if self.index is not None and not self.index.is_trained:
            self.index.train(x_train)

        # Set nprobe (if we have IVF)
        ivf = _unwrap_to_ivf(self.index.index)
        if ivf is not None and hasattr(ivf, "nprobe"):
            # Clamp nprobe reasonably based on nlist if available
            nlist = int(getattr(ivf, "nlist", max(1, self.nprobe)))
            ivf.nprobe = min(self.nprobe, max(1, nlist))

    def search(self, q: np.ndarray, topk: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Search for top-k nearest neighbors. Supports runtime nprobe override."""
        q = q.astype(np.float32, copy=False)
        # Runtime override for nprobe
        ivf = _unwrap_to_ivf(self.index.index)
        if ivf is not None and hasattr(ivf, "nprobe"):
            ivf.nprobe = int(kwargs.get("nprobe", self.nprobe))
        return self.index.search(q, topk)

    def stats(self) -> dict[str, object]:
        """Return index statistics."""
        ivf = _unwrap_to_ivf(self.index.index) if self.index is not None else None
        return {
            "ntotal": int(self.index.ntotal) if self.index is not None else 0,
            "kind": "ivfpq",
            "metric": self.metric,
            "nlist": int(getattr(ivf, "nlist", -1)) if ivf is not None else None,
            "nprobe": int(getattr(ivf, "nprobe", -1)) if ivf is not None else None,
        }
