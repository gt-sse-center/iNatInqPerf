import os, math, numpy as np, faiss
from typing import Dict, Any, Tuple, Sequence
from .base import VectorBackend

def _metric_to_faiss(metric: str):
    return faiss.METRIC_INNER_PRODUCT if metric.lower() in ("ip","cosine") else faiss.METRIC_L2

class FaissFlat(VectorBackend):
    name = "faiss.flat"
    def __init__(self):
        self.index = None
        self.metric = "ip"
        self.dim = None

    def init(self, dim: int, metric: str, **params):
        self.dim = dim
        self.metric = metric.lower()
        base = faiss.IndexFlatIP(dim) if self.metric in ("ip","cosine") else faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIDMap2(base)

    def train(self, X_train: np.ndarray):  # not needed
        return

    def upsert(self, ids: np.ndarray, X: np.ndarray):
        self.index.remove_ids(faiss.IDSelectorArray(ids.astype("int64")))
        self.index.add_with_ids(X.astype("float32"), ids.astype("int64"))

    def delete(self, ids: Sequence[int]):
        arr = np.asarray(list(ids), dtype="int64")
        self.index.remove_ids(faiss.IDSelectorArray(arr))

    def search(self, Q: np.ndarray, topk: int, **kwargs) -> Tuple[np.ndarray,np.ndarray]:
        return self.index.search(Q.astype("float32"), topk)

    def stats(self) -> Dict[str,Any]:
        return {"ntotal": int(self.index.ntotal), "kind":"flat", "metric": self.metric}

    def drop(self): self.index = None


def _unwrap_to_ivf(base):
    """
    Return the IVF index inside a composite FAISS index, or None if not found.
    Works across FAISS builds with/without extract_index_ivf.
    """
    # Try the official helper first
    if hasattr(faiss, "extract_index_ivf"):
        try:
            ivf = faiss.extract_index_ivf(base)
            if ivf is not None:
                return ivf
        except Exception:
            pass
    # Fallback: walk .index fields until we find .nlist
    node = base
    visited = 0
    while node is not None and visited < 5:
        if hasattr(node, "nlist"):  # IVF layer
            return node
        node = getattr(node, "index", None)
        visited += 1
    return None

class FaissIVFPQ(VectorBackend):
    name = "faiss.ivfpq"
    def __init__(self):
        self.index = None
        self.metric = "ip"
        self.dim = None
        self.nprobe = 32
        self.m = 64
        self.nbits = 8

    def init(self, dim: int, metric: str, **params):
        self.dim = dim
        self.metric = metric.lower()
        nlist = int(params.get("nlist", 32768))
        self.m = int(params.get("m", 64))
        self.nbits = int(params.get("nbits", 8))
        self.nprobe = int(params.get("nprobe", 32))

        # Build a robust composite index via index_factory
        desc = f"OPQ{self.m},IVF{nlist},PQ{self.m}x{self.nbits}"
        metric_type = faiss.METRIC_INNER_PRODUCT if self.metric in ("ip", "cosine") else faiss.METRIC_L2
        base = faiss.index_factory(dim, desc, metric_type)
        self.index = faiss.IndexIDMap2(base)

    def train(self, X_train: np.ndarray):
        X_train = X_train.astype("float32", copy=False)
        n = int(X_train.shape[0])

        # If dataset is smaller than nlist, rebuild with reduced nlist
        ivf = _unwrap_to_ivf(self.index.index)
        if ivf is not None and hasattr(ivf, "nlist"):
            current_nlist = int(ivf.nlist)
            effective_nlist = max(1, min(current_nlist, n))
            if effective_nlist != current_nlist:
                # Recreate with smaller nlist to avoid training failures
                self.init(self.dim, self.metric,
                          nlist=effective_nlist, m=self.m, nbits=self.nbits, nprobe=self.nprobe)
                ivf = _unwrap_to_ivf(self.index.index)

        # Train if needed
        if self.index is not None and not self.index.is_trained:
            self.index.train(X_train)

        # Set nprobe (if we have IVF)
        ivf = _unwrap_to_ivf(self.index.index)
        if ivf is not None and hasattr(ivf, "nprobe"):
            # Clamp nprobe reasonably based on nlist if available
            nlist = int(getattr(ivf, "nlist", max(1, self.nprobe)))
            ivf.nprobe = min(self.nprobe, max(1, nlist))

    def upsert(self, ids: np.ndarray, X: np.ndarray):
        self.index.remove_ids(faiss.IDSelectorArray(ids.astype("int64")))
        self.index.add_with_ids(X.astype("float32"), ids.astype("int64"))

    def delete(self, ids: Sequence[int]):
        arr = np.asarray(list(ids), dtype="int64")
        self.index.remove_ids(faiss.IDSelectorArray(arr))

    def search(self, Q: np.ndarray, topk: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        Q = Q.astype("float32", copy=False)
        # Runtime override for nprobe
        ivf = _unwrap_to_ivf(self.index.index)
        if ivf is not None and hasattr(ivf, "nprobe"):
            ivf.nprobe = int(kwargs.get("nprobe", self.nprobe))
        return self.index.search(Q, topk)

    def stats(self) -> Dict[str, Any]:
        ivf = _unwrap_to_ivf(self.index.index) if self.index is not None else None
        return {
            "ntotal": int(self.index.ntotal) if self.index is not None else 0,
            "kind": "ivfpq",
            "metric": self.metric,
            "nlist": int(getattr(ivf, "nlist", -1)) if ivf is not None else None,
            "nprobe": int(getattr(ivf, "nprobe", -1)) if ivf is not None else None,
        }
    
    def drop(self): self.index = None


BACKENDS = {
  "faiss.flat": FaissFlat,
  "faiss.ivfpq": FaissIVFPQ,
}