"""__init__.py for adaptors."""

from inatinqperf.adaptors.faiss_adaptor import FaissFlat, FaissIVFPQ
from inatinqperf.adaptors.qdrant_adaptor import Qdrant

VECTORDBS = {
    "faiss.flat": FaissFlat,
    "faiss.ivfpq": FaissIVFPQ,
    "qdrant.hnsw": Qdrant,
}
