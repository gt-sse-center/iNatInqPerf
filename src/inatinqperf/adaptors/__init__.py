"""__init__.py for adaptors."""

from inatinqperf.adaptors.faiss_backend import FaissFlat, FaissIVFPQ

VECTORDBS = {
    "faiss.flat": FaissFlat,
    "faiss.ivfpq": FaissIVFPQ,
}
