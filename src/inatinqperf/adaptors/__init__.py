"""Adaptor registry for vector databases."""

from inatinqperf.adaptors.faiss_adaptor import FaissFlat, FaissIVFPQ
from inatinqperf.adaptors.qdrant_adaptor import Qdrant
from inatinqperf.adaptors.weaviate_adaptor import Weaviate

VECTORDBS = {
    "faiss.flat": FaissFlat,
    "faiss.ivfpq": FaissIVFPQ,
    "qdrant.hnsw": Qdrant,
    "weaviate.hnsw": Weaviate,
}
