"""Adaptor registry for vector databases."""

from inatinqperf.adaptors.faiss_adaptor import Faiss
from inatinqperf.adaptors.milvus_adaptor import Milvus
from inatinqperf.adaptors.qdrant_adaptor import Qdrant
from inatinqperf.adaptors.weaviate_adaptor import Weaviate

VECTORDBS = {
    "faiss": Faiss,
    "qdrant.hnsw": Qdrant,
    "weaviate.hnsw": Weaviate,
    "milvus": Milvus,
}
