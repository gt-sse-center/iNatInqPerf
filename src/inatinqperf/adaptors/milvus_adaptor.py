"""Milvus adaptor."""

from collections.abc import Sequence
from enum import Enum
from typing import override

import numpy as np
from loguru import logger
from pymilvus import DataType, MilvusClient, connections, utility

from inatinqperf.adaptors.base import VectorDatabase
from inatinqperf.adaptors.Metric import Metric


class MilvusIndexType(Enum):
    """Enum for various index types supported by Milvus. For more details, see https://milvus.io/docs/index.md?tab=floating."""

    IVF_FLAT = "ivf_flat"
    IVF_SQ8 = "ivf_sq8"
    IVF_PQ = "ivf_pq"
    HNSW = "hnsw"
    HNSW_SQ = "hnsw_sq"
    HNSW_PQ = "hnsw_pq"


class Milvus(VectorDatabase):
    """Milvus vector database."""

    DATABASE_NAME: str = "default"
    COLLECTION_NAME: str = "collection_name"
    SERVER_HOST = "localhost"
    SERVER_PORT = "19530"  # default milvus server port

    @logger.catch
    def __init__(self, dim: int, metric: Metric, index_name: MilvusIndexType) -> None:
        super().__init__()
        try:
            connections.connect(host="localhost", port="19530")
            server_type = utility.get_server_type()
            logger.add(f"Milvus server is running. Server type: {server_type}")
        except Exception:
            logger.exception("Milvus server is not running or connection failed")

        self.client = MilvusClient(uri=f"http://{self.SERVER_HOST}:{self.SERVER_PORT}")

        self.schema = self.client.create_schema(
            auto_id=False, enable_dynamic_field=True
        )  # TODO: look into what these params are
        self.schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        self.schema.add_field(
            field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=384
        )  # TODO: replace hard coded dim

        if not self.client.collection_exists(collection_name=self.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                dimension=dim,
                index_type=index_name,
                index_param={"nlist": 1024},
            )

        self.client.switch_collection(collection_name=self.COLLECTION_NAME)

        self.dim = dim
        self.index_name = index_name
        self.metric = metric

    @override
    def upsert(self, ids: np.ndarray, x: np.ndarray) -> None:
        """Upsert vectors with given IDs."""

        data = [{"id": ids[i], "data": x[i]} for i in range(x.shape[0])]

        self.client.upsert(collection_name=self.COLLECTION_NAME, data=data)

    @override
    def delete(self, ids: Sequence[int]) -> None:
        """Delete vectors with given IDs."""
        self.client.delete(collection_name=self.COLLECTION_NAME, ids=ids)

    @override
    def drop_index(self) -> None:
        """Drop the index."""
        if self.client.collection_exists(collection_name=self.COLLECTION_NAME):
            self.collection.release()

        self.client.drop_index(index_name=self.index_name)

        self.index = None

    @override
    def stats(self) -> None:
        """Return index statistics."""
        return {
            "ntotal": int(self.index.ntotal),
            "kind": "milvus",
            "metric": self.metric,
        }

    @override
    def search(self, q: np.ndarray, topk: int, **kwargs) -> None:
        """Search for top-k nearest neighbors."""
        return self.index.search(q, topk)

    @override
    def train_index(self, x_train: np.ndarray) -> None:
        """Train the index with given vectors."""
        self.index.train(x_train)
