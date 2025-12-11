"""Qdrant vector database adaptor."""

import time
from collections.abc import Generator, Sequence
from shutil import which
from urllib.parse import urlparse

import numpy as np
import requests
from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct
from tqdm import tqdm

from inatinqperf.adaptors.base import DataPoint, HuggingFaceDataset, Query, SearchResult, VectorDatabase
from inatinqperf.adaptors.enums import Metric
from inatinqperf.container import log_single_container_tail


class Qdrant(VectorDatabase):
    """Qdrant vector database.

    Qdrant only supports a single dense vector index: HNSW.
    However, it supports indexes on the attributes (aka payload) associated with each vector.
    These payload indexes can greatly improve search efficiency.
    """

    def __init__(  # noqa: PLR0913
        self,
        dataset: HuggingFaceDataset,
        metric: Metric,
        *,
        url: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
        collection_name: str = "default_collection",
        m: int = 32,
        ef: int = 128,
        batch_size: int = 1024,
        wait_for_startup: bool = True,
        startup_timeout: float = 60.0,
        container_names: Sequence[str] | None = None,
        **params,  # noqa: ARG002
    ) -> None:
        super().__init__(dataset, metric)

        self.collection_name = collection_name
        self.m = m
        # The ef value used during collection construction
        self.ef = ef

        endpoint = self._normalise_endpoint(f"{url}:{port}", str(port))
        if wait_for_startup:
            self._wait_for_startup(endpoint, timeout=startup_timeout, container_names=container_names)

        self.client = QdrantClient(
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,  # Use gRPC since it is faster
            timeout=10,  # Extend the timeout to 10 seconds
        )

        self._initialize_collection(dataset, batch_size)

    def _get_vectors_config(self) -> models.VectorParams:
        """Get the Qdrant VectorParams config."""
        return models.VectorParams(
            size=self.dim,
            distance=self._translate_metric(self.metric),
            on_disk=True,  # save to disk immediately
        )

    def _get_index_params(self, m: int) -> models.HnswConfigDiff:
        """Get the Qdrant indexing config."""
        return models.HnswConfigDiff(
            m=m,
            ef_construct=self.ef,
            max_indexing_threads=0,
            on_disk=True,  # Store index on disk
        )

    def _upload_dataset_in_batches(self, dataset: HuggingFaceDataset, batch_size: int) -> None:
        """Upload `dataset` in batches of size `batch_size`."""
        # Batch insert dataset
        num_batches = int(np.ceil(len(dataset) / batch_size))
        next_id = 0
        for batch in tqdm(dataset.iter(batch_size=batch_size), total=num_batches):
            ids, next_id = self._resolve_ids(batch, next_id)
            vectors = batch["embedding"]

            self.client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=vectors,
                ),
            )

    def _initialize_collection(self, dataset: HuggingFaceDataset, batch_size: int) -> None:
        """Helper method to initialize collection."""
        if self.client.collection_exists(collection_name=self.collection_name):
            logger.info("Deleted existing collection")
            self.client.delete_collection(collection_name=self.collection_name)

        logger.patch(lambda r: r.update(function="constructor")).info(
            f"Creating collection {self.collection_name}"
        )

        ids, vectors = self._prepare_upload_data(dataset)
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=self._get_vectors_config(),
            hnsw_config=self._get_index_params(m=self.m),
            shard_number=4,  # reasonable default as per qdrant docs
        )
        self.client.upload_collection(
            collection_name=self.collection_name,
            vectors=vectors,
            ids=ids,
            batch_size=batch_size,
            wait=True,
        )

        # Log the number of point uploaded
        num_points_in_db = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count
        logger.info(f"Number of points in Qdrant database: {num_points_in_db}")

        logger.info("Waiting for indexing to complete")
        self.wait_for_index_ready(self.collection_name)
        logger.info("Indexing complete!")

    @staticmethod
    def _translate_metric(metric: Metric) -> Distance:
        """Helper method to convert from Metric enum to Qdrant Distance."""
        if metric == Metric.INNER_PRODUCT:
            return Distance.DOT
        if metric == Metric.COSINE:
            return Distance.COSINE
        if metric == Metric.L2:
            return Distance.EUCLID
        if metric == Metric.MANHATTAN:
            return Distance.MANHATTAN

        msg = f"{metric} metric specified is not a valid one for Qdrant."
        raise ValueError(msg)

    @staticmethod
    def _normalise_endpoint(endpoint: str, default_port: str | int) -> str:
        if "://" not in endpoint:
            endpoint = f"http://{endpoint}"

        parsed = urlparse(endpoint)
        scheme = parsed.scheme or "http"
        host = parsed.hostname
        port = parsed.port or int(default_port)

        if host is None:
            msg = f"Invalid Qdrant endpoint '{endpoint}'"
            raise ValueError(msg)

        return f"{scheme}://{host}:{port}"

    @staticmethod
    def _wait_for_startup(
        endpoint: str,
        timeout: float,
        container_names: Sequence[str] | None = None,
    ) -> None:
        """Poll the node health endpoint until it becomes reachable."""
        health_url = endpoint.rstrip("/") + "/healthz"
        deadline = time.monotonic() + timeout
        next_log = time.monotonic()

        while time.monotonic() < deadline:
            try:
                response = requests.get(health_url, timeout=3.0)
                if response.status_code == requests.codes.ok:
                    return
            except requests.RequestException:
                pass

            if container_names and time.monotonic() >= next_log:
                Qdrant._log_container_tails(container_names)
                next_log = time.monotonic() + 5.0

            time.sleep(1.0)

        msg = f"Timed out waiting for Qdrant node '{endpoint}' to become ready"
        raise TimeoutError(msg)

    @staticmethod
    def _log_container_tails(container_names: Sequence[str]) -> None:
        """Dump the tail of docker logs for the configured containers."""
        docker_cmd = which("docker")
        if docker_cmd is None:
            logger.warning("Docker binary not found; cannot fetch container logs")
            return

        for name in container_names:
            log_single_container_tail(docker_cmd=docker_cmd, container_name=name)

    def wait_for_index_ready(self, collection_name: str, poll_interval: float = 5.0) -> None:
        """Wait until Qdrant reports the collection is fully indexed and ready."""
        while True:
            info = self.client.get_collection(collection_name)

            status = info.status
            optimizer_status = info.optimizer_status

            if status == "green" and optimizer_status == "ok":
                logger.info(f"✅ Index for '{collection_name}' is ready!")
                break

            logger.info(f"⏳ Waiting... status={status}, optimizer_status={optimizer_status}")
            time.sleep(poll_interval)

    @staticmethod
    def _points_iterator(data_points: Sequence[DataPoint]) -> Generator[PointStruct]:
        """A generator to help with creating PointStructs."""
        for data_point in data_points:
            yield PointStruct(id=data_point.id, vector=data_point.vector)

    @staticmethod
    def _prepare_upload_data(dataset: HuggingFaceDataset) -> tuple[Sequence[int], Sequence[Sequence[float]]]:
        """Return IDs and vectors, falling back to synthetic IDs when missing."""
        if "id" in dataset.column_names:
            ids = dataset["id"]
        elif "photo_id" in dataset.column_names:
            ids = dataset["photo_id"]
        elif "query_id" in dataset.column_names:
            ids = dataset["query_id"]
        else:
            ids = list(range(len(dataset)))

        vectors = dataset["embedding"]
        return ids, vectors

    @staticmethod
    def _resolve_ids(batch: dict, next_id: int) -> tuple[Sequence[int], int]:
        """Choose an ID column if present, otherwise auto-generate sequential IDs."""
        for field in ("id", "photo_id", "query_id"):
            if field in batch:
                return batch[field], next_id

        size = len(batch.get("embedding", []))
        ids = list(range(next_id, next_id + size))
        return ids, next_id + size

    def upsert(self, x: Sequence[DataPoint]) -> None:
        """Upsert vectors with given IDs. This also builds the HNSW index."""
        # Qdrant will override points with the same ID if they already exist,
        # which is the same behavior as `upsert`.
        # Hence we use `upload_points` for performance.
        logger.info("Uploading points to database")
        self.client.upload_points(
            collection_name=self.collection_name,
            points=self._points_iterator(data_points=x),
            parallel=4,
            wait=True,
        )

    def search(self, q: Query, topk: int, **kwargs) -> Sequence[SearchResult]:
        """Search for top-k nearest neighbors."""
        # Has support for attribute filter: https://qdrant.tech/documentation/quickstart/#add-a-filter

        ef = kwargs.get("ef", 128)
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=q.vector,
            with_payload=False,
            with_vectors=False,
            limit=topk,
            search_params=models.SearchParams(hnsw_ef=ef, exact=False),
        )

        return [SearchResult(point.id, point.score) for point in search_result.points]

    def delete(self, ids: Sequence[int]) -> None:
        """Delete vectors with given IDs."""
        self.client.delete(collection_name=self.collection_name, points_selector=ids)

    def delete_collection(self) -> None:
        """Delete the collection associated with this adaptor instance."""
        logger.info(f"Deleting collection {self.collection_name}")
        self.client.delete_collection(collection_name=self.collection_name)

    def stats(self) -> dict[str, object]:
        """Return index statistics."""
        return {
            "metric": self.metric.value,
            "m": self.m,
            "ef_construct": self.ef,
        }

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, "client") and self.client:
            client_close = getattr(self.client, "close", None)
            if callable(client_close):
                client_close()


class QdrantCluster(Qdrant):
    """Adaptor for a sharded/replicated Qdrant cluster."""

    def __init__(  # noqa: PLR0913
        self,
        dataset: HuggingFaceDataset,
        metric: Metric,
        *,
        url: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        collection_name: str = "default_collection",
        m: int = 32,
        ef: int = 128,
        batch_size: int = 1000,
        node_urls: Sequence[str] | None = None,
        container_names: Sequence[str] | None = None,
        shard_number: int = 3,
        replication_factor: int = 1,
        write_consistency_factor: int | None = None,
        startup_timeout: float = 60.0,
        **params,
    ) -> None:
        self.node_urls = self._resolve_node_urls(node_urls, url, port)
        self.container_names = list(container_names) if container_names else None
        self.shard_number = shard_number
        self.replication_factor = replication_factor
        self.write_consistency_factor = write_consistency_factor

        primary = self.node_urls[0]
        self._wait_for_startup(primary, timeout=startup_timeout, container_names=self.container_names)

        super().__init__(
            dataset,
            metric=metric,
            url=primary,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            collection_name=collection_name,
            m=m,
            ef=ef,
            batch_size=batch_size,
            wait_for_startup=False,
            startup_timeout=startup_timeout,
            container_names=self.container_names,
            **params,
        )

    def _initialize_collection(self, dataset: HuggingFaceDataset, batch_size: int) -> None:
        """Helper method to initialize collection."""
        if self.client.collection_exists(collection_name=self.collection_name):
            logger.info("Deleted existing collection")
            self.client.delete_collection(collection_name=self.collection_name)

        logger.patch(lambda r: r.update(function="constructor")).info(
            f"Creating collection {self.collection_name} with {self.shard_number} shards"
        )

        ids, vectors = self._prepare_upload_data(dataset)
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=self._get_vectors_config(),
            hnsw_config=self._get_index_params(m=self.m),
            shard_number=self.shard_number,
            replication_factor=self.replication_factor,
            write_consistency_factor=self.write_consistency_factor,
        )
        self.client.upload_collection(
            collection_name=self.collection_name,
            vectors=vectors,
            ids=ids,
            batch_size=batch_size,
            wait=True,
        )

        num_points_in_db = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count
        logger.info(f"Number of points in Qdrant database: {num_points_in_db}")

        logger.info("Waiting for indexing to complete")
        self.wait_for_index_ready(self.collection_name)
        logger.info("Indexing complete!")

    @staticmethod
    def _resolve_node_urls(
        node_urls: Sequence[str] | None,
        default_url: str,
        default_port: str | int,
    ) -> list[str]:
        if node_urls:
            return [QdrantCluster._normalise_endpoint(raw, default_port) for raw in node_urls]
        return [QdrantCluster._normalise_endpoint(f"{default_url}:{default_port}", default_port)]

    def stats(self) -> dict[str, object]:
        """Return cluster-aware index statistics."""
        stats = super().stats()
        stats.update(
            {
                "shard_number": self.shard_number,
                "replication_factor": self.replication_factor,
                "write_consistency_factor": self.write_consistency_factor,
                "nodes": self.node_urls,
            }
        )
        return stats

    @staticmethod
    def _batched(
        dataset: HuggingFaceDataset,
        batch_size: int,
    ) -> Generator[list[dict[str, object]], None, None]:
        """Yield dataset rows in batches, keeping payload fields for upsert."""
        batch: list[dict[str, object]] = []
        for row in dataset:
            batch.append(dict(row))
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch
