"""Qdrant vector database adaptor."""

import subprocess
import time
from collections.abc import Generator, Sequence
from itertools import islice
from shutil import which
from urllib.parse import urlparse

import requests
from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct
from tqdm import tqdm

from inatinqperf.adaptors.base import DataPoint, HuggingFaceDataset, Query, SearchResult, VectorDatabase
from inatinqperf.adaptors.enums import Metric

HTTP_STATUS_OK = 200
DEFAULT_GRPC_PORT = 6334
_ALLOWED_CONTAINER_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def _is_safe_container_name(name: str) -> bool:
    return bool(name) and all(char in _ALLOWED_CONTAINER_CHARS for char in name)


def _log_single_container_tail(docker_cmd: str, container_name: str) -> None:
    if not _is_safe_container_name(container_name):
        logger.warning(f"Skipping unsafe container name: {container_name!r}")
        return

    try:
        result = subprocess.run(  # noqa: S603
            [docker_cmd, "logs", "--tail", "20", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(f"Failed to fetch logs for container '{container_name}': {exc}")
        return

    output = result.stdout.strip()
    if output:
        logger.warning(f"[{container_name}] {output}")
    error_output = result.stderr.strip()
    if error_output:
        logger.warning(f"[{container_name}][stderr] {error_output}")


class Qdrant(VectorDatabase):
    """Qdrant vector database.

    Qdrant only supports a single dense vector index: HNSW.
    However, it supports indexes on the attributes (aka payload) associated with each vector.
    These payload indexes can greatly improve search efficiency.
    """

    def __init__(
        self,
        dataset: HuggingFaceDataset,
        metric: Metric,
        url: str = "localhost",
        port: str = "6333",
        collection_name: str = "default_collection",
        m: int = 32,
        ef: int = 128,
        batch_size: int = 1024,
        **params,  # noqa: ARG002
    ) -> None:
        super().__init__(dataset, metric)

        self.client = QdrantClient(
            url=url,
            port=port,
            prefer_grpc=True,  # Use gRPC since it is faster
            timeout=10,  # Extend the timeout to 10 seconds
        )
        self.collection_name = collection_name

        self.m = m
        # The ef value used during collection construction
        self.ef = ef

        if self.client.collection_exists(collection_name=collection_name):
            logger.info("Deleted existing collection")
            self.client.delete_collection(collection_name=collection_name)

        logger.patch(lambda r: r.update(function="constructor")).info(
            f"Creating collection {collection_name}"
        )

        vectors_config = models.VectorParams(
            size=self.dim,
            distance=self._translate_metric(metric),
            on_disk=True,  # save to disk immediately
        )

        index_params = models.HnswConfigDiff(
            m=0,  # disable indexing until dataset upload is complete
            ef_construct=ef,
            max_indexing_threads=0,
            on_disk=True,  # Store index on disk
        )

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            hnsw_config=index_params,
            shard_number=4,  # reasonable default as per qdrant docs
        )

        # Batch insert dataset
        num_batches = int(np.ceil(len(dataset) / batch_size))
        for batch in tqdm(dataset.iter(batch_size=batch_size), total=num_batches):
            ids = batch["id"]
            vectors = batch["embedding"]

            self.client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=vectors,
                ),
            )

        # Set the indexing params
        self.client.update_collection(
            collection_name=collection_name,
            hnsw_config=models.HnswConfigDiff(m=m),
        )

        # Log the number of point uploaded
        num_points_in_db = self.client.count(
            collection_name=collection_name,
            exact=True,
        ).count
        logger.info(f"Number of points in Qdrant database: {num_points_in_db}")

        logger.info("Waiting for indexing to complete")
        self.wait_for_index_ready(collection_name)
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
            self.client.close()


class QdrantCluster(Qdrant):
    """Adaptor for a sharded/replicated Qdrant cluster."""

    def __init__(  # noqa: PLR0913
        self,
        dataset: HuggingFaceDataset,
        metric: Metric,
        url: str = "localhost",
        port: str = "6333",
        node_urls: Sequence[str] | None = None,
        container_names: Sequence[str] | None = None,
        collection_name: str = "default_collection",
        m: int = 32,
        ef: int = 128,
        shard_number: int = 3,
        replication_factor: int = 1,
        write_consistency_factor: int | None = None,
        grpc_port: str | None = None,
        *,
        prefer_grpc: bool = False,
        batch_size: int = 1000,
        startup_timeout: float = 60.0,
        **params,  # noqa: ARG002
    ) -> None:
        VectorDatabase.__init__(self, dataset, metric)

        self.collection_name = collection_name
        self.node_urls = self._resolve_node_urls(node_urls, url, port)
        self.container_names = list(container_names) if container_names else None
        self.m = m
        self.ef = ef
        self.shard_number = shard_number
        self.replication_factor = replication_factor
        self.write_consistency_factor = write_consistency_factor

        primary = self.node_urls[0]
        self._wait_for_startup(primary, timeout=startup_timeout, container_names=self.container_names)
        grpc = int(grpc_port) if grpc_port is not None else 6334
        self.client = QdrantClient(url=primary, grpc_port=grpc, prefer_grpc=prefer_grpc)

        if self.client.collection_exists(collection_name=collection_name):
            logger.info("Deleted existing collection")
            self.client.delete_collection(collection_name=collection_name)

        logger.patch(lambda r: r.update(function="constructor")).info(
            f"Creating collection {collection_name} with {self.shard_number} shards"
        )

        qdrant_index_params = models.HnswConfigDiff(
            m=self.m,
            ef_construct=self.ef,
            max_indexing_threads=0,
            on_disk=False,
        )

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.dim,
                distance=self._translate_metric(metric),
                hnsw_config=qdrant_index_params,
            ),
            shard_number=self.shard_number,
            replication_factor=self.replication_factor,
            write_consistency_factor=self.write_consistency_factor,
        )

        for batch in self._batched(dataset, batch_size):
            ids = [point.pop("id") for point in batch]
            vectors = [point.pop("embedding") for point in batch]

            self.client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=vectors,
                    payloads=batch,
                ),
            )

        num_points_in_db = self.client.count(
            collection_name=collection_name,
            exact=True,
        ).count
        logger.info(f"Number of points in Qdrant database: {num_points_in_db}")

    @staticmethod
    def _resolve_node_urls(
        node_urls: Sequence[str] | None,
        default_url: str,
        default_port: str,
    ) -> list[str]:
        if node_urls:
            return [QdrantCluster._normalise_endpoint(raw, default_port) for raw in node_urls]
        return [QdrantCluster._normalise_endpoint(f"{default_url}:{default_port}", default_port)]

    @staticmethod
    def _normalise_endpoint(endpoint: str, default_port: str) -> str:
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
                if response.status_code == HTTP_STATUS_OK:
                    return
            except requests.RequestException:
                pass

            if container_names and time.monotonic() >= next_log:
                QdrantCluster._log_container_tails(container_names)
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
            _log_single_container_tail(docker_cmd=docker_cmd, container_name=name)

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
