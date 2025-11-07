"""FAISS vector database adaptor."""

from collections.abc import Sequence

import faiss
import numpy as np
from loguru import logger
from tqdm import tqdm

from inatinqperf.adaptors.base import (
    DataPoint,
    HuggingFaceDataset,
    Query,
    SearchResult,
    VectorDatabase,
)
from inatinqperf.adaptors.enums import IndexTypeBase, Metric


class FaissIndexType(IndexTypeBase):
    """Enum for index types used with FAISS."""

    FLAT = "flat"
    IVFPQ = "ivfpq"


class Faiss(VectorDatabase):
    """Base class for FAISS vector database."""

    def __init__(
        self,
        dataset: HuggingFaceDataset,
        metric: Metric = Metric.INNER_PRODUCT,
        nlist: int = 32768,
        m: int = 64,
        nbits: int = 8,
        nprobe: int = 32,
        index_type: str = "flat",
        batch_size: int = 8192,
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Constructor for the FAISS adaptor.

        Args:
            dataset (HuggingFaceDataset): The dataset to load into the vector database for indexing.
            metric (Metric, optional): The metric to use for computing similarity during search.
                Defaults to Metric.INNER_PRODUCT.
            nlist (int, optional): The number of clusters to generate when creating
                an inverted file (IVF) index. Defaults to 32768.
            m (int, optional): The number of sub-vectors to split the vectors into
                for multi-codebook quantization. Defaults to 64.
            nbits (int, optional): The number of bits to use for each sub-vector quantizer.
                This gives the number of clusters as 2^nbits.
                Defaults to 8.
            nprobe (int, optional): The number of clusters visited during search. Defaults to 32.
            index_type (str, optional): The type of index to use. Defaults to "flat".
            batch_size (int, optional): The batch size used during index building.
            **kwargs: All the extra keyword arguments. This is not used.
        """
        super().__init__(dataset, metric)
        self.index_type: FaissIndexType = self._translate_index_type(index_type)
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe

        if self.index_type == FaissIndexType.FLAT:
            self.index = self._build_flat_index(
                metric=self.metric,
                dim=self.dim,
                dataset=dataset,
                batch_size=batch_size,
            )

        elif self.index_type == FaissIndexType.IVFPQ:
            self.index = self._build_ivfpq_index(
                metric=self.metric,
                dim=self.dim,
                dataset=dataset,
                nlist=nlist,
                m=m,
                nbits=nbits,
                nprobe=nprobe,
                batch_size=batch_size,
            )

    @staticmethod
    def _translate_metric(metric: Metric) -> str:
        """Map the metric value to a string value which is used by the FAISS client."""
        return metric.value

    @staticmethod
    def _translate_index_type(index_type: str) -> FaissIndexType:
        """Return the proper FaissIndexType enum."""
        if index_type.lower() == "flat":
            return FaissIndexType.FLAT

        if index_type.lower() == "ivfpq":
            return FaissIndexType.IVFPQ

        msg = f"Invalid index type: {index_type}"
        raise ValueError(msg)

    @staticmethod
    def _build_flat_index(
        metric: Metric,
        dim: int,
        dataset: HuggingFaceDataset,
        batch_size: int,
    ) -> faiss.Index:
        if metric in (Metric.INNER_PRODUCT, Metric.COSINE):
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            metric_type = faiss.METRIC_L2

        logger.info(f"Ingesting {len(dataset)}x{dim} vectors into Faiss FLAT index (batch_size={batch_size})")

        dataset.add_faiss_index(
            column="embedding",
            string_factory="Flat",
            metric_type=metric_type,
            batch_size=batch_size,
        )
        index = dataset.get_index("embedding").faiss_index
        logger.info(f"Built Faiss FLAT index with {len(dataset)} vectors")

        return index

    @staticmethod
    def _build_ivfpq_index(
        metric: Metric,
        dim: int,
        dataset: HuggingFaceDataset,
        nlist: int,
        m: int,
        nbits: int,
        nprobe: int,
        batch_size: int,
    ) -> faiss.Index:
        embeddings = np.asarray(dataset["embedding"], dtype=np.float32)
        n = embeddings.shape[0]

        # Since FAISS hardcodes the minimum number
        # of clustering points to 39, we make sure
        # to set the effective nlist accordingly.
        effective_nlist = max(1, min(nlist, int(np.floor(n / 39))))

        # Build a robust composite index via index_factory
        # NOTE: OPQ always uses 2^8 centroids, as this is hardcoded in FAISS (https://github.com/facebookresearch/faiss/blob/3b14dad6d9ac48a0764e5ba01a45bca1d7d738ee/faiss/VectorTransform.cpp#L1068)
        # Hence we check for the effective nlist against 2^8 to decide whether to use OPQ.
        if effective_nlist < 2**8:
            desc = f"IVF{effective_nlist},PQ{m}x{nbits}"
        else:
            desc = f"OPQ{m},IVF{effective_nlist},PQ{m}x{nbits}"

        if metric in (Metric.INNER_PRODUCT, Metric.COSINE):
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            metric_type = faiss.METRIC_L2

        base = faiss.index_factory(dim, desc, metric_type)
        index = faiss.IndexIDMap2(base)

        # Train the index
        logger.info(f"Training IVFPQ with embeddings of shape {embeddings.shape} {embeddings.dtype}")
        index.train(embeddings)
        # Release the training matrix before streaming ingestion to reduce peak memory.
        del embeddings

        logger.info(f"Adding {len(dataset)} vectors to Faiss IVFPQ index (batch_size={batch_size})")
        num_batches = int(np.ceil(len(dataset) / batch_size))
        for batch in tqdm(
            dataset.iter(batch_size=batch_size),
            desc="FAISS IVFPQ ingest",
            total=num_batches,
        ):
            ids = np.asarray(batch["id"], dtype=np.int64, order="C")
            embeddings = np.asarray(batch["embedding"], dtype=np.float32, order="C")
            # Stream batches straight into the index to avoid a monolithic array.
            index.add_with_ids(embeddings, ids)

        logger.info(f"Built Faiss FLAT index with {len(dataset)} vectors")

        # Set nprobe (if we have IVF)
        ivf = _unwrap_to_ivf(index.index)
        if hasattr(ivf, "nprobe"):
            # Clamp nprobe reasonably based on nlist if available
            nlist = int(getattr(ivf, "nlist", max(1, nprobe)))
            ivf.nprobe = min(nprobe, max(1, nlist))

        return index

    def upsert(self, x: Sequence[DataPoint]) -> None:
        """Upsert vectors with given IDs."""
        ids, vectors = [], []
        for d in x:
            ids.append(d.id)

            if len(d.vector) != self.dim:
                msg = (
                    f"Vector being upserted has incorrect dimension={len(d.vector)}, should be dim{self.dim}."
                )
                raise ValueError(msg)

            vectors.append(d.vector)

        ids = np.asarray(ids, dtype=np.int64, order="C")
        vectors = np.asarray(vectors, dtype=np.float32, order="C")
        self.index.remove_ids(faiss.IDSelectorArray(ids))

        if isinstance(self.index, faiss.IndexFlat):
            self.index.add(vectors)
        else:
            self.index.add_with_ids(vectors, ids)

    def search(self, q: Query, topk: int, **kwargs) -> Sequence[SearchResult]:
        """Search for top-k nearest neighbors."""
        query_vector = np.asarray(q.vector, dtype=np.float32, order="C")

        if query_vector.ndim > 1:
            msg = "Query vector should be 1-dimensional."
            raise ValueError(msg)

        if query_vector.shape[0] != self.dim:
            msg = f"Query vector has incorrect dimension={query_vector.shape[0]}"
            raise ValueError(msg)

        # Add extra dimension to make the query vector compatible with FAISS
        query_vector = query_vector[None, :]

        if self.index_type == FaissIndexType.IVFPQ:
            # Runtime override for nprobe
            ivf = _unwrap_to_ivf(self.index.index)
            if ivf is not None and hasattr(ivf, "nprobe"):
                ivf.nprobe = int(kwargs.get("nprobe", self.nprobe))

        distances, labels = self.index.search(query_vector, topk)

        distances = np.array(distances).reshape(-1).astype(np.float32, copy=False, order="C")
        labels = np.array(labels).reshape(-1).astype(np.int64, copy=False, order="C")

        results: list[SearchResult] = []
        for score, label in zip(distances, labels):
            results.append(SearchResult(id=int(label), score=float(score)))

        return results

    def delete(self, ids: Sequence[int]) -> None:
        """Delete vectors with given IDs."""
        arr = np.asarray(list(ids), dtype=np.int64, order="C")
        self.index.remove_ids(faiss.IDSelectorArray(arr))

    def stats(self) -> dict[str, object]:
        """Return index statistics."""
        if self.index_type == FaissIndexType.FLAT:
            ivf = None
        else:
            ivf = _unwrap_to_ivf(self.index.index) if self.index is not None else None

        return {
            "ntotal": int(self.index.ntotal),
            "kind": self.index_type.value,
            "metric": self.metric.value,
            "nlist": int(getattr(ivf, "nlist", -1)) if ivf is not None else None,
            "nprobe": int(getattr(ivf, "nprobe", -1)) if ivf is not None else None,
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
