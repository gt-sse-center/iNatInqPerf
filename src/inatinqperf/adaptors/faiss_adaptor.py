"""FAISS vector database adaptor."""

from collections.abc import Generator, Sequence

import faiss
import numpy as np
import pyarrow as pa
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

# Process large datasets in chunks to keep peak FAISS ingest memory bounded.
_DEFAULT_BATCH_SIZE = 8192
Batch = tuple[np.ndarray, np.ndarray]


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

        Keyword Args:
            **kwargs: All the extra keyword arguments. This is not used.
        """
        super().__init__(dataset, metric)
        self.index_type: FaissIndexType = self._translate_index_type(index_type)
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe

        if self.index_type == FaissIndexType.FLAT:
            self.index = self._build_flat_index(metric=self.metric, dim=self.dim, dataset=dataset)
        elif self.index_type == FaissIndexType.IVFPQ:
            self.index = self._build_ivfpq_index(
                metric=self.metric,
                dim=self.dim,
                dataset=dataset,
                nlist=nlist,
                m=m,
                nbits=nbits,
                nprobe=nprobe,
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
    ) -> faiss.Index:
        if metric in (Metric.INNER_PRODUCT, Metric.COSINE):
            base = faiss.IndexFlatIP(dim)
        else:
            base = faiss.IndexFlatL2(dim)

        index = faiss.IndexIDMap2(base)

        logger.info(
            f"Ingesting {len(dataset)} vectors into Faiss FLAT index (batch_size={_DEFAULT_BATCH_SIZE})"
        )
        # Iterate over zero-copy Arrow batches when available to minimise conversion overhead.
        batch_iter = _iter_dataset_batches(
            dataset,
            _DEFAULT_BATCH_SIZE,
            show_progress=True,
            desc="FAISS FLAT ingest",
        )
        total_added = 0
        for ids_batch, embeddings_batch in batch_iter:
            if not len(ids_batch):
                continue
            # Stream batches straight into the index to avoid a monolithic array.
            index.add_with_ids(embeddings_batch, ids_batch)
            total_added += len(ids_batch)

        logger.info(f"Building Faiss FLAT index with {total_added} vectors")

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
    ) -> faiss.Index:
        n = len(dataset)

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

        ivf = _unwrap_to_ivf(index.index)

        # Extract the entire embedding matrix once for training; this prefers zero-copy Arrow buffers.
        full_embeddings = _extract_embeddings(dataset)
        if full_embeddings.size == 0:
            msg = "No embeddings available to train the FAISS index."
            raise ValueError(msg)

        logger.info(
            f"Training Faiss IVFPQ index with embeddings of shape {full_embeddings.shape} "
            f"(using all vectors, effective_nlist={effective_nlist})"
        )
        # FAISS training expects a contiguous block; we now feed every vector.
        index.train(full_embeddings)
        # Release the training matrix before streaming ingestion to reduce peak memory.
        del full_embeddings

        logger.info(f"Adding {len(dataset)} vectors to Faiss IVFPQ index (batch_size={_DEFAULT_BATCH_SIZE})")
        batch_iter = _iter_dataset_batches(
            dataset,
            _DEFAULT_BATCH_SIZE,
            show_progress=True,
            desc="FAISS IVFPQ ingest",
        )
        total_added = 0
        for ids_batch, embeddings_batch in batch_iter:
            if not len(ids_batch):
                continue
            index.add_with_ids(embeddings_batch, ids_batch)
            total_added += len(ids_batch)

        logger.info(f"_build_ivfpq_index : added {total_added} vectors to index")

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

        ids = np.asarray(ids, dtype=np.int64)
        self.index.remove_ids(faiss.IDSelectorArray(ids))
        self.index.add_with_ids(np.asarray(vectors, dtype=np.float32), ids)

    def search(self, q: Query, topk: int, **kwargs) -> Sequence[SearchResult]:
        """Search for top-k nearest neighbors."""
        query_vector = np.asarray(q.vector, dtype=np.float32)

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

        distances = np.array(distances).reshape(-1).astype(np.float32, copy=False)
        labels = np.array(labels).reshape(-1).astype(np.int64, copy=False)

        results: list[SearchResult] = []
        for score, label in zip(distances, labels):
            results.append(SearchResult(id=int(label), score=float(score)))

        return results

    def delete(self, ids: Sequence[int]) -> None:
        """Delete vectors with given IDs."""
        arr = np.asarray(list(ids), dtype=np.int64)
        self.index.remove_ids(faiss.IDSelectorArray(arr))

    def stats(self) -> dict[str, object]:
        """Return index statistics."""
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


def _arrow_array_to_numpy(
    obj: pa.Array | pa.ChunkedArray | pa.Buffer,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Return a NumPy view of an Arrow array or buffer, copying only if unavoidable."""
    if isinstance(obj, pa.ChunkedArray):
        obj = obj.combine_chunks()

    if isinstance(obj, pa.Buffer):
        np_array = np.frombuffer(obj, dtype=dtype or np.int32)
    elif isinstance(obj, pa.Array):
        try:
            np_array = obj.to_numpy(zero_copy_only=True)
        except (pa.ArrowInvalid, ValueError):
            np_array = obj.to_numpy(zero_copy_only=False)
    else:
        msg = f"Unsupported Arrow object type: {type(obj)}"
        raise TypeError(msg)

    if dtype is not None and np_array.dtype != dtype:
        np_array = np_array.astype(dtype, copy=False)

    return np_array


def _embedding_feature_length(dataset: HuggingFaceDataset) -> int:
    """Return embedding dimensionality hint from dataset features."""
    feature = dataset.features.get("embedding") if hasattr(dataset, "features") else None
    length = getattr(getattr(feature, "feature", None), "length", -1)
    return int(length) if isinstance(length, int) and length > 0 else -1


def _arrow_list_to_matrix(array: pa.Array | pa.ChunkedArray, feature_length: int) -> np.ndarray:
    """Convert an Arrow list-based array into a 2D C-contiguous float32 matrix."""
    if isinstance(array, pa.ChunkedArray):
        if array.num_chunks == 1:
            return _arrow_list_to_matrix(array.chunk(0), feature_length)

        chunks = [_arrow_list_to_matrix(array.chunk(i), feature_length) for i in range(array.num_chunks)]
        if not chunks:
            dim = max(0, feature_length)
            return np.empty((0, dim), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    if isinstance(array, pa.FixedSizeListArray):
        values = _arrow_array_to_numpy(array.values, dtype=np.float32)
        dim = int(array.type.list_size)
        vectors = values.reshape(len(array), dim)
        return np.require(vectors, requirements=["C"])

    if isinstance(array, (pa.ListArray, pa.LargeListArray)):
        offsets = _arrow_array_to_numpy(array.offsets, dtype=np.int64)
        num_rows = len(offsets) - 1
        if num_rows == 0:
            dim = max(0, feature_length)
            return np.empty((0, dim), dtype=np.float32)

        inferred_dim = int(offsets[1] - offsets[0]) if len(offsets) > 1 else feature_length
        if inferred_dim <= 0 and feature_length > 0:
            inferred_dim = feature_length
        if inferred_dim <= 0:
            msg = "Unable to infer embedding dimension from Arrow offsets."
            raise ValueError(msg)

        values = _arrow_array_to_numpy(array.values, dtype=np.float32)
        vectors = values.reshape(num_rows, inferred_dim)
        return np.require(vectors, requirements=["C"])

    # Fallback: materialise to NumPy for unexpected layouts.
    numpy_values = np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float32)
    if numpy_values.ndim == 1:
        dim = feature_length if feature_length > 0 else numpy_values.shape[0]
        numpy_values = numpy_values.reshape(1, dim)
    return np.require(numpy_values, requirements=["C"])


def _iter_dataset_batches(
    dataset: HuggingFaceDataset,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    *,
    show_progress: bool = False,
    desc: str | None = None,
) -> Generator[Batch, None, None]:
    """Yield `(ids, vectors)` batches, favouring zero-copy Arrow views where possible."""
    total = len(dataset)
    progress = tqdm(total=total, desc=desc, unit="vec") if show_progress else None

    try:
        arrow_data = getattr(dataset, "data", None)
        if arrow_data is not None and hasattr(arrow_data, "to_batches"):
            # Hugging Face datasets expose their Arrow table via .data; iterate batches directly.
            table = arrow_data
            feature_length = _embedding_feature_length(dataset)
            for batch in table.to_batches(max_chunksize=batch_size):
                try:
                    ids_arr = _arrow_array_to_numpy(batch.column("id"), dtype=np.int64)
                    emb_arr = _arrow_list_to_matrix(batch.column("embedding"), feature_length)
                except Exception:
                    ids_arr = np.asarray(batch.column("id").to_numpy(zero_copy_only=False), dtype=np.int64)
                    emb_arr = np.asarray(batch.column("embedding").to_pylist(), dtype=np.float32)
                ids_arr = np.require(ids_arr, requirements=["C"])
                emb_arr = np.require(emb_arr, requirements=["C"])
                if progress is not None:
                    progress.update(len(ids_arr))
                yield ids_arr, emb_arr
            return

        # Fall back to numpy conversions when Arrow metadata is not available (e.g. in tests).
        ids = dataset["id"]
        embeddings = dataset["embedding"]
        if not isinstance(ids, np.ndarray):
            ids = np.asarray(ids, dtype=np.int64)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings, dtype=np.float32)
        # Slice numpy views per batch when only list-backed columns are available.
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            ids_batch = np.require(ids[start:end].astype(np.int64, copy=False), requirements=["C"])
            emb_batch = np.require(embeddings[start:end].astype(np.float32, copy=False), requirements=["C"])
            if progress is not None:
                progress.update(len(ids_batch))
            yield ids_batch, emb_batch
    finally:
        if progress is not None:
            progress.close()


def _extract_embeddings(dataset: HuggingFaceDataset) -> np.ndarray:
    """Extract embeddings as a 2D float32 NumPy array without materialising Python lists."""
    arrow_data = getattr(dataset, "data", None)
    if arrow_data is not None and hasattr(arrow_data, "column"):
        try:
            # Prefer Arrow-backed buffers so training can proceed without extra copies.
            column = arrow_data.column("embedding")
            return _arrow_list_to_matrix(column, _embedding_feature_length(dataset))
        except (KeyError, pa.ArrowInvalid, ValueError):
            pass

    embeddings = dataset["embedding"]
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32, copy=False)
    return np.require(embeddings, requirements=["C"])
