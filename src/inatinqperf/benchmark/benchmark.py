"""Vector database-agnostic benchmark orchestrator."""

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml
from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from inatinqperf.adaptors import VECTORDBS, DataPoint, Faiss, Query, SearchResult, VectorDatabase
from inatinqperf.benchmark.configuration import Config
from inatinqperf.benchmark.container import container_context
from inatinqperf.utils import (
    ImageDatasetWithEmbeddings,
    Profiler,
    embed_images,
    embed_text,
    export_images,
    get_table,
    load_huggingface_dataset,
    to_huggingface_dataset,
)


class Benchmarker:
    """Class to encapsulate all benchmarking operations."""

    def __init__(self, config_file: Path, base_path: Path | None = None) -> None:
        """Construct the benchmark orchestrator.

        Args:
            config_file (Path): Path to the config file with the parameters required to run the benchmark.
            base_path (Path | None, optional): The path to which all data will be saved.
                If None, it will be set to the root directory of the project.
        """
        logger.patch(lambda r: r.update(function="constructor")).info(f"Loading config: {config_file}")

        with config_file.open("r") as f:
            cfg = yaml.safe_load(f)
        # Load into Config class to validate properties
        self.cfg = Config(**cfg)

        if base_path is None:
            self.base_path = Path(__file__).resolve().parent.parent
        else:
            self.base_path = base_path

    def download(self) -> None:
        """Download HF dataset and optionally export images."""
        dataset_id = self.cfg.dataset.dataset_id

        dataset_dir = self.base_path / self.cfg.dataset.directory

        if dataset_dir.exists():
            logger.info("Dataset already exists, continuing...")
            return

        ensure_dir(dataset_dir)
        export_raw_images = self.cfg.dataset.export_images
        splits = self.cfg.dataset.splits

        with Profiler(f"download-{dataset_id.split('/')[-1]}-{splits}"):
            ds = load_huggingface_dataset(dataset_id, splits)
            ds.save_to_disk(dataset_dir)

            if export_raw_images:
                export_dir = dataset_dir / "images"
                manifest = export_images(ds, export_dir)
                logger.info(f"Exported images to: {export_dir}\nManifest: {manifest}")

        logger.info(f"Downloaded HuggingFace dataset to: {dataset_dir}")

    def embed(self) -> Dataset:
        """Compute CLIP embeddings of the dataset images and save to embedding directory."""

        embeddings_dir = self.base_path / self.cfg.embedding.directory

        if embeddings_dir.exists():
            logger.info("Embeddings found, loading instead of computing")
            return Dataset.load_from_disk(dataset_path=embeddings_dir)

        model_id = self.cfg.embedding.model_id
        batch_size = self.cfg.embedding.batch_size

        dataset_dir = self.base_path / self.cfg.dataset.directory
        logger.info(f"Generating embeddings with model={model_id} and saving to {dataset_dir}")

        with Profiler("embed-images"):
            dse: ImageDatasetWithEmbeddings = embed_images(dataset_dir, model_id, batch_size)

        return self.save_as_huggingface_dataset(dse, embeddings_dir=embeddings_dir)

    def save_as_huggingface_dataset(
        self,
        dse: ImageDatasetWithEmbeddings,
        embeddings_dir: Path | None = None,
    ) -> Dataset:
        """Convert to HuggingFace dataset format and save to `embeddings_dir`."""

        if embeddings_dir is None:
            embeddings_dir = self.base_path / self.cfg.embedding.directory

        ensure_dir(embeddings_dir)

        logger.info(f"Saving dataset to {embeddings_dir}")
        huggingface_dataset: Dataset = to_huggingface_dataset(dse)
        huggingface_dataset.save_to_disk(embeddings_dir)

        logger.info(f"Embeddings: {dse.embeddings.shape} -> {embeddings_dir}")

        return huggingface_dataset.with_format("numpy")

    def build(self, dataset: Dataset) -> VectorDatabase:
        """Build index for the specified vectordb."""
        vdb_type = self.cfg.vectordb.type
        logger.info(f"Building {vdb_type} vector database")

        init_params = self.cfg.vectordb.params.to_dict()

        with Profiler(f"build-{vdb_type}"):
            vdb = VECTORDBS[vdb_type](dataset, **init_params)

        logger.info(f"Stats: {vdb.stats()}")

        return vdb

    def build_baseline(self, dataset: Dataset) -> VectorDatabase:
        """Build the FAISS vector database with a `IndexFlat` index as a baseline."""
        metric = self.cfg.vectordb.params.metric.lower()

        # Create exact baseline
        faiss_flat_db = Faiss(dataset, metric=metric, index_type="FLAT")
        logger.info("Created exact baseline index")

        return faiss_flat_db

    def search(self, dataset: Dataset, vectordb: VectorDatabase, baseline_vectordb: VectorDatabase) -> None:
        """Profile search and compute recall@K vs exact baseline."""
        params = self.cfg.vectordb.params
        model_id = self.cfg.embedding.model_id

        topk = self.cfg.search.topk

        dataset_dir = self.base_path / self.cfg.dataset.directory
        ds = Dataset.load_from_disk(dataset_dir)
        if "query" in ds.column_names:
            queries = ds["query"]

        else:
            queries_file = Path(__file__).resolve().parent.parent / self.cfg.search.queries_file
            queries = [q.strip() for q in queries_file.read_text(encoding="utf-8").splitlines() if q.strip()]

        q = embed_text(queries, model_id)
        logger.info("Embedded all queries")

        logger.info("Performing search on baseline")
        with Profiler("search-baseline-FaissFlat") as p:
            i0 = np.full((q.shape[0], topk), -1.0, dtype=float)
            for i in tqdm(range(q.shape[0])):
                base_results = baseline_vectordb.search(Query(q[i]), topk)  # exact
                padded = _ids_to_fixed_array(base_results, topk)
                i0[i] = padded

        # search + profile
        logger.info(f"Performing search on {self.cfg.vectordb.type}")
        with Profiler(f"search-{self.cfg.vectordb.type}") as p:
            latencies = []
            for i in tqdm(range(q.shape[0])):
                t0 = time.perf_counter()
                vectordb.search(Query(q[i]), topk, **params.to_dict())
                latencies.append((time.perf_counter() - t0) * 1000.0)

            p.sample()

        logger.info("recall@K (compare last retrieved to baseline per query")
        # For simplicity compute approximate on whole Q at once:
        i1 = np.full((q.shape[0], topk), -1.0, dtype=float)
        for i in tqdm(range(q.shape[0])):
            results = vectordb.search(Query(q[i]), topk, **params.to_dict())
            padded = _ids_to_fixed_array(results, topk)
            i1[i] = padded
        rec = recall_at_k(i1, i0, topk)

        x = np.asarray(dataset["embedding"], dtype=np.float32)

        stats = {
            "vectordb": self.cfg.vectordb.type,
            "index_type": self.cfg.vectordb.params.index_type,
            "topk": topk,
            "lat_ms_avg": float(np.mean(latencies)),
            "lat_ms_p50": float(np.percentile(latencies, 50)),
            "lat_ms_p95": float(np.percentile(latencies, 95)),
            "recall@k": rec,
            "ntotal": int(x.shape[0]),
        }

        # Make values as lists so `tabulate` can print properly.
        table = get_table(stats)
        logger.info(f"\n\n{table}\n\n")

    def update(self, dataset: Dataset, vectordb: VectorDatabase) -> None:
        """Upsert + delete small batch and re-search."""
        vdb_type = self.cfg.vectordb.type

        add_n = self.cfg.update["add_count"]
        del_n = self.cfg.update["delete_count"]

        x = np.asarray(dataset["embedding"], dtype=np.float32)

        # craft new vectors by slight noise around existing (simulating fresh writes)
        rng = np.random.default_rng(42)
        add_vecs = x[:add_n].copy()
        add_vecs += rng.normal(0, 0.01, size=add_vecs.shape).astype(np.float32)
        add_ids = list(range(add_n))

        with Profiler(f"update-add-{vdb_type}"):
            data_points = [DataPoint(id=i, vector=v, metadata={}) for i, v in zip(add_ids, add_vecs)]
            vectordb.upsert(data_points)

        with Profiler(f"update-delete-{vdb_type}"):
            del_ids = add_ids[:del_n]
            vectordb.delete(del_ids)

        logger.info(f"Update complete: {vectordb.stats()}")

    def run(self) -> None:
        """Run end-to-end benchmark with all steps."""
        # Download dataset
        self.download()

        # Compute embeddings
        dataset = self.embed()

        with container_context(self.cfg):
            # Build baseline vector database
            baseline_vectordb = self.build_baseline(dataset)

            # Build specified vector database
            vectordb = self.build(dataset)

            # Perform search
            self.search(dataset, vectordb, baseline_vectordb)

            # Update operations
            self.update(dataset, vectordb)

            # Clean up the vectordb
            del vectordb


def ensure_dir(p: Path) -> Path:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def recall_at_k(approx_i: np.ndarray, exact_i: np.ndarray, k: int) -> float:
    """Compute recall@K between two sets of indices."""
    hits = 0
    for i in range(approx_i.shape[0]):
        hits += len(set(approx_i[i, :k]).intersection(set(exact_i[i, :k])))
    return hits / float(approx_i.shape[0] * k)


def _ids_to_fixed_array(results: Sequence[SearchResult], topk: int) -> np.ndarray:
    """Convert a list of SearchResult objects into a fixed-length array."""

    arr = np.full(topk, -1.0, dtype=float)
    if not results:
        return arr

    count = min(topk, len(results))
    arr[:count] = [float(results[i].id) for i in range(count)]
    return arr
