"""Vector database-agnostic benchmark orchestrator."""

import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import tqdm
import yaml
from datasets import Dataset
from loguru import logger

from inatinqperf.adaptors import VECTORDBS
from inatinqperf.adaptors.base import VectorDatabase
from inatinqperf.benchmark.configuration import Config
from inatinqperf.utils.dataio import export_images, load_huggingface_dataset
from inatinqperf.utils.embed import (
    ImageDatasetWithEmbeddings,
    embed_images,
    embed_text,
    to_huggingface_dataset,
)
from inatinqperf.utils.profiler import Profiler


class Benchmarker:
    """Class to encapsulate all benchmarking operations."""

    def __init__(self, config_file: Path, base_path: Path | None = None) -> None:
        """Construct the benchmark orchestrator.

        Args:
            config_file (Path): Path to the config file with the parameters required to run the benchmark.
            base_path (Path | None, optional): The path to which all data will be saved.
                If None, it will be set to the root directory of the project.
        """
        logger.info(f"Loading config: {config_file}")
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

        return huggingface_dataset

    @staticmethod
    def init_vectordb(vdb_type: str, dim: int, init_params: dict[str, Any]) -> VectorDatabase:
        """Initialize vector database."""
        return VECTORDBS[vdb_type](dim=dim, **init_params)

    def build(self, dataset: Dataset) -> VectorDatabase:
        """Build index for the specified vectordb."""
        vdb_type = self.cfg.vectordb.type

        x = np.stack(dataset["embedding"]).astype(np.float32)
        ids = np.array(dataset["id"], dtype=np.int64)

        logger.info("Building vector database")
        init_params = self.cfg.vectordb.params.to_dict()
        vdb = self.init_vectordb(vdb_type, x.shape[1], init_params=init_params)

        with Profiler(f"build-{vdb_type}"):
            # training if needed
            vdb.train_index(x)
            vdb.upsert(ids, x)

        logger.info(f"Stats: {vdb.stats()}")

        return vdb

    def build_baseline(self, dataset: Dataset) -> VectorDatabase:
        """Build the FAISS vector database with a `IndexFlat` index as a baseline."""
        x = np.stack(dataset["embedding"]).astype(np.float32)
        metric = self.cfg.vectordb.params.metric.lower()

        # Create exact baseline
        d = x.shape[1]
        base_index = faiss.IndexFlatIP(d) if metric in ("ip", "cosine") else faiss.IndexFlatL2(d)
        index = faiss.IndexIDMap2(base_index)

        ids = np.arange(x.shape[0], dtype=np.int64)
        index.add_with_ids(x.astype(np.float32), ids)

        logger.info("Created exact baseline index")
        return index

    def search(self, dataset: Dataset, vectordb: VectorDatabase, baseline_vectordb: VectorDatabase) -> None:
        """Profile search and compute recall@K vs exact baseline."""
        params = self.cfg.vectordb.params
        model_id = self.cfg.embedding.model_id

        x = np.stack(dataset["embedding"]).astype(np.float32)

        topk = self.cfg.search.topk
        queries_file = Path(__file__).resolve().parent.parent / self.cfg.search.queries_file

        queries = [q.strip() for q in queries_file.read_text(encoding="utf-8").splitlines() if q.strip()]
        q = embed_text(queries, model_id)
        logger.info("Embedded all queries")

        # search + profile
        with Profiler(f"search-{self.cfg.vectordb.type}") as p:
            lat = []
            _, i0 = baseline_vectordb.search(q, topk)  # exact

            for i in tqdm.tqdm(range(q.shape[0])):
                t0 = time.perf_counter()
                _, _ = vectordb.search(q[i : i + 1], topk, **params.to_dict())
                lat.append((time.perf_counter() - t0) * 1000.0)

            p.sample()

        # recall@K (compare last retrieved to baseline per query)
        # For simplicity compute approximate on whole Q at once:
        _, i1 = vectordb.search(q, topk, **params.to_dict())
        rec = recall_at_k(i1, i0, topk)

        stats = {
            "vectordb": self.cfg.vectordb.type,
            "topk": topk,
            "lat_ms_avg": float(np.mean(lat)),
            "lat_ms_p50": float(np.percentile(lat, 50)),
            "lat_ms_p95": float(np.percentile(lat, 95)),
            "recall@k": rec,
            "ntotal": int(x.shape[0]),
        }

        logger.info(json.dumps(stats, indent=2))

    def update(self, dataset: Dataset, vectordb: VectorDatabase) -> None:
        """Upsert + delete small batch and re-search."""
        vdb_type = self.cfg.vectordb.type

        add_n = self.cfg.update["add_count"]
        del_n = self.cfg.update["delete_count"]

        x = np.stack(dataset["embedding"]).astype(np.float32)

        # craft new vectors by slight noise around existing (simulating fresh writes)
        rng = np.random.default_rng(42)
        add_vecs = x[:add_n].copy()
        add_vecs += rng.normal(0, 0.01, size=add_vecs.shape).astype(np.float32)
        add_vecs /= np.linalg.norm(add_vecs, axis=1, keepdims=True) + 1e-9
        add_ids = np.arange(10_000_000, 10_000_000 + add_n, dtype=np.int64)

        with Profiler(f"update-add-{vdb_type}"):
            vectordb.upsert(add_ids, add_vecs)

        with Profiler(f"update-delete-{vdb_type}"):
            del_ids = list(add_ids[:del_n])
            vectordb.delete(del_ids)

        logger.info("Update complete.", vectordb.stats())

    def run(self) -> None:
        """Run end-to-end benchmark with all steps."""
        # Download dataset
        self.download()

        # Compute embeddings
        dataset = self.embed()

        # Build baseline vector database
        baseline_vectordb = self.build_baseline(dataset)

        # Build specified vector database
        vectordb = self.build(dataset)

        # Perform search
        self.search(dataset, vectordb, baseline_vectordb)

        # Update operations
        self.update(dataset, vectordb)


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
