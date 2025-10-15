"""Tests for the benchmarking code."""

import numpy as np
import pytest
import requests
from collections.abc import Sequence
from datasets import Dataset

from inatinqperf import adaptors
from inatinqperf.adaptors.base import DataPoint, SearchResult, VectorDatabase
from inatinqperf.adaptors.metric import Metric
from inatinqperf.benchmark import Benchmarker, benchmark
from inatinqperf.benchmark.configuration import VectorDatabaseParams
from inatinqperf.utils.embed import ImageDatasetWithEmbeddings


@pytest.fixture(name="data_path", scope="session")
def data_path_fixture(tmp_path_factory):
    """Fixture to return a temporary data path which can be used for all tests within a session.

    The common path will ensure the HuggingFace dataset isn't repeatedly downloaded.
    """
    return tmp_path_factory.mktemp("data")


# ---------- Helpers / fixtures ----------
def _fake_ds_embeddings(n=5, d=4):
    rng = np.random.default_rng(42)
    return {"embedding": [rng.uniform(0, 100, d).astype(np.float32) for _ in range(n)], "id": list(range(n))}


class DummyVectorDB(VectorDatabase):
    """A dummy vector database for mocking."""

    def __init__(self, dataset, metric=Metric.INNER_PRODUCT, **params):
        self.metric = metric
        self.params = params
        self.ntotal = 0

        self.initialized = True

        self.upsert_called = False
        self.num_upserted_points = 0

        self.delete_called = False
        self.num_deleted_points = 0

    def upsert(self, x: Sequence[DataPoint]):
        self.upsert_called = True
        self.num_upserted_points = len(x)

    def search(self, q, topk, **kwargs):
        return [SearchResult(id=i, score=0) for i in np.arange(topk)]

    def delete(self, ids):
        self.delete_called = True
        self.num_deleted_points = len(ids)

    def stats(self):
        return {
            "ntotal": self.ntotal,
            "kind": "dummy",
            "metric": str(getattr(self, "metric", "ip")),
        }


class MockExactBaseline:
    """A mock of an exact baseline index such as FAISS Flat."""

    def search(self, q, k) -> Sequence[SearchResult]:
        ids = np.arange(k)
        scores = np.zeros_like(ids, dtype=np.float32)
        return [SearchResult(id=i, score=score) for i, score in zip(ids, scores)]


@pytest.fixture(name="mocked_benchmark_module")
def mocked_benchmark_fixture(monkeypatch):
    # Use fake embeddings dataset on disk
    monkeypatch.setattr(
        benchmark, "load_huggingface_dataset", lambda path=None: _fake_ds_embeddings(n=9984, d=64)
    )
    return benchmark


def test_load_cfg(config_yaml, data_path):
    benchmarker = Benchmarker(config_yaml, base_path=data_path)
    assert benchmarker.cfg.dataset.dataset_id == "sagecontinuum/INQUIRE-Benchmark-small"

    # Bad path: missing file raises (FileNotFoundError or OSError depending on impl)
    with pytest.raises((FileNotFoundError, OSError, IOError)):
        Benchmarker(data_path / "nope.yaml", base_path=data_path)


def test_download(config_yaml, data_path):
    benchmarker = Benchmarker(config_yaml, base_path=data_path)
    benchmarker.download()

    export_dir = data_path / benchmarker.cfg.dataset.directory / "images"
    assert export_dir.exists()
    assert (export_dir / "manifest.csv").exists()


def test_download_no_export(tmp_path, config_yaml):
    """Test dataset download without exporting raw images."""
    benchmarker = Benchmarker(config_yaml, base_path=tmp_path)
    benchmarker.cfg.dataset.export_images = False

    benchmarker.download()

    assert not (tmp_path / benchmarker.cfg.dataset.directory / "images").exists()


def test_download_preexisting(tmp_path, config_yaml, caplog):
    """Test dataset download if the dataset directory already exists."""
    benchmarker = Benchmarker(config_yaml, base_path=tmp_path)
    benchmarker.cfg.dataset.export_images = False

    # Create the dataset directory
    (tmp_path / benchmarker.cfg.dataset.directory).mkdir(parents=True, exist_ok=True)

    benchmarker.download()

    assert "Dataset already exists, continuing..." in caplog.text


def test_embed(config_yaml, data_path):
    benchmarker = Benchmarker(config_yaml, base_path=data_path)
    benchmarker.download()
    ds = benchmarker.embed()

    ds = ds.with_format("numpy")

    assert ds["embedding"].shape == (200, 512)
    assert len(ds["id"]) == 200
    assert len(ds["label"]) == 200


def test_embed_preexisting(tmp_path, config_yaml, caplog, monkeypatch):
    """Test dataset download if the dataset directory already exists."""
    benchmarker = Benchmarker(config_yaml, base_path=tmp_path)

    # Create the embedding directory
    (tmp_path / benchmarker.cfg.embedding.directory).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(Dataset, "load_from_disk", lambda *args, **kwargs: None)

    benchmarker.embed()

    assert "Embeddings found, loading instead of computing" in caplog.text


def test_save_as_huggingface_dataset(config_yaml, tmp_path):
    benchmarker = Benchmarker(config_yaml, base_path=tmp_path)

    dse = ImageDatasetWithEmbeddings(
        np.random.default_rng(42).random((2, 3), dtype=np.float32),
        [10, 11],
        [0, 1],
    )
    benchmarker.save_as_huggingface_dataset(dse)

    embedding_dir = tmp_path / "data" / "inquire_benchmark" / "emb"
    assert embedding_dir.exists()
    assert (embedding_dir / "dataset_info.json").exists()


def test_build(config_yaml, data_path, mocked_benchmark_module):
    dataset = mocked_benchmark_module.load_huggingface_dataset(data_path)

    benchmarker = Benchmarker(config_yaml)

    params = {"metric": Metric.INNER_PRODUCT, "nlist": 123, "m": 16, "index_type": "IVFPQ"}
    benchmarker.cfg.vectordb.params = VectorDatabaseParams(**params)

    vdb = benchmarker.build(dataset)

    assert vdb.dim == 64
    assert vdb.metric == Metric.INNER_PRODUCT
    assert vdb.nlist == 123
    assert vdb.m == 16


def test_build_with_dummy_vectordb(monkeypatch, data_path, caplog, config_yaml, mocked_benchmark_module):
    monkeypatch.setitem(adaptors.VECTORDBS, "faiss", DummyVectorDB)
    benchmarker = Benchmarker(config_yaml, data_path)

    dataset = mocked_benchmark_module.load_huggingface_dataset(data_path)

    vdb = benchmarker.build(dataset)

    assert vdb.initialized
    assert "Stats:" in caplog.text


def test_search(config_yaml, data_path, caplog):
    """Test vector DB search."""
    benchmarker = Benchmarker(config_yaml, base_path=data_path)

    benchmarker.download()

    dataset = benchmarker.embed()
    vectordb = benchmarker.build(dataset)

    benchmarker.search(dataset, vectordb, MockExactBaseline())

    assert "faiss" in caplog.text
    assert "IVFPQ" in caplog.text
    assert "recall@k" in caplog.text


def test_update_with_dummy_vectordb(monkeypatch, data_path, config_yaml, mocked_benchmark_module):
    monkeypatch.setitem(adaptors.VECTORDBS, "faiss", DummyVectorDB)

    benchmarker = Benchmarker(config_yaml, data_path)

    dataset = mocked_benchmark_module.load_huggingface_dataset(data_path)
    vectordb = benchmarker.build(dataset)

    benchmarker.update(dataset, vectordb)

    assert vectordb.upsert_called
    assert vectordb.num_upserted_points == benchmarker.cfg.update["add_count"]

    assert vectordb.delete_called
    assert vectordb.num_deleted_points == benchmarker.cfg.update["delete_count"]


# ---------- Edge cases for helpers ----------
def test_recall_at_k_edges():
    # No hits when there are no neighbors (1 row, 0 columns -> denominator = 1*k)
    I_true = np.empty((1, 0), dtype=int)
    I_test = np.empty((1, 0), dtype=int)
    assert benchmark.recall_at_k(I_true, I_test, 1) == 0.0

    # k larger than available neighbors
    I_true = np.array([[0]], dtype=int)
    I_test = np.array([[0, 1, 2]], dtype=int)
    assert 0.0 <= benchmark.recall_at_k(I_true, I_test, 5) <= 1.0


def test_run_all(config_yaml, tmp_path, caplog):
    benchmarker = Benchmarker(config_yaml, base_path=tmp_path)
    benchmarker.run()

    assert "faiss" in caplog.text
    assert "IVFPQ" in caplog.text
    assert "topk" in caplog.text and "10" in caplog.text
