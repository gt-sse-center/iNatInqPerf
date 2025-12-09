"""Tests for the benchmarking code."""

import numpy as np
import pytest
from collections.abc import Sequence
from datasets import Dataset as HuggingFaceDataset

from inatinqperf import adaptors
from inatinqperf.adaptors.base import SearchResult
from inatinqperf.adaptors.enums import Metric
from inatinqperf.benchmark import Benchmarker, benchmark
from inatinqperf.benchmark.configuration import VectorDatabaseParams


@pytest.fixture(name="data_path", scope="session")
def data_path_fixture(tmp_path_factory):
    """Fixture to return a temporary data path which can be used for all tests within a session.

    The common path will ensure the HuggingFace dataset isn't repeatedly downloaded.
    """
    return tmp_path_factory.mktemp("data")


@pytest.fixture(name="vector_database_params")
def vdb_params_fixture():
    params = {
        "url": "localhost",
        "port": "8000",
        "metric": Metric.INNER_PRODUCT,
        "nlist": 123,
        "m": 16,
        "nbits": 2,  # This decides the number of clusters in PQ
        "nprobe": 2,
        "index_type": "IVFPQ",
    }
    return params


@pytest.fixture(name="benchmark_module")
def mocked_benchmark_module(monkeypatch):
    def _fake_ds_embeddings(path=None, splits=None):
        n = 256
        d = 64
        rng = np.random.default_rng(42)
        data_dict = {
            "id": list(range(n)),
            "embedding": [rng.uniform(0, 100, d).astype(np.float32) for _ in range(n)],
        }

        return HuggingFaceDataset.from_dict(data_dict)

    # patch benchmark.load_huggingface_dataset
    monkeypatch.setattr(benchmark, "load_huggingface_dataset", _fake_ds_embeddings)
    return benchmark


class MockExactBaseline:
    """A mock of an exact baseline index such as FAISS Flat."""

    def search(self, q, k) -> Sequence[SearchResult]:
        ids = np.arange(k)
        scores = np.zeros_like(ids, dtype=np.float32)
        return [SearchResult(id=i, score=score) for i, score in zip(ids, scores)]


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

    assert "Dataset already exists" in caplog.text


def test_embed(config_yaml, data_path):
    benchmarker = Benchmarker(config_yaml, base_path=data_path)
    benchmarker.download()
    ds = benchmarker.embed()

    assert len(ds["embedding"]) == 256
    assert len(ds["embedding"][0]) == 512
    assert len(ds["id"]) == 256
    assert len(ds["label"]) == 256


def test_embed_preexisting(tmp_path, config_yaml, caplog, monkeypatch):
    """Test dataset download if the dataset directory already exists."""
    benchmarker = Benchmarker(config_yaml, base_path=tmp_path)

    # Create the embedding directory
    (tmp_path / benchmarker.cfg.embedding.directory).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(HuggingFaceDataset, "load_from_disk", lambda *args, **kwargs: None)

    benchmarker.embed()

    assert "Embeddings found" in caplog.text
    assert "loading instead of computing" in caplog.text


def test_save_as_huggingface_dataset(config_yaml, tmp_path):
    benchmarker = Benchmarker(config_yaml, base_path=tmp_path)

    ds = HuggingFaceDataset.from_dict(
        {
            "id": [10, 11],
            "embedding": np.random.default_rng(42).random((2, 3), dtype=np.float32).tolist(),
            "label": [0, 1],
        }
    )
    benchmarker.save_as_huggingface_dataset(ds)

    embedding_dir = tmp_path / "data" / "inquire_benchmark" / "emb"
    assert embedding_dir.exists()
    assert (embedding_dir / "dataset_info.json").exists()


def test_build(config_yaml, data_path, benchmark_module, vector_database_params):
    dataset = benchmark_module.load_huggingface_dataset(data_path)

    benchmarker = Benchmarker(config_yaml)

    benchmarker.cfg.vectordb.params = VectorDatabaseParams(**vector_database_params)

    vdb = benchmarker.build(dataset)

    assert vdb.dim == 64
    assert vdb.metric == Metric.INNER_PRODUCT
    assert vdb.nlist == 123
    assert vdb.m == 16
    assert vdb.nbits == 2
    assert vdb.nprobe == 2


def test_build_with_faiss(data_path, caplog, config_yaml, benchmark_module):
    dataset = benchmark_module.load_huggingface_dataset(data_path)
    benchmarker = Benchmarker(config_yaml, data_path)

    vdb = benchmarker.build(dataset)
    assert isinstance(vdb, adaptors.Faiss)
    assert "Stats:" in caplog.text


def test_search(config_yaml, data_path, caplog):
    """Test vector DB search."""
    benchmarker = Benchmarker(config_yaml, base_path=data_path)

    benchmarker.download()

    dataset = benchmarker.embed()
    vectordb = benchmarker.build(dataset)

    benchmarker.search(dataset, vectordb)

    # The configured index type drives the log message; assert against the configured value.
    expected_index_type = benchmarker.cfg.vectordb.params.index_type.upper()
    assert expected_index_type in caplog.text


def test_update(data_path, config_yaml, benchmark_module):
    dataset = benchmark_module.load_huggingface_dataset(data_path)
    benchmarker = Benchmarker(config_yaml, data_path)

    # Use a FLAT index during tests to avoid IVFPQ removal instability with tiny datasets.
    benchmarker.cfg.vectordb.params.index_type = "FLAT"
    benchmarker.cfg.containers = []
    benchmarker.cfg.container_network = ""

    vectordb = benchmarker.build(dataset)

    previous_total = vectordb.index.ntotal

    benchmarker.update(dataset, vectordb)

    assert (
        vectordb.index.ntotal
        == previous_total + benchmarker.cfg.update["add_count"] - benchmarker.cfg.update["delete_count"]
    )


def test_update_and_search_invokes_all(monkeypatch, config_yaml, data_path):
    """Ensure the combined post-update search operation runs both updates and search."""
    benchmarker = Benchmarker(config_yaml, data_path)

    calls: dict[str, list] = {"update": [], "search": []}

    def fake_update(dataset, db):
        calls["update"].append(db)

    def fake_search(dataset, vdb):
        calls["search"].append((vdb))

    monkeypatch.setattr(benchmarker, "update", fake_update)
    monkeypatch.setattr(benchmarker, "search", fake_search)

    dataset = object()
    vectordb = object()

    benchmarker.update_and_search(dataset, vectordb)

    assert calls["update"] == [vectordb]
    assert calls["search"] == [(vectordb)]


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
    # Disable distributed deployment knobs for unit tests to keep FAISS setup deterministic.
    benchmarker.cfg.vectordb.params.index_type = "FLAT"
    benchmarker.cfg.containers = []
    benchmarker.cfg.container_network = ""
    benchmarker.run()

    assert "faiss" in caplog.text
    # Mirror the log assertion with whatever index type the config specifies.
    expected_index_type = benchmarker.cfg.vectordb.params.index_type.upper()
    assert expected_index_type in caplog.text
    assert "topk" in caplog.text and "10" in caplog.text
