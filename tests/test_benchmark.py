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
from inatinqperf.utils.embed import ImageDatasetWithEmbeddings


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

    assert "Dataset already exists, continuing..." in caplog.text


def test_embed(config_yaml, data_path):
    benchmarker = Benchmarker(config_yaml, base_path=data_path)
    benchmarker.download()
    ds = benchmarker.embed()

    ds = ds.with_format("numpy")

    assert ds["embedding"].shape == (256, 512)
    assert len(ds["id"]) == 256
    assert len(ds["label"]) == 256


def test_embed_preexisting(tmp_path, config_yaml, caplog, monkeypatch):
    """Test dataset download if the dataset directory already exists."""
    benchmarker = Benchmarker(config_yaml, base_path=tmp_path)

    # Create the embedding directory
    (tmp_path / benchmarker.cfg.embedding.directory).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(HuggingFaceDataset, "load_from_disk", lambda *args, **kwargs: None)

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

    benchmarker.search(dataset, vectordb, MockExactBaseline())

    assert "faiss" in caplog.text
    assert "IVFPQ" in caplog.text
    assert "recall@k" in caplog.text


def test_update(data_path, config_yaml, benchmark_module):
    dataset = benchmark_module.load_huggingface_dataset(data_path)
    benchmarker = Benchmarker(config_yaml, data_path)

    vectordb = benchmarker.build(dataset)

    previous_total = vectordb.index.ntotal

    benchmarker.update(dataset, vectordb)

    assert (
        vectordb.index.ntotal
        == previous_total + benchmarker.cfg.update["add_count"] - benchmarker.cfg.update["delete_count"]
    )


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
