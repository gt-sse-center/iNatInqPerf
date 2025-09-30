# tests/test_benchmark.py
import sys
import pickle
import numpy as np
import pytest

from inatinqperf.benchmark import benchmark, Benchmarker
from inatinqperf.utils.embed import ImageDatasetWithEmbeddings


# ---------- Helpers / fixtures ----------
def _fake_ds_embeddings(n=5, d=4):
    return {"embeddings": [np.ones(d, dtype=np.float32) for _ in range(n)], "ids": list(range(n))}


class DummyVectorDB:
    def __init__(self, dim, metric="ip", **params):
        self.dim = dim
        self.metric = metric
        self.params = params
        self.ntotal = 0

        self.trained = False

        self.train_calls: list[int] = []
        self.upsert_calls: list[list[int]] = []
        self.delete_calls: list[list[int]] = []

    def train_index(self, X):
        self.train_calls.append(len(X))
        self.trained = True

    def upsert(self, ids, X):
        self.upsert_calls.append(list(ids))

    def delete(self, ids):
        ids_list = list(ids)
        self.delete_calls.append(ids_list)

    def search(self, Q, topk, **kwargs):
        n = Q.shape[0]
        I = np.tile(np.arange(topk), (n, 1))
        D = np.zeros_like(I, dtype=np.float32)
        return D, I

    def stats(self):
        return {"ntotal": self.ntotal, "kind": "dummy", "metric": getattr(self, "metric", "ip")}


def _capture_vectordb(name, dim, init_params):
    inst = DummyVectorDB(dim=dim, **init_params)
    return inst


class MockExactBaseline:
    def search(self, Q, k):
        n = Q.shape[0]
        I = np.tile(np.arange(k), (n, 1))
        D = np.zeros_like(I, dtype=np.float32)
        return D, I


# ===============================
# Original orchestration-safe tests
# ===============================
def test_download(config_yaml, tmp_path):
    benchmarker = Benchmarker(config_yaml, base_path=tmp_path)
    benchmarker.download()

    export_dir = tmp_path / benchmarker.cfg.dataset.directory / "images"
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


def test_embed(monkeypatch, config_yaml, tmp_path):
    benchmarker = Benchmarker(config_yaml, base_path=tmp_path)
    benchmarker.download()
    ds = benchmarker.embed()

    ds = ds.with_format("numpy")

    assert ds["embeddings"].shape == (200, 512)
    assert len(ds["ids"]) == 200
    assert len(ds["labels"]) == 200


def test_embed_preexisting(tmp_path, config_yaml, caplog, monkeypatch):
    """Test dataset download if the dataset directory already exists."""
    benchmarker = Benchmarker(config_yaml, base_path=tmp_path)

    # Create the embedding directory
    (tmp_path / benchmarker.cfg.embedding.directory).mkdir(parents=True, exist_ok=True)

    from datasets import Dataset

    monkeypatch.setattr(Dataset, "load_from_disk", lambda *args, **kwargs: None)

    benchmarker.embed()

    assert "Embeddings found, loading instead of computing" in caplog.text


def test_save_as_huggingface_dataset(config_yaml, tmp_path):
    benchmarker = Benchmarker(config_yaml, base_path=tmp_path)

    dse = ImageDatasetWithEmbeddings(
        np.ones((2, 3), dtype=np.float32),
        [10, 11],
        [0, 1],
    )
    benchmarker.save_as_huggingface_dataset(dse)

    embedding_dir = tmp_path / "data" / "inquire_benchmark" / "emb"
    assert embedding_dir.exists()
    assert (embedding_dir / "dataset_info.json").exists()


def test_build_with_dummy_vectordb(monkeypatch, tmp_path, caplog, config_yaml):
    # Use fake embeddings dataset on disk
    monkeypatch.setattr(
        benchmark, "load_huggingface_dataset", lambda path=None: _fake_ds_embeddings(n=4, d=2)
    )
    dataset = benchmark.load_huggingface_dataset(tmp_path)

    benchmarker = Benchmarker(config_yaml, tmp_path)
    monkeypatch.setattr(benchmarker, "init_vectordb", _capture_vectordb)

    vdb = benchmarker.build(dataset)

    assert vdb.train_calls == [4]
    assert vdb.upsert_calls == [list(range(4))]
    assert "Stats:" in caplog.text


def test_search_safe_pickle_and_vectordb(monkeypatch, tmp_path, caplog, config_yaml):
    # Ensure search loads a DummyVDB instead of FAISS from pickle
    monkeypatch.setattr(
        pickle,
        "load",
        lambda f: DummyVectorDB(dim=512, metric="ip", params={"metric": "ip", "nlist": 123, "m": 16}),
    )
    monkeypatch.setattr(benchmark, "embed_text", lambda qs, mid: np.ones((len(qs), 2), dtype=np.float32))
    # Return a fake embeddings dataset so load_huggingface_dataset doesn't touch the filesystem
    monkeypatch.setattr(
        benchmark, "load_huggingface_dataset", lambda path=None: _fake_ds_embeddings(n=3, d=2)
    )

    qfile = tmp_path / "queries.txt"
    qfile.write_text("a\nb\n")

    dataset = benchmark.load_huggingface_dataset(tmp_path)

    benchmarker = Benchmarker(config_yaml)
    monkeypatch.setattr(benchmarker, "init_vectordb", _capture_vectordb)
    vectordb = benchmarker.build(dataset)

    benchmarker.search(dataset, vectordb, MockExactBaseline())

    assert '"vectordb": "faiss.ivfpq"' in caplog.text
    assert '"recall@k"' in caplog.text


def test_update_with_dummy_vectordb(monkeypatch, tmp_path, config_yaml):
    monkeypatch.setattr(
        benchmark, "load_huggingface_dataset", lambda path=None: _fake_ds_embeddings(n=5, d=2)
    )

    dataset = benchmark.load_huggingface_dataset(tmp_path)

    benchmarker = Benchmarker(config_yaml, tmp_path)
    monkeypatch.setattr(benchmarker, "init_vectordb", _capture_vectordb)
    vectordb = benchmarker.build(dataset)

    benchmarker.update(dataset, vectordb)

    assert vectordb.train_calls  # vectordb trained at least once
    assert vectordb.upsert_calls[0] == list(range(5))
    assert len(vectordb.upsert_calls[1]) == benchmarker.cfg.update["add_count"]
    assert vectordb.delete_calls == [
        list(range(10_000_000, 10_000_000 + benchmarker.cfg.update["delete_count"]))
    ]


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


def test_load_cfg(tmp_path, config_yaml):
    benchmarker = Benchmarker(config_yaml, base_path=tmp_path)
    assert benchmarker.cfg.dataset.dataset_id == "sagecontinuum/INQUIRE-Benchmark-small"

    # Bad path: missing file raises (FileNotFoundError or OSError depending on impl)
    with pytest.raises((FileNotFoundError, OSError, IOError)):
        Benchmarker(tmp_path / "nope.yaml", base_path=tmp_path)


def test_init_vectordb(monkeypatch, config_yaml):
    params = {"metric": "ip", "nlist": 123, "m": 16}

    benchmarker = Benchmarker(config_yaml)
    vdb = benchmarker.init_vectordb("faiss.ivfpq", dim=64, init_params=params)

    assert vdb.dim == 64
    assert vdb.metric == "ip"
    assert vdb.nlist == 123
    assert vdb.m == 16


def test_run_all(config_yaml, tmp_path, caplog):
    benchmarker = Benchmarker(config_yaml, base_path=tmp_path)
    # Change nbits so we have a smaller number of clusters
    benchmarker.cfg.vectordb.params.nbits = 2
    benchmarker.run()

    assert '"vectordb": "faiss.ivfpq"' in caplog.text
    assert '"topk": 10' in caplog.text
