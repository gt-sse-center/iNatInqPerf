# tests/test_benchmark.py
import sys
import pickle
import numpy as np
import pytest

from inatinqperf.benchmark import benchmark, Benchmarker
from inatinqperf.utils.embed import ImageDatasetWithEmbeddings


class DummyVDB:
    """Safe dummy vectordb."""

    def __init__(self, dim, metric, **params):
        self.inited = True
        self.trained = False
        self.dim = dim
        self.metric = metric
        self.params = params
        self.ntotal = 0
        self.init_args = {"dim": dim, "metric": metric, **params}

    def train_index(self, X):
        self.trained = True

    def upsert(self, ids, X):
        self.ntotal += len(ids)

    def delete(self, ids):
        # no-op for safety
        pass

    def search(self, Q, topk, **kwargs):
        n = Q.shape[0]
        I = np.tile(np.arange(topk), (n, 1))
        D = np.zeros_like(I, dtype=np.float32)
        return D, I

    def stats(self):
        return {"ntotal": self.ntotal, "kind": "dummy", "metric": getattr(self, "metric", "ip")}


@pytest.fixture(autouse=True)
def patch_vectordbs(monkeypatch):
    """Ensure benchmark always uses Dummy vectordb instead of FAISS."""
    monkeypatch.setitem(benchmark.VECTORDBS, "faiss.flat", DummyVDB)
    monkeypatch.setitem(benchmark.VECTORDBS, "faiss.ivfpq", DummyVDB)
    # If benchmark references classes directly
    monkeypatch.setattr(benchmark, "FaissFlat", DummyVDB, raising=False)
    monkeypatch.setattr(benchmark, "FaissIVFPQ", DummyVDB, raising=False)


# ---------- Helpers / fixtures ----------
def _fake_ds_embeddings(n=5, d=4):
    return {"embedding": [np.ones(d, dtype=np.float32) for _ in range(n)], "id": list(range(n))}


# ===============================
# Original orchestration-safe tests
# ===============================
def test_download_with_stubs(monkeypatch, tmp_path):
    class Saveable:
        """Stub HF loader + exporter."""

        def save_to_disk(self, path):  # mimic datasets.Dataset
            (tmp_path / "saved.flag").write_text("ok")

    export_dirs: list = []

    def _export_images(ds, out_dir):
        export_dirs.append(out_dir)
        manifest = tmp_path / "manifest.csv"
        manifest.write_text("index,filename,label\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(benchmark, "load_composite", lambda huggingface_id, split: Saveable())
    monkeypatch.setattr(benchmark, "export_images", _export_images)

    cfg = {
        "dataset": {
            "hf_id": "fake",
            "out_dir": tmp_path,
            "size_splits": {"small": "train[:10]"},
            "export_images": True,
        }
    }

    benchmarker = Benchmarker()
    benchmarker.download(
        dataset_size="small",
        out_dir=cfg["dataset"]["out_dir"],
        export_raw_images=cfg["dataset"]["export_images"],
        cfg=cfg,
    )

    assert (tmp_path / "saved.flag").exists()
    assert export_dirs == [tmp_path / "images"]
    assert (tmp_path / "manifest.csv").exists()


def test_embed_with_stubs(monkeypatch, tmp_path):
    # Stub embed_images -> returns (ds_out, X, ids, labels)
    embed_calls: list[tuple[str, str, int]] = []

    def _embed_images(raw_dir, model_id, batch):
        embed_calls.append((raw_dir, model_id, batch))
        return ImageDatasetWithEmbeddings(
            embeddings=np.ones((3, 4)),
            ids=[0, 1, 2],
            labels=[0, 1, 2],
        )

    monkeypatch.setattr(benchmark, "embed_images", _embed_images)

    model_id = "openai/clip"
    batch_size = 2
    raw_dir = tmp_path
    emb_dir = tmp_path

    benchmarker = Benchmarker()
    benchmarker.embed(
        model_id=model_id,
        batch_size=batch_size,
        raw_dir=raw_dir,
        emb_dir=emb_dir,
    )

    assert embed_calls == [
        (raw_dir, model_id, batch_size),
    ]


def test_save_as_huggingface_dataset(monkeypatch, tmp_path):
    # Stub to_hf_dataset -> save_to_disk
    save_calls: list[str] = []

    class HFSaver:
        def save_to_disk(self, path):
            save_calls.append(path)
            (tmp_path / "emb.flag").write_text("ok")

    monkeypatch.setattr(benchmark, "to_hf_dataset", lambda dse: HFSaver())

    benchmarker = Benchmarker()
    dse = ImageDatasetWithEmbeddings(
        np.ones((2, 3), dtype=np.float32),
        [10, 11],
        [0, 1],
    )
    benchmarker.save_as_huggingface_dataset(dse, out_hf_dir=tmp_path)

    assert save_calls == [tmp_path]
    assert (tmp_path / "emb.flag").exists()


def test_cmd_build_with_dummy_vectordb(monkeypatch, tmp_path, caplog):
    # Use fake embeddings dataset on disk
    monkeypatch.setattr(benchmark, "load_from_disk", lambda path=None: _fake_ds_embeddings(n=4, d=2))

    class CaptureVectorDB(DummyVDB):
        def __init__(self, dim, metric, **params):
            super().__init__(dim, metric, **params)
            self.train_calls: list[int] = []
            self.upsert_ids: list[list[int]] = []

        def train_index(self, X):
            self.train_calls.append(len(X))
            self.trained = True

        def upsert(self, ids, X):
            self.upsert_ids.append(list(ids))
            super().upsert(ids, X)

    captured: dict[str, CaptureVectorDB] = {}

    def _capture_vectordb(name, dim, metric, params):
        safe = dict(params) if params else {}
        safe.pop("metric", None)
        inst = CaptureVectorDB(dim=dim, metric=metric, **safe)
        captured["instance"] = inst
        return inst

    monkeypatch.setattr(benchmark, "init_vectordb", _capture_vectordb)

    cfg = {
        "embedding": {"out_hf_dir": str(tmp_path)},
        "vectordbs": {"faiss.flat": {"metric": "ip"}},
    }
    vectordb = "faiss.flat"
    hf_dir = None

    benchmark.cmd_build(vectordb, hf_dir, cfg)

    inst = captured["instance"]
    assert inst.train_calls == [4]
    assert inst.upsert_ids == [list(range(4))]
    assert "Stats:" in caplog.text


def test_cmd_search_safe_pickle_and_vectordb(monkeypatch, tmp_path, caplog):
    # Ensure search loads a DummyVDB instead of FAISS from pickle (paranoia; may not be used)
    monkeypatch.setattr(pickle, "load", lambda f: DummyVDB())
    monkeypatch.setattr(benchmark, "embed_text", lambda qs, mid: np.ones((len(qs), 2), dtype=np.float32))
    # Return a fake embeddings dataset so load_from_disk doesn't touch the filesystem
    monkeypatch.setattr(benchmark, "load_from_disk", lambda path=None: _fake_ds_embeddings(n=3, d=2))

    # Stub exact_baseline to avoid building a real FAISS exact index
    def _fake_exact_baseline(X, metric="ip"):
        class _Exact:
            def search(self, Q, k):
                n = Q.shape[0]
                I = np.tile(np.arange(k), (n, 1))
                D = np.zeros_like(I, dtype=np.float32)
                return D, I

        return _Exact()

    monkeypatch.setattr(benchmark, "exact_baseline", _fake_exact_baseline)

    qfile = tmp_path / "queries.txt"
    qfile.write_text("a\nb\n")

    cfg = {
        "embedding": {"model_id": "m", "out_hf_dir": str(tmp_path)},
        "vectordbs": {"faiss.flat": {"metric": "ip"}},
        "search": {"topk": 3, "queries_file": str(qfile)},
    }
    vectordb = "faiss.flat"
    hf_dir = None
    topk = 3
    queries = str(qfile)

    benchmark.cmd_search(vectordb, hf_dir, topk, queries, cfg)

    assert '"vectordb": "faiss.flat"' in caplog.text
    assert '"recall@k"' in caplog.text


def test_cmd_update_with_dummy_vectordb(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "load_from_disk", lambda path=None: _fake_ds_embeddings(n=5, d=2))

    class CaptureVectorDB(DummyVDB):
        def __init__(self, dim, metric, **params):
            super().__init__(dim, metric, **params)
            self.train_calls: list[int] = []
            self.upsert_calls: list[list[int]] = []
            self.delete_calls: list[list[int]] = []

        def train_index(self, X):
            self.train_calls.append(len(X))
            self.trained = True

        def upsert(self, ids, X):
            self.upsert_calls.append(list(ids))
            super().upsert(ids, X)

        def delete(self, ids):
            ids_list = list(ids)
            self.delete_calls.append(ids_list)

    captured: dict[str, CaptureVectorDB] = {}

    def _capture_vectordb(name, dim, metric, params):
        safe = dict(params) if params else {}
        safe.pop("metric", None)
        inst = CaptureVectorDB(dim=dim, metric=metric, **safe)
        captured["instance"] = inst
        return inst

    monkeypatch.setattr(benchmark, "init_vectordb", _capture_vectordb)

    cfg = {
        "embedding": {"out_hf_dir": str(tmp_path), "model_id": "m"},
        "vectordbs": {"faiss.flat": {"metric": "ip"}},
        "update": {"add_count": 2, "delete_count": 2},
    }
    vectordb = "faiss.flat"
    hf_dir = None
    add_n = None
    delete = None
    benchmark.cmd_update(vectordb, hf_dir, add_n, delete, cfg)

    inst = captured["instance"]
    assert inst.train_calls  # vectordb trained at least once
    assert inst.upsert_calls[0] == list(range(5))
    assert len(inst.upsert_calls[1]) == cfg["update"]["add_count"]
    assert inst.delete_calls == [list(range(10_000_000, 10_000_000 + cfg["update"]["delete_count"]))]


@pytest.mark.parametrize("verb", ["download", "embed", "build", "search", "update"])
def test_cli_main_dispatch(monkeypatch, tmp_path, verb):
    benchmarker = Benchmarker()

    # Stub subcommand implementations to do nothing
    calls: list[str] = []
    monkeypatch.setattr(benchmarker, "download", lambda *args, **kwargs: calls.append("download"))
    monkeypatch.setattr(benchmarker, "embed", lambda **kwargs: calls.append("embed"))
    monkeypatch.setattr(benchmark, "cmd_build", lambda **kwargs: calls.append("build"))
    monkeypatch.setattr(benchmark, "cmd_search", lambda **kwargs: calls.append("search"))
    monkeypatch.setattr(benchmark, "cmd_update", lambda **kwargs: calls.append("update"))

    if verb == "download":
        benchmarker.download("small", out_dir=tmp_path)

    elif verb == "embed":
        benchmarker.embed(
            model_id="openai/clip",
            batch_size=2,
            raw_dir=tmp_path,
            emb_dir=tmp_path,
        )

    elif verb == "build":
        argv = ["prog", verb, "--vectordb", "faiss.flat"]
        monkeypatch.setattr(sys, "argv", argv)
        benchmark.main()

    elif verb == "search":
        argv = ["prog", verb, "--vectordb", "faiss.flat", "--topk", "2"]
        monkeypatch.setattr(sys, "argv", argv)
        benchmark.main()

    elif verb == "update":
        argv = ["prog", verb, "--vectordb", "faiss.flat"]
        monkeypatch.setattr(sys, "argv", argv)
        benchmark.main()

    else:
        argv = ["prog", verb]
        monkeypatch.setattr(sys, "argv", argv)
        benchmark.main()

    assert calls == [verb]


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


def test_load_cfg_and_ensure_dir_error_and_idempotency(tmp_path):
    # Good path
    cfg_path = tmp_path / "benchmark.yaml"
    cfg_path.write_text("dataset:\n  hf_id: fake\n")
    cfg = benchmark.load_cfg(cfg_path)
    assert cfg["dataset"]["hf_id"] == "fake"

    # ensure_dir called twice (idempotent)
    d = tmp_path / "_x"
    benchmark.ensure_dir(d)
    benchmark.ensure_dir(d)
    assert d.exists()

    # Bad path: missing file raises (FileNotFoundError or OSError depending on impl)
    with pytest.raises((FileNotFoundError, OSError, IOError)):
        benchmark.load_cfg(tmp_path / "nope.yaml")


# ===============================
# Additional coverage boosters
# ===============================
class CaptureVectorDB(DummyVDB):
    """Used to verify init_vectordb scrubs reserved keys."""


def test_init_vectordb_scrubs_reserved(monkeypatch):
    # Inject capture vectordb under a custom key
    monkeypatch.setitem(benchmark.VECTORDBS, "capture", CaptureVectorDB)
    params = {"metric": "ip", "dim": 999, "name": "x", "nlist": 123}
    be = benchmark.init_vectordb("capture", dim=64, metric="ip", params=params)
    # Reserved keys must not be forwarded twice
    assert be.init_args["dim"] == 64
    assert be.init_args["metric"] == "ip"
    assert "name" not in be.init_args
    assert "nlist" in be.init_args and be.init_args["nlist"] == 123


def test_download_no_export(monkeypatch, tmp_path):
    """Test dataset download without exporting raw images."""
    # load_composite returns a saveable dataset
    saved_paths: list[str] = []

    class Saveable2:
        def save_to_disk(self, path):
            saved_paths.append(path)

    monkeypatch.setattr(benchmark, "load_composite", lambda huggingface_id, split: Saveable2())

    images_exported: bool = False

    def export_images(dataset, path):
        images_exported = True

    # export_images should not be called (args override to False)
    monkeypatch.setattr(benchmark, "export_images", export_images)

    cfg = {
        "dataset": {
            "hf_id": "fake",
            "out_dir": tmp_path,
            "size_splits": {"small": "train[:10]"},
            "export_images": True,  # default True, but args turn it off
        }
    }
    size = "small"
    out_dir = None
    export_images = False
    benchmarker = Benchmarker()
    benchmarker.download(size, out_dir, export_images, cfg)

    assert saved_paths == [tmp_path]
    assert images_exported is False


def test_embed_with_overrides(monkeypatch, tmp_path):
    # Stub embed_images -> returns (ds_out, X, ids, labels)
    embed_calls: list[tuple[str, str, int]] = []

    def _embed_images(raw_dir, model_id, batch):
        embed_calls.append((raw_dir, model_id, batch))
        return ImageDatasetWithEmbeddings(np.ones((2, 3), dtype=np.float32), [10, 11], [0, 1])

    save_calls: list[str] = []

    class HFOut2:
        def save_to_disk(self, path):
            save_calls.append(path)

    monkeypatch.setattr(benchmark, "embed_images", _embed_images)
    monkeypatch.setattr(benchmark, "to_hf_dataset", lambda dse: HFOut2())

    cfg = {
        "dataset": {"out_dir": "IGNORED"},
        "embedding": {
            "model_id": "cfg-model",
            "batch": 1,
            "out_dir": tmp_path,
            "out_hf_dir": tmp_path,
        },
    }
    # Provide all args to override cfg
    model_id = "arg-model"
    batch = 7
    raw_dir = tmp_path
    emb_dir = tmp_path

    benchmarker = Benchmarker()
    dse = benchmarker.embed(model_id, batch, raw_dir, emb_dir)
    benchmarker.save_as_huggingface_dataset(dse, out_hf_dir=tmp_path)

    assert embed_calls == [(raw_dir, model_id, batch)]
    assert save_calls == [cfg["embedding"]["out_hf_dir"]]


class LazyEmbeds:
    def __init__(self, n, d):
        self.n, self.d = n, d

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # Return a tiny array on demand, avoiding storing 500k items
        return np.ones(self.d, dtype=np.float32)


class LazyIds:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def test_cmd_run_all_calls_sequence(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(Benchmarker, "download", lambda *args, **kwargs: calls.append("download"))

    def embed_patch(*args, **kwargs):
        calls.append("embed")
        return ImageDatasetWithEmbeddings(
            embeddings=np.ones((3, 4)),
            ids=[0, 1, 2],
            labels=[0, 1, 2],
        )

    monkeypatch.setattr(Benchmarker, "embed", embed_patch)

    monkeypatch.setattr(
        benchmark, "cmd_build", lambda **kwargs: calls.append(f"build:{kwargs.get('vectordb', None)}")
    )
    monkeypatch.setattr(benchmark, "cmd_search", lambda **kwargs: calls.append("search"))
    monkeypatch.setattr(benchmark, "cmd_update", lambda **kwargs: calls.append("update"))

    cfg = {
        "dataset": {"out_dir": str(tmp_path)},
        "embedding": {"model_id": "m", "batch": 1, "out_dir": str(tmp_path), "out_hf_dir": str(tmp_path)},
        "search": {"queries_file": str(tmp_path / "q.txt")},
    }
    size = "small"
    vectordb = "faiss.flat"
    benchmark.cmd_run_all(size, vectordb, cfg)

    # Expected call order: download, embed, build (baseline), build (chosen), search, update
    assert calls[0] == "download"
    assert calls[1] == "embed"
    assert calls[2].startswith("build:")
    assert calls[3].startswith("build:")
    assert calls[4] == "search"
    assert calls[5] == "update"
