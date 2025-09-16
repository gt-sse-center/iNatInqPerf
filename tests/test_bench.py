# tests/test_bench.py
import io
import sys
import types
import pickle
import numpy as np
import pytest

import inatinqperf.bench.bench as bench


# ---------- Safe dummy backend ----------
class DummyBE:
    def __init__(self):
        self.inited = False
        self.trained = False
        self.ntotal = 0
        self.init_args = None

    def init(self, dim, metric, **params):
        self.inited = True
        self.dim = dim
        self.metric = metric
        self.params = params
        self.init_args = {"dim": dim, "metric": metric, **params}

    def train(self, X):
        self.trained = True

    def upsert(self, ids, X):
        self.ntotal += len(ids)

    def delete(self, ids):
        # no-op for safety
        pass

    def search(self, Q, topk, **kwargs):
        n = Q.shape[0]
        I = np.tile(np.arange(topk), (n, 1))
        D = np.zeros_like(I, dtype="float32")
        return D, I

    def stats(self):
        return {"ntotal": self.ntotal, "kind": "dummy", "metric": getattr(self, "metric", "ip")}


@pytest.fixture(autouse=True)
def patch_backends(monkeypatch):
    """Ensure bench always uses Dummy backend instead of FAISS."""
    monkeypatch.setitem(bench.ALL_BACKENDS, "faiss.flat", DummyBE)
    monkeypatch.setitem(bench.ALL_BACKENDS, "faiss.ivfpq", DummyBE)
    # If bench references classes directly
    monkeypatch.setattr(bench, "FaissFlat", DummyBE, raising=False)
    monkeypatch.setattr(bench, "FaissIVFPQ", DummyBE, raising=False)


# ---------- Helpers / fixtures ----------
def _fake_ds_embeddings(n=5, d=4):
    return {"embedding": [np.ones(d, dtype="float32") for _ in range(n)], "id": list(range(n))}


# ===============================
# Original orchestration-safe tests
# ===============================
def test_cmd_download_with_stubs(monkeypatch, tmp_path):
    # Stub HF loader + exporter
    class Saveable:
        def save_to_disk(self, path):  # mimic datasets.Dataset
            (tmp_path / "saved.flag").write_text("ok")

    monkeypatch.setattr(bench, "load_composite", lambda hf_id, split: Saveable())
    monkeypatch.setattr(bench, "export_images", lambda ds, out_dir: tmp_path / "manifest.csv")

    cfg = {
        "dataset": {
            "hf_id": "fake",
            "out_dir": str(tmp_path),
            "size_splits": {"small": "train[:10]"},
            "export_images": True,
        }
    }
    args = types.SimpleNamespace(size="small", out_dir=None, export_images=None)

    bench.cmd_download(args, cfg)
    # both code paths executed without exceptions


def test_cmd_embed_with_stubs(monkeypatch, tmp_path):
    # Stub embed_images -> returns (ds_out, X, ids, labels)
    monkeypatch.setattr(
        bench,
        "embed_images",
        lambda raw_dir, model_id, batch: ([], np.ones((3, 4), dtype="float32"), [0, 1, 2], [0, 1, 2]),
    )

    # Stub to_hf_dataset -> save_to_disk
    class HFSaver:
        def save_to_disk(self, path):
            (tmp_path / "emb.flag").write_text("ok")

    monkeypatch.setattr(bench, "to_hf_dataset", lambda X, ids, labels: HFSaver())

    cfg = {
        "dataset": {"out_dir": str(tmp_path)},
        "embedding": {
            "model_id": "openai/clip",
            "batch": 2,
            "out_dir": str(tmp_path),
            "out_hf_dir": str(tmp_path),
        },
    }
    args = types.SimpleNamespace(model_id=None, batch=None, raw_dir=None, emb_dir=None)

    bench.cmd_embed(args, cfg)


def test_cmd_build_with_dummy_backend(monkeypatch, tmp_path, capsys):
    # Use fake embeddings dataset on disk
    monkeypatch.setattr(bench, "load_from_disk", lambda path=None: _fake_ds_embeddings(n=4, d=2))

    cfg = {
        "embedding": {"out_hf_dir": str(tmp_path)},
        "backends": {"faiss.flat": {"metric": "ip"}},
    }
    args = types.SimpleNamespace(backend="faiss.flat", hf_dir=None)

    bench.cmd_build(args, cfg)
    out = capsys.readouterr().out
    assert "Stats:" in out  # hit printing path


def test_cmd_search_safe_pickle_and_backend(monkeypatch, tmp_path, capsys):
    # Ensure search loads a DummyBE instead of FAISS from pickle (paranoia; may not be used)
    monkeypatch.setattr(pickle, "load", lambda f: DummyBE())
    monkeypatch.setattr(bench, "embed_text", lambda qs, mid: np.ones((len(qs), 2), dtype="float32"))
    # Return a fake embeddings dataset so load_from_disk doesn't touch the filesystem
    monkeypatch.setattr(bench, "load_from_disk", lambda path=None: _fake_ds_embeddings(n=3, d=2))

    # Stub exact_baseline to avoid building a real FAISS exact index
    def _fake_exact_baseline(X, metric="ip"):
        class _Exact:
            def search(self, Q, k):
                n = Q.shape[0]
                I = np.tile(np.arange(k), (n, 1))
                D = np.zeros_like(I, dtype="float32")
                return D, I

        return _Exact()

    monkeypatch.setattr(bench, "exact_baseline", _fake_exact_baseline)

    qfile = tmp_path / "queries.txt"
    qfile.write_text("a\nb\n")

    cfg = {
        "embedding": {"model_id": "m", "out_hf_dir": str(tmp_path)},
        "backends": {"faiss.flat": {"metric": "ip"}},
        "search": {"topk": 3, "queries_file": str(qfile)},
    }
    args = types.SimpleNamespace(backend="faiss.flat", hf_dir=None, topk=3, queries=str(qfile))

    bench.cmd_search(args, cfg)
    out = capsys.readouterr().out
    # The search command prints a JSON summary (and [PROFILE] line), not 'Stats:' in this path
    assert '"backend": "faiss.flat"' in out
    assert '"recall@k"' in out


def test_cmd_update_with_dummy_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "load_from_disk", lambda path=None: _fake_ds_embeddings(n=5, d=2))
    cfg = {
        "embedding": {"out_hf_dir": str(tmp_path), "model_id": "m"},
        "backends": {"faiss.flat": {"metric": "ip"}},
        "update": {"add_count": 2, "delete_count": 2},
    }
    args = types.SimpleNamespace(backend="faiss.flat", hf_dir=None, add=None, delete=None)
    bench.cmd_update(args, cfg)


@pytest.mark.parametrize("verb", ["download", "embed", "build", "search", "update"])
def test_cli_main_dispatch(monkeypatch, tmp_path, verb):
    # Stub subcommand implementations to do nothing
    monkeypatch.setattr(bench, "cmd_download", lambda a, c: None)
    monkeypatch.setattr(bench, "cmd_embed", lambda a, c: None)
    monkeypatch.setattr(bench, "cmd_build", lambda a, c: None)
    monkeypatch.setattr(bench, "cmd_search", lambda a, c: None)
    monkeypatch.setattr(bench, "cmd_update", lambda a, c: None)

    if verb == "download":
        argv = ["prog", verb, "--size", "small"]
    elif verb == "build":
        argv = ["prog", verb, "--backend", "faiss.flat"]
    elif verb == "search":
        argv = ["prog", verb, "--backend", "faiss.flat", "--topk", "2"]
    elif verb == "update":
        argv = ["prog", verb, "--backend", "faiss.flat"]
    else:
        argv = ["prog", verb]
    monkeypatch.setattr(sys, "argv", argv)
    bench.main()


# ---------- Edge cases for helpers ----------
def test_recall_at_k_edges():
    # No hits when there are no neighbors (1 row, 0 columns -> denominator = 1*k)
    I_true = np.empty((1, 0), dtype=int)
    I_test = np.empty((1, 0), dtype=int)
    assert bench.recall_at_k(I_true, I_test, 1) == 0.0

    # k larger than available neighbors
    I_true = np.array([[0]], dtype=int)
    I_test = np.array([[0, 1, 2]], dtype=int)
    assert 0.0 <= bench.recall_at_k(I_true, I_test, 5) <= 1.0


def test_load_cfg_and_ensure_dir_error_and_idempotency(tmp_path):
    # Good path
    cfg_path = tmp_path / "bench.yaml"
    cfg_path.write_text("dataset:\n  hf_id: fake\n")
    cfg = bench.load_cfg(cfg_path)
    assert cfg["dataset"]["hf_id"] == "fake"

    # ensure_dir called twice (idempotent)
    d = tmp_path / "_x"
    bench.ensure_dir(d)
    bench.ensure_dir(d)
    assert d.exists()

    # Bad path: missing file raises (FileNotFoundError or OSError depending on impl)
    with pytest.raises((FileNotFoundError, OSError, IOError)):
        bench.load_cfg(tmp_path / "nope.yaml")


# ===============================
# Additional coverage boosters
# ===============================
class CaptureBE(DummyBE):
    """Used to verify _init_backend scrubs reserved keys."""

    pass


def test_init_backend_scrubs_reserved(monkeypatch):
    # Inject capture backend under a custom key
    monkeypatch.setitem(bench.ALL_BACKENDS, "capture", CaptureBE)
    params = {"metric": "ip", "dim": 999, "name": "x", "nlist": 123}
    be = bench._init_backend("capture", dim=64, metric="ip", params=params)
    # Reserved keys must not be forwarded twice
    assert be.init_args["dim"] == 64
    assert be.init_args["metric"] == "ip"
    assert "name" not in be.init_args
    assert "nlist" in be.init_args and be.init_args["nlist"] == 123


class Saveable2:
    def save_to_disk(self, path):
        # simulate datasets.Dataset.save_to_disk
        return None


def test_cmd_download_no_export(monkeypatch, tmp_path):
    # load_composite returns a saveable dataset
    monkeypatch.setattr(bench, "load_composite", lambda hf_id, split: Saveable2())
    # export_images should not be called (args override to False)
    monkeypatch.setattr(
        bench,
        "export_images",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not export")),
    )

    cfg = {
        "dataset": {
            "hf_id": "fake",
            "out_dir": str(tmp_path),
            "size_splits": {"small": "train[:10]"},
            "export_images": True,  # default True, but args turn it off
        }
    }
    args = types.SimpleNamespace(size="small", out_dir=None, export_images=False)
    bench.cmd_download(args, cfg)


class HFOut2:
    def save_to_disk(self, path):
        # mimic datasets.DatasetDict.save_to_disk
        return None


def test_cmd_embed_with_overrides(monkeypatch, tmp_path):
    # Stub embed_images -> returns (ds_out, X, ids, labels)
    monkeypatch.setattr(
        bench,
        "embed_images",
        lambda raw_dir, model_id, batch: ([], np.ones((2, 3), dtype="float32"), [10, 11], [0, 1]),
    )
    monkeypatch.setattr(bench, "to_hf_dataset", lambda X, ids, labels: HFOut2())

    cfg = {
        "dataset": {"out_dir": "IGNORED"},
        "embedding": {
            "model_id": "cfg-model",
            "batch": 1,
            "out_dir": str(tmp_path),
            "out_hf_dir": str(tmp_path),
        },
    }
    # Provide all args to override cfg
    args = types.SimpleNamespace(
        model_id="arg-model",
        batch=7,
        raw_dir=str(tmp_path),
        emb_dir=str(tmp_path),
    )
    bench.cmd_embed(args, cfg)


class LazyEmbeds:
    def __init__(self, n, d):
        self.n, self.d = n, d

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # Return a tiny array on demand, avoiding storing 500k items
        return np.ones(self.d, dtype="float32")


class LazyIds:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


# def test_cmd_build_triggers_subsample(monkeypatch, tmp_path):
#     n = 500001  # force X.shape[0] >= 500000 path
#     d = 2
#     ds = {"embedding": LazyEmbeds(n, d), "id": LazyIds(n)}
#     monkeypatch.setattr(bench, "load_from_disk", lambda p=None: ds)

#     cfg = {
#         "embedding": {"out_hf_dir": str(tmp_path)},
#         "backends": {"faiss.flat": {"metric": "ip"}},
#     }
#     args = types.SimpleNamespace(backend="faiss.flat", hf_dir=None)
#     bench.cmd_build(args, cfg)


def test_cmd_run_all_calls_sequence(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(bench, "cmd_download", lambda a, c: calls.append("download"))
    monkeypatch.setattr(bench, "cmd_embed", lambda a, c: calls.append("embed"))
    monkeypatch.setattr(bench, "cmd_build", lambda a, c: calls.append(f"build:{getattr(a, 'backend', None)}"))
    monkeypatch.setattr(bench, "cmd_search", lambda a, c: calls.append("search"))
    monkeypatch.setattr(bench, "cmd_update", lambda a, c: calls.append("update"))

    cfg = {
        "dataset": {"out_dir": str(tmp_path)},
        "embedding": {"model_id": "m", "batch": 1, "out_dir": str(tmp_path), "out_hf_dir": str(tmp_path)},
        "search": {"queries_file": str(tmp_path / "q.txt")},
    }
    args = types.SimpleNamespace(size="small", backend="faiss.flat")
    bench.cmd_run_all(args, cfg)

    # Expected call order: download, embed, build (baseline), build (chosen), search, update
    assert calls[0] == "download"
    assert calls[1] == "embed"
    assert calls[2].startswith("build:")
    assert calls[3].startswith("build:")
    assert calls[4] == "search"
    assert calls[5] == "update"
