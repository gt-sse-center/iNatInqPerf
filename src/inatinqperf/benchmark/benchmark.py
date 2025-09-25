"""Backend-agnostic benchmark orchestrator.

Subcommands:
  download  -> HF dataset + optional image export
  embed     -> CLIP embeddings, saves HF dataset with 'embedding'
  build     -> build index on chosen backend
  search    -> profile search + recall@K vs Flat
  update    -> upsert/delete small batch and re-search
  run-all   -> download->embed->build(Faiss Flat + chosen backend)->search->update
"""

import argparse  # TODO(Varun): Consider using `typer` instead
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml
from datasets import load_from_disk
from loguru import logger

from inatinqperf.adaptors import BACKENDS
from inatinqperf.utils.dataio import export_images, load_composite
from inatinqperf.utils.embed import embed_images, embed_text, to_hf_dataset
from inatinqperf.utils.profiler import Profiler

# Get the `inatinqperf` directory which is the grandparent directory of this file.
ROOT = Path(__file__).resolve().parents[1]

SAMPLE_SIZE = 500_000  # max samples for training if needed
BENCHMARK_CFG = ROOT / "configs" / "benchmark.yaml"


def load_cfg(path: Path) -> Mapping[str, object]:
    """Load YAML config file."""
    logger.info(f"Loading config: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> Path:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def exact_baseline(x: np.ndarray, metric: str) -> faiss.Index:
    """Exact baseline index using FAISS IndexFlat*."""
    d = x.shape[1]
    base = faiss.IndexFlatIP(d) if metric in ("ip", "cosine") else faiss.IndexFlatL2(d)
    index = faiss.IndexIDMap2(base)
    ids = np.arange(x.shape[0], dtype="int64")
    index.add_with_ids(x.astype("float32"), ids)
    return index


def recall_at_k(approx_i: np.ndarray, exact_i: np.ndarray, k: int) -> float:
    """Compute recall@K between two sets of indices."""
    hits = 0
    for i in range(approx_i.shape[0]):
        hits += len(set(approx_i[i, :k]).intersection(set(exact_i[i, :k])))
    return hits / float(approx_i.shape[0] * k)


def cmd_download(size: str, out_dir: Path, perform_export_images: bool, cfg: Mapping[str, Any]) -> None:  # noqa: FBT001
    """Download HF dataset and optionally export images."""
    # TODO(Varun): Make all the cmd_* functions take a class encapsulating the arguments.
    hf_id = cfg["dataset"]["hf_id"]
    out_dir = out_dir or cfg["dataset"]["out_dir"]
    export = (
        perform_export_images
        if perform_export_images is not None
        else cfg["dataset"].get("export_images", True)
    )
    split_map = cfg["dataset"]["size_splits"]
    split = split_map.get(size, split_map["small"])
    ensure_dir(out_dir)
    with Profiler(f"download-{hf_id}-{size}"):
        ds = load_composite(hf_id, split)
        ds.save_to_disk(out_dir)
        if export:
            export_dir = Path(out_dir) / "images"
            manifest = export_images(ds, export_dir)
            logger.info(f"Exported images to: {export_dir}\nManifest: {manifest}")
    logger.info(f"Saved HuggingFace dataset to: {out_dir}")


def cmd_embed(model_id: int, batch_size: int, raw_dir: Path, emb_dir: Path, cfg: Mapping[str, Any]) -> None:
    """Compute CLIP embeddings and save HF dataset with 'embedding'."""
    model_id = model_id or cfg["embedding"]["model_id"]
    batch_size = batch_size or int(cfg["embedding"]["batch"])
    raw_dir = raw_dir or cfg["dataset"]["out_dir"]
    emb_dir = emb_dir or cfg["embedding"]["out_dir"]
    out_hf_dir = cfg["embedding"]["out_hf_dir"]
    ensure_dir(emb_dir)
    ensure_dir(out_hf_dir)
    with Profiler("embed-images"):
        dataset_with_embeddings = embed_images(raw_dir, model_id, batch_size)
        x = dataset_with_embeddings.dataset
        ids = dataset_with_embeddings.ids
        labels = dataset_with_embeddings.labels
        ds2 = to_hf_dataset(x, ids, labels)
        ds2.save_to_disk(out_hf_dir)
        logger.info(f"Embeddings: {x.shape} -> {out_hf_dir}")


def _init_backend(backend_name: str, dim: int, metric: str, params: dict[str, object]) -> dict:
    """Instantiate backend, scrubbing reserved keys from params.

    Prevents errors like: TypeError: ... init() got multiple values for keyword 'metric'.
    """
    backend = BACKENDS[backend_name]

    # Avoid passing duplicate values for explicit kwargs
    safe_params = dict(params) if params else {}
    for k in ("metric", "dim", "name"):
        safe_params.pop(k, None)
    return backend(dim=dim, metric=metric, **safe_params)


def cmd_build(backend: str, hf_dir: Path, cfg: Mapping[str, Any]) -> None:
    """Build index for a backend."""
    params = cfg["backends"][backend]
    hf_dir = hf_dir or cfg["embedding"]["out_hf_dir"]
    ds = load_from_disk(hf_dir)
    x = np.stack(ds["embedding"]).astype("float32")
    metric = params.get("metric", "ip").lower()
    be = _init_backend(backend, x.shape[1], metric, params)
    with Profiler(f"build-{backend}"):
        # training if needed
        be.train(
            x
            if x.shape[0] < SAMPLE_SIZE
            else x[np.random.default_rng().choice(x.shape[0], SAMPLE_SIZE, replace=False)]
        )
        ids = np.array(ds["id"], dtype="int64")
        be.upsert(ids, x)
    logger.info("Stats:", be.stats())


def cmd_search(backend: str, hf_dir: Path, topk: int, queries: Sequence[str], cfg: Mapping[str, Any]) -> None:
    """Profile search and compute recall@K vs exact baseline."""
    params = cfg["backends"][backend]
    hf_dir = hf_dir or cfg["embedding"]["out_hf_dir"]
    topk = topk or int(cfg["search"]["topk"])
    queries_file = queries or cfg["search"]["queries_file"]
    model_id = cfg["embedding"]["model_id"]

    ds = load_from_disk(hf_dir)
    x = np.stack(ds["embedding"]).astype("float32")
    ids = np.array(ds["id"], dtype="int64")
    metric = params.get("metric", "ip").lower()
    # exact baseline
    base = exact_baseline(x, metric="ip" if metric in ("ip", "cosine") else "l2")
    # backend
    be = _init_backend(backend, x.shape[1], "ip" if metric in ("ip", "cosine") else "l2", params)
    be.train(
        x
        if x.shape[0] < SAMPLE_SIZE
        else x[np.random.default_rng().choice(x.shape[0], SAMPLE_SIZE, replace=False)]
    )
    be.upsert(ids, x)

    queries = [q.strip() for q in Path(queries_file).read_text(encoding="utf-8").splitlines() if q.strip()]
    q = embed_text(queries, model_id)

    # search + profile
    with Profiler(f"search-{backend}") as p:
        lat = []
        _, i0 = base.search(q, topk)  # exact
        for i in range(q.shape[0]):
            t0 = time.perf_counter()
            _, _ = be.search(q[i : i + 1], topk, **params)
            lat.append((time.perf_counter() - t0) * 1000.0)
        p.sample()
    # recall@K (compare last retrieved to baseline per query)
    # For simplicity compute approximate on whole Q at once:
    _, i1 = be.search(q, topk, **params)
    rec = recall_at_k(i1, i0, topk)
    stats = {
        "backend": backend,
        "topk": topk,
        "lat_ms_avg": float(np.mean(lat)),
        "lat_ms_p50": float(np.percentile(lat, 50)),
        "lat_ms_p95": float(np.percentile(lat, 95)),
        "recall@k": rec,
        "ntotal": int(x.shape[0]),
    }
    logger.info(json.dumps(stats, indent=2))


def cmd_update(backend: str, hf_dir: Path, add_n: int, delete: int, cfg: Mapping[str, Any]) -> None:
    """Upsert + delete small batch and re-search."""
    params = cfg["backends"][backend]
    hf_dir = hf_dir or cfg["embedding"]["out_hf_dir"]
    add_n = add_n or int(cfg["update"]["add_count"])
    del_n = delete or int(cfg["update"]["delete_count"])

    ds = load_from_disk(hf_dir)
    x = np.stack(ds["embedding"]).astype("float32")
    ids = np.array(ds["id"], dtype="int64")
    be = _init_backend(backend, x.shape[1], "ip", params)
    be.train(x[: min(500000, len(x))])
    be.upsert(ids, x)

    # craft new vectors by slight noise around existing (simulating fresh writes)
    rng = np.random.default_rng(42)
    add_vecs = x[:add_n].copy()
    add_vecs += rng.normal(0, 0.01, size=add_vecs.shape).astype("float32")
    add_vecs /= np.linalg.norm(add_vecs, axis=1, keepdims=True) + 1e-9
    add_ids = np.arange(10_000_000, 10_000_000 + add_n, dtype="int64")

    with Profiler(f"update-add-{backend}"):
        be.upsert(add_ids, add_vecs)

    with Profiler(f"update-delete-{backend}"):
        del_ids = list(add_ids[:del_n])
        be.delete(del_ids)

    logger.info("Update complete.", be.stats())


def cmd_run_all(size: str, backend: str, cfg: Mapping[str, Any]) -> None:
    """Run end-to-end benchmark with all steps."""

    # minimal run for small dataset
    class A:
        pass

    a = A()
    a.size = size or "small"
    a.out_dir = cfg["dataset"]["out_dir"]
    a.export_images = False
    cmd_download(a, cfg)

    b = A()
    b.model_id = cfg["embedding"]["model_id"]
    b.batch = int(cfg["embedding"]["batch"])
    b.raw_dir = cfg["dataset"]["out_dir"]
    b.emb_dir = cfg["embedding"]["out_dir"]
    cmd_embed(b, cfg)

    # Build FAISS Flat baseline then chosen backend
    for be in ["faiss.flat", backend or "faiss.ivfpq"]:
        c = A()
        c.backend = be
        c.hf_dir = cfg["embedding"]["out_hf_dir"]
        cmd_build(c, cfg)

    d = A()
    d.backend = backend or "faiss.ivfpq"
    d.hf_dir = cfg["embedding"]["out_hf_dir"]
    d.topk = 10
    d.queries = cfg["search"]["queries_file"]
    cmd_search(d, cfg)

    e = A()
    e.backend = backend or "faiss.ivfpq"
    e.hf_dir = cfg["embedding"]["out_hf_dir"]
    e.add = None
    e.delete = None
    cmd_update(e, cfg)


def main() -> None:
    """Entry point."""
    p = argparse.ArgumentParser(description="VectorDB-agnostic benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("download", help="Download HF dataset and optionally export images")
    sp.add_argument("--size", choices=["small", "large", "xlarge", "xxlarge"], default="small")
    sp.add_argument("--out_dir", default=None)
    sp.add_argument("--export-images", action="store_true")
    sp.add_argument("--no-export-images", dest="export_images", action="store_false")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("embed", help="Compute CLIP embeddings and save HF dataset with 'embedding'")
    sp.add_argument("--raw_dir", default=None, type=Path)
    sp.add_argument("--emb_dir", default=None, type=Path)
    sp.add_argument("--model_id", default=None)
    sp.add_argument("--batch", type=int, default=None)
    sp.set_defaults(func=cmd_embed)

    sp = sub.add_parser("build", help="Build index for a backend")
    sp.add_argument("--backend", required=True, choices=list(BACKENDS.keys()))
    sp.add_argument("--hf_dir", default=None)
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("search", help="Profile search on a backend and compute recall@K vs exact baseline")
    sp.add_argument("--backend", required=True, choices=list(BACKENDS.keys()))
    sp.add_argument("--hf_dir", default=None)
    sp.add_argument("--topk", type=int, default=None)
    sp.add_argument("--queries", default=None)
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("update", help="Upsert + delete workflow on a backend")
    sp.add_argument("--backend", required=True, choices=list(BACKENDS.keys()))
    sp.add_argument("--hf_dir", type=Path, default=None)
    sp.add_argument("--add", type=int, default=None)
    sp.add_argument("--delete", type=int, default=None)
    sp.set_defaults(func=cmd_update)

    sp = sub.add_parser(
        "run-all", help="Run small end-to-end: download -> embed -> build -> search -> update"
    )
    sp.add_argument("--size", default="small")
    sp.add_argument("--backend", default="faiss.ivfpq")
    sp.set_defaults(func=cmd_run_all)

    args = p.parse_args()
    cfg = load_cfg(BENCHMARK_CFG)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
