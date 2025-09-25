"""Backend-agnostic benchmark orchestrator.

Subcommands:
  download  -> HF dataset + optional image export
  embed     -> CLIP embeddings, saves HF dataset with 'embedding'
  build     -> build index on chosen backend
  search    -> profile search + recall@K vs Flat
  update    -> upsert/delete small batch and re-search
  run-all   -> download->embed->build(Faiss Flat + chosen backend)->search->update
"""

import json
import time
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Annotated

import numpy as np
import yaml
from datasets import load_from_disk
from loguru import logger
import typer

from inatinqperf.adaptors import BACKENDS
from inatinqperf.utils.dataio import export_images, load_composite
from inatinqperf.utils.embed import embed_images, embed_text, to_hf_dataset
from inatinqperf.utils.profiler import Profiler

# Get the `inatinqperf` directory which is the grandparent directory of this file.
ROOT = Path(__file__).resolve().parents[1]

SAMPLE_SIZE = 500_000  # max samples for training if needed
BENCHMARK_CFG = ROOT / "configs" / "benchmark.yaml"


class DatasetSize(str, Enum):
    """CLI-safe enumeration of dataset splits."""

    small = "small"
    large = "large"
    xlarge = "xlarge"
    xxlarge = "xxlarge"


app = typer.Typer(help="VectorDB-agnostic benchmark orchestrator.")


def load_cfg(path: Path) -> Mapping[str, object]:
    """Load YAML config file."""
    logger.info(f"Loading config: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def _require_faiss() -> object:
    """Import faiss lazily so lightweight commands avoid the dependency."""

    try:
        import faiss  # noqa: PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover - surface to CLI
        msg = "Faiss is required for this command. Install `faiss-gpu` or `faiss-cpu`."
        raise typer.BadParameter(msg) from exc
    return faiss


def ensure_dir(p: Path) -> Path:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_hf_dir(hf_dir: Path | None, cfg: Mapping[str, object]) -> Path:
    """Resolve a HuggingFace dataset directory and validate its contents."""

    candidate = Path(hf_dir) if hf_dir else Path(cfg["embedding"]["out_hf_dir"])
    dataset_info_json = candidate / "dataset_info.json"
    dataset_dict_json = candidate / "dataset_dict.json"
    if not candidate.exists() or not (dataset_info_json.exists() or dataset_dict_json.exists()):
        msg = (
            f"Embedded dataset not found at '{candidate}'. Run the embed command first or "
            "point --hf-dir at a directory produced by HuggingFace Dataset.save_to_disk()."
        )
        raise typer.BadParameter(msg)
    return candidate


def _resolve_queries_path(queries: Path | None, cfg: Mapping[str, object]) -> Path:
    """Resolve a queries text file, checking repo-relative fallbacks."""

    candidate = Path(queries) if queries else Path(cfg["search"]["queries_file"])
    if candidate.is_absolute() and candidate.exists():
        return candidate

    for base in (Path.cwd(), ROOT):
        probe = base / candidate
        if probe.exists():
            return probe

    msg = (
        f"Queries file '{candidate}' not found. Provide a valid path via --queries or update "
        "search.queries_file in the config."
    )
    raise typer.BadParameter(msg)


def exact_baseline(x: np.ndarray, metric: str) -> object:
    """Exact baseline index using FAISS IndexFlat*."""
    faiss = _require_faiss()
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


def cmd_download(
    size: str,
    out_dir: Path | None,
    *,
    export_images_flag: bool | None,
    cfg: Mapping[str, object],
) -> None:
    """Download HF dataset and optionally export images."""
    hf_id = cfg["dataset"]["hf_id"]
    out_dir = out_dir if out_dir else Path(cfg["dataset"]["out_dir"])
    export = cfg["dataset"].get("export_images", True) if export_images_flag is None else export_images_flag
    split_map = cfg["dataset"]["size_splits"]
    split = split_map.get(size, split_map["small"])
    ensure_dir(out_dir)
    with Profiler(f"download-{hf_id}-{size}"):
        ds = load_composite(hf_id, split)
        ds.save_to_disk(str(out_dir))
        if export:
            export_dir = out_dir / "images"
            manifest = export_images(ds, export_dir)
            logger.info(f"Exported images to: {export_dir}\nManifest: {manifest}")
    logger.info(f"Saved HuggingFace dataset to: {out_dir}")


def cmd_embed(
    model_id: str | None,
    batch_size: int | None,
    raw_dir: Path | None,
    emb_dir: Path | None,
    cfg: Mapping[str, object],
) -> None:
    """Compute CLIP embeddings and save HF dataset with 'embedding'."""
    model_id = model_id or cfg["embedding"]["model_id"]
    batch_size = batch_size or int(cfg["embedding"]["batch"])
    raw_dir = raw_dir if raw_dir else Path(cfg["dataset"]["out_dir"])
    emb_dir = emb_dir if emb_dir else Path(cfg["embedding"]["out_dir"])
    out_hf_dir = Path(cfg["embedding"]["out_hf_dir"])
    ensure_dir(emb_dir)
    ensure_dir(out_hf_dir)
    with Profiler("embed-images"):
        dataset_with_embeddings = embed_images(raw_dir, model_id, batch_size)
        embeddings = dataset_with_embeddings.embeddings
        ids = dataset_with_embeddings.ids
        labels = dataset_with_embeddings.labels
        ds2 = to_hf_dataset(embeddings, ids, labels)
        ds2.save_to_disk(str(out_hf_dir))
        logger.info(f"Embeddings: {embeddings.shape} -> {out_hf_dir}")


def _init_backend(backend_name: str, dim: int, metric: str, params: dict[str, object]) -> dict:
    """Instantiate backend, scrubbing reserved keys from params.

    Prevents errors like: TypeError: ... init() got multiple values for keyword 'metric'.
    """
    _require_faiss()  # ensure faiss is present before touching backend factory
    backend = BACKENDS[backend_name]

    # Avoid passing duplicate values for explicit kwargs
    safe_params = dict(params) if params else {}
    for k in ("metric", "dim", "name"):
        safe_params.pop(k, None)
    return backend(dim=dim, metric=metric, **safe_params)


def cmd_build(backend: str, hf_dir: Path | None, cfg: Mapping[str, object]) -> None:
    """Build index for a backend."""
    if backend not in BACKENDS:
        msg = f"Unknown backend '{backend}'. Available: {', '.join(sorted(BACKENDS))}"
        raise typer.BadParameter(msg)
    params = cfg["backends"][backend]
    resolved_hf = _resolve_hf_dir(hf_dir, cfg)
    ds = load_from_disk(str(resolved_hf))
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


def cmd_search(
    backend: str,
    hf_dir: Path | None,
    topk: int | None,
    queries: Path | None,
    cfg: Mapping[str, object],
) -> None:
    """Profile search and compute recall@K vs exact baseline."""
    if backend not in BACKENDS:
        msg = f"Unknown backend '{backend}'. Available: {', '.join(sorted(BACKENDS))}"
        raise typer.BadParameter(msg)
    params = cfg["backends"][backend]
    resolved_hf = _resolve_hf_dir(hf_dir, cfg)
    topk = topk or int(cfg["search"]["topk"])
    queries_file = _resolve_queries_path(queries, cfg)
    model_id = cfg["embedding"]["model_id"]

    ds = load_from_disk(str(resolved_hf))
    x = np.stack(ds["embedding"]).astype("float32")
    ids = np.array(ds["id"], dtype="int64")
    metric = params.get("metric", "ip").lower()
    # exact baseline
    base = exact_baseline(x, metric="ip" if metric in ("ip", "cosine") else "l2")
    # backend
    be = _init_backend(
        backend,
        x.shape[1],
        "ip" if metric in ("ip", "cosine") else "l2",
        params,
    )
    be.train(
        x
        if x.shape[0] < SAMPLE_SIZE
        else x[np.random.default_rng().choice(x.shape[0], SAMPLE_SIZE, replace=False)]
    )
    be.upsert(ids, x)

    queries = [q.strip() for q in queries_file.read_text(encoding="utf-8").splitlines() if q.strip()]
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


def cmd_update(
    backend: str,
    hf_dir: Path | None,
    add_n: int | None,
    delete: int | None,
    cfg: Mapping[str, object],
) -> None:
    """Upsert + delete small batch and re-search."""
    if backend not in BACKENDS:
        msg = f"Unknown backend '{backend}'. Available: {', '.join(sorted(BACKENDS))}"
        raise typer.BadParameter(msg)
    params = cfg["backends"][backend]
    resolved_hf = _resolve_hf_dir(hf_dir, cfg)
    add_n = add_n or int(cfg["update"]["add_count"])
    del_n = delete or int(cfg["update"]["delete_count"])

    ds = load_from_disk(str(resolved_hf))
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


def cmd_run_all(size: str, backend: str, cfg: Mapping[str, object]) -> None:
    """Run end-to-end benchmark with all steps."""

    backend = backend or "faiss.ivfpq"
    if backend not in BACKENDS:
        msg = f"Unknown backend '{backend}'. Available: {', '.join(sorted(BACKENDS))}"
        raise typer.BadParameter(msg)

    cmd_download(
        size,
        Path(cfg["dataset"]["out_dir"]),
        export_images_flag=False,
        cfg=cfg,
    )

    cmd_embed(
        cfg["embedding"]["model_id"],
        int(cfg["embedding"]["batch"]),
        Path(cfg["dataset"]["out_dir"]),
        Path(cfg["embedding"]["out_dir"]),
        cfg,
    )

    # Build FAISS Flat baseline then chosen backend
    for be_name in ["faiss.flat", backend]:
        cmd_build(be_name, Path(cfg["embedding"]["out_hf_dir"]), cfg)

    cmd_search(
        backend,
        Path(cfg["embedding"]["out_hf_dir"]),
        10,
        Path(cfg["search"]["queries_file"]),
        cfg,
    )

    cmd_update(
        backend,
        Path(cfg["embedding"]["out_hf_dir"]),
        None,
        None,
        cfg,
    )


@app.command()
def download(
    size: Annotated[DatasetSize, typer.Option("--size", help="Dataset split to fetch.")] = DatasetSize.small,
    out_dir: Annotated[Path | None, typer.Option("--out-dir", help="Directory to store the dataset.")] = None,
    export_images: Annotated[
        bool | None,
        typer.Option(
            "--export-images/--no-export-images",
            help="Explicitly enable or disable image export.",
        ),
    ] = None,
) -> None:
    """Download dataset artifacts."""

    cfg = load_cfg(BENCHMARK_CFG)
    cmd_download(size.value, out_dir, export_images_flag=export_images, cfg=cfg)


@app.command()
def embed(
    raw_dir: Annotated[
        Path | None, typer.Option("--raw-dir", help="Input directory from download step.")
    ] = None,
    emb_dir: Annotated[Path | None, typer.Option("--emb-dir", help="Directory to store embeddings.")] = None,
    model_id: Annotated[str | None, typer.Option("--model-id", help="CLIP model identifier.")] = None,
    batch: Annotated[int | None, typer.Option("--batch", help="Batch size for embedding inference.")] = None,
) -> None:
    """Compute CLIP embeddings for the dataset."""

    cfg = load_cfg(BENCHMARK_CFG)
    cmd_embed(model_id, batch, raw_dir, emb_dir, cfg)


@app.command()
def build(
    backend: Annotated[str, typer.Option("--backend", help="Backend name to build.")] = ...,
    hf_dir: Annotated[Path | None, typer.Option("--hf-dir", help="Directory with embedded dataset.")] = None,
) -> None:
    """Build an index for a backend."""

    cfg = load_cfg(BENCHMARK_CFG)
    cmd_build(backend, hf_dir, cfg)


@app.command()
def search(
    backend: Annotated[str, typer.Option("--backend", help="Backend name to search.")] = ...,
    hf_dir: Annotated[Path | None, typer.Option("--hf-dir", help="Directory with embedded dataset.")] = None,
    topk: Annotated[
        int | None, typer.Option("--topk", help="Number of nearest neighbors to retrieve.")
    ] = None,
    queries: Annotated[
        Path | None,
        typer.Option("--queries", help="Text file with newline-delimited queries."),
    ] = None,
) -> None:
    """Profile search latency and recall for a backend."""

    cfg = load_cfg(BENCHMARK_CFG)
    cmd_search(backend, hf_dir, topk, queries, cfg)


@app.command()
def update(
    backend: Annotated[str, typer.Option("--backend", help="Backend name to exercise update path.")] = ...,
    hf_dir: Annotated[Path | None, typer.Option("--hf-dir", help="Directory with embedded dataset.")] = None,
    add: Annotated[int | None, typer.Option("--add", help="How many vectors to upsert.")] = None,
    delete: Annotated[int | None, typer.Option("--delete", help="How many vectors to delete.")] = None,
) -> None:
    """Exercise the upsert/delete workflow for a backend."""

    cfg = load_cfg(BENCHMARK_CFG)
    cmd_update(backend, hf_dir, add, delete, cfg)


@app.command("run-all")
def run_all(
    size: Annotated[DatasetSize, typer.Option("--size", help="Dataset split to use.")] = DatasetSize.small,
    backend: Annotated[str, typer.Option("--backend", help="Backend name for the workflow.")] = "faiss.ivfpq",
) -> None:
    """Run the full benchmark workflow."""

    cfg = load_cfg(BENCHMARK_CFG)
    cmd_run_all(size.value, backend, cfg)


def main() -> None:
    """Typer entry point."""

    app()


if __name__ == "__main__":
    main()
