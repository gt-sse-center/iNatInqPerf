"""Utilities for data I/O operations."""

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from loguru import logger
from PIL import Image


def load_composite(hf_id: str, splits: Sequence[str]) -> Dataset:
    """Load a composite HuggingFace dataset with ID `hf_id` from multiple splits."""
    # TODO(Varun): Use the HuggingFace API to download the data more efficiently.
    splits = [p.strip() for p in splits if isinstance(p, str)]
    out = []
    for split in splits:
        try:
            out.append(load_dataset(hf_id, split=split))
        except Exception:  # noqa: PERF203
            logger.warning(f"[DATAIO] Warning: failed to load dataset split '{split}' from '{hf_id}'")
    if not out:
        return load_dataset(hf_id, split="train")
    if len(out) == 1:
        return out[0]

    return concatenate_datasets(out)


def export_images(ds: Dataset, export_dir: Path) -> Path:
    """Export images from a HuggingFace dataset to a directory with a manifest CSV."""
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(export_dir) / "manifest.csv"
    with Path(manifest).open("w", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        w.writerow(["index", "filename", "label"])
        for i, row in enumerate(ds):
            img = row["image"]
            pil = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img)).convert("RGB")
            fname = f"{i:08d}.jpg"
            fp = Path(export_dir) / fname
            pil.save(fp, format="JPEG", quality=90)
            label = row.get("labels", row.get("label", ""))
            w.writerow([i, fname, int(label) if isinstance(label, (int, np.integer)) else label])
    return manifest
