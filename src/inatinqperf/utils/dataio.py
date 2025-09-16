"""Utilities for data I/O operations."""

import csv
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset
from PIL import Image
from pathlib import Path


def load_composite(hf_id: str, expr: str) -> Dataset:
    """Load a composite HuggingFace dataset from multiple splits."""
    parts = [p.strip() for p in expr.split("+") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(load_dataset(hf_id, split=p))
        except Exception:
            print(f"[DATAIO] Warning: failed to load dataset split '{p}' from '{hf_id}'")
    if not out:
        return load_dataset(hf_id, split="train")
    if len(out) == 1:
        return out[0]
    return concatenate_datasets(out)


def export_images(ds: Dataset, export_dir: str) -> str:
    """Export images from a HuggingFace dataset to a directory with a manifest CSV."""
    Path(export_dir).mkdir(parents=True, exist_ok=True)
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
