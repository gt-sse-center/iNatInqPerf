"""Utilities for data I/O operations."""

import csv
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from loguru import logger
from PIL import Image


def load_huggingface_dataset(dataset_id: str, splits: str) -> Dataset:
    """Load a HuggingFace dataset with ID `dataset_id` with multiple `splits` using the HuggingFace API."""
    try:
        return load_dataset(dataset_id, split=splits)
    except Exception:
        logger.error("Could not load dataset. Perhaps check the splits specified?")
        raise


def export_images(ds: Dataset, export_dir: Path) -> Path:
    """Export images from a HuggingFace dataset to a directory with a manifest CSV."""
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(export_dir) / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as mf:
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
