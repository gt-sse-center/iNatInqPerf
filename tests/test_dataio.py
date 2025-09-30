# tests/test_dataio.py
import csv
from typing import Any, List, Optional

import numpy as np
import pytest
from PIL import Image

from inatinqperf.utils import dataio
from inatinqperf.utils.dataio import export_images, load_huggingface_dataset


class FakeDataset(list):
    """Minimal stand-in for `datasets.Dataset` supporting iteration and indexing."""

    def __init__(self, data: Optional[list[Any]] = None):
        # Pythonic idiom for list default arg
        if data is None:
            data = []

        super().__init__(data)


def test_load_huggingface_dataset_with_failure():
    with pytest.raises(Exception):
        load_huggingface_dataset("hf/any", "bad+worse")


def test_load_huggingface_dataset(monkeypatch):
    monkeypatch.setattr(
        dataio,
        "load_dataset",
        lambda dataset_id, split: FakeDataset([{"split": split}]),
        raising=True,
    )

    ds = load_huggingface_dataset("hf/some", "train[:4]")
    assert isinstance(ds, FakeDataset)
    assert ds[0]["split"] == "train[:4]"


def test_export_images_writes_jpegs_and_manifest(tmp_path):
    pil_img = Image.new("RGB", (8, 8), color=(10, 20, 30))
    np_img = np.ones((8, 8, 3), dtype=np.uint8) * 127

    ds = FakeDataset(
        [
            {"image": pil_img, "label": 7},
            {"image": np_img, "labels": "butterfly"},
        ]
    )

    export_dir = tmp_path / "images_out"
    manifest_path = export_images(ds, export_dir)

    with open(manifest_path, "r", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    header = [col.lower() for col in rows[0]]
    col_idx = {name: header.index(name) for name in ("index", "filename", "label")}

    row1, row2 = rows[1], rows[2]
    assert row1[col_idx["label"]] in ("7", 7)
    assert row2[col_idx["label"]].lower() == "butterfly"

    fpaths = [export_dir / row[col_idx["filename"]] for row in (row1, row2)]
    for fp in fpaths:
        assert fp.exists()
        with Image.open(fp) as img:
            assert img.size == (8, 8)
