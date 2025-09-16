# tests/test_dataio.py
import csv
import io
from typing import Any, List

import numpy as np
import pytest
from PIL import Image

from inatinqperf.utils.dataio import load_composite, export_images


# ----------------------------
# Helpers / fakes for datasets
# ----------------------------
class FakeDataset(list):
    """Minimal stand-in for `datasets.Dataset` that supports iteration and indexing."""

    # In these tests we only iterate; no extra API needed.
    pass


@pytest.fixture
def fake_load_dataset(monkeypatch):
    """Monkeypatch `datasets.load_dataset` with controllable behavior."""
    calls: List[Any] = []

    def _load(hf_id: str, split: str):
        calls.append((hf_id, split))
        # Simulate failures for a specific split
        if split == "bad":
            raise RuntimeError("split not available")
        # Return a small fake dataset tagged by split for identification
        return FakeDataset([{"split": split}])

    import inatinqperf.utils.dataio as dataio

    monkeypatch.setattr(dataio, "load_dataset", _load, raising=True)
    return calls


@pytest.fixture
def fake_concatenate(monkeypatch):
    """Monkeypatch `datasets.concatenate_datasets` to simply join lists."""

    def _concat(parts):
        # confirm it's getting a list-like
        out = FakeDataset()
        for p in parts:
            out.extend(p)
        # Attach a flag so we can assert this path was taken
        out.concatenated = True  # type: ignore[attr-defined]
        return out

    import inatinqperf.utils.dataio as dataio

    monkeypatch.setattr(dataio, "concatenate_datasets", _concat, raising=True)


# ----------------------------
# Tests for load_composite(...)
# ----------------------------
def test_load_composite_concatenates_on_plus_expression(fake_load_dataset, fake_concatenate, monkeypatch):
    import inatinqperf.utils.dataio as dataio

    monkeypatch.setattr(
        dataio, "load_dataset", lambda hf_id, split: FakeDataset([{"split": split}]), raising=True
    )

    # Expression with spaces; one missing/empty chunk should be ignored
    ds = load_composite("hf/some-id", "train[:5] + validation[:3] + ")

    # We expect a concatenated FakeDataset with 2 items (one per split)
    assert isinstance(ds, FakeDataset)
    assert hasattr(ds, "concatenated")
    # Order is not guaranteed; compare as a set
    assert set(row["split"] for row in ds) == {"train[:5]", "validation[:3]"}


def test_load_composite_single_part_returns_dataset(fake_load_dataset, monkeypatch):
    import inatinqperf.utils.dataio as dataio

    monkeypatch.setattr(
        dataio, "load_dataset", lambda hf_id, split: FakeDataset([{"split": split}]), raising=True
    )
    monkeypatch.setattr(
        dataio, "concatenate_datasets", lambda _: (_ for _ in ()).throw(AssertionError), raising=True
    )

    ds = load_composite("hf/some-id", "train[:2]")
    assert isinstance(ds, FakeDataset)
    assert len(ds) == 1 and ds[0]["split"] == "train[:2]"


def test_load_composite_fallback_to_train_when_all_parts_fail(monkeypatch):
    calls = []

    def _load(hf_id: str, split: str):
        calls.append((hf_id, split))
        if split != "train":
            raise RuntimeError("boom")
        return FakeDataset([{"split": split}])

    import inatinqperf.utils.dataio as dataio

    monkeypatch.setattr(dataio, "load_dataset", _load, raising=True)

    ds = load_composite("hf/any", "bad + also_bad")
    assert isinstance(ds, FakeDataset)
    assert [row["split"] for row in ds] == ["train"]
    # Ensure it attempted the requested splits before falling back
    splits = [split for _, split in calls]
    assert "bad" in splits and "also_bad" in splits and "train" in splits


# ----------------------------
# Tests for export_images(...)
# ----------------------------
def test_export_images_writes_jpegs_and_manifest(tmp_path):
    # Build a tiny dataset with:
    #  - one PIL.Image
    #  - one NumPy array (HxWxC)
    #  - labels in different forms: int and string
    pil_img = Image.new("RGB", (8, 8), color=(10, 20, 30))
    np_img = np.ones((8, 8, 3), dtype=np.uint8) * 127  # gray image

    ds = FakeDataset(
        [
            {"image": pil_img, "label": 7},
            {"image": np_img, "labels": "butterfly"},
        ]
    )

    export_dir = tmp_path / "images_out"
    manifest_path = export_images(ds, str(export_dir))

    # Read manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 3  # header + 2 items

    # Header may be exactly these columns; if not, map by position assuming 3 cols
    header = rows[0]
    assert len(header) == 3
    # Normalize header names to lower for flexible matching
    header_lower = [h.strip().lower() for h in header]
    assert set(header_lower) == {"index", "filename", "label"}

    # Build helper to get column index by name independent of order
    col_idx = {name: header_lower.index(name) for name in ("index", "filename", "label")}
    row1, row2 = rows[1], rows[2]

    # Labels preserved: first is int-like, second is string-like
    assert row1[col_idx["label"]] in ("7", 7)
    assert str(row2[col_idx["label"]]).lower() == "butterfly"

    # Files exist and are valid JPEGs
    f1 = export_dir / row1[col_idx["filename"]]
    f2 = export_dir / row2[col_idx["filename"]]
    assert f1.exists() and f2.exists()
    with Image.open(f1) as im0:
        assert im0.size == (8, 8)
    with Image.open(f2) as im1:
        # Mode may be normalized to RGB by exporter; just ensure it opens
        assert im1.size == (8, 8)
