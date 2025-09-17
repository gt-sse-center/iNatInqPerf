# tests/test_dataio.py
import csv
from typing import Any, List, Optional

import numpy as np
import pytest
from PIL import Image

from inatinqperf.utils import dataio
from inatinqperf.utils.dataio import export_images, load_composite


class FakeDataset(list):
    """Minimal stand-in for `datasets.Dataset` supporting iteration and indexing."""

    def __init__(self, data: Optional[list[Any]] = None):
        # Pythonic idiom for list default arg
        if data is None:
            data = []

        super().__init__(data)
        self.concatenated = False


@pytest.fixture(name="fake_loader")
def fake_loader_fixture(monkeypatch):
    """Monkeypatch `load_dataset` to capture split requests."""

    calls: List[Any] = []

    def _load(hf_id: str, split: str):
        calls.append((hf_id, split))
        if split in {"bad", "worse"}:
            raise RuntimeError("boom")
        return FakeDataset([{"split": split}])

    monkeypatch.setattr(dataio, "load_dataset", _load, raising=True)
    return calls


@pytest.fixture(name="fake_concat")
def fake_concat_fixture(monkeypatch):
    """Monkeypatch `concatenate_datasets` to track usage and join iterables."""

    def _concat(parts):
        out = FakeDataset()
        for part in parts:
            out.extend(part)
        out.concatenated = True  # type: ignore[attr-defined]
        return out

    monkeypatch.setattr(dataio, "concatenate_datasets", _concat, raising=True)


def test_load_composite_all_parts_fail_falls_back_to_train(fake_loader, caplog):
    ds = load_composite("hf/any", ("bad", "worse"))

    splits = [s for _, s in fake_loader]
    assert splits == ["bad", "worse", "train"]
    assert [row["split"] for row in ds] == ["train"]
    assert "[DATAIO] Warning: failed to load dataset split 'bad'" in caplog.text


def test_load_composite_single_part_avoids_concat(monkeypatch):
    monkeypatch.setattr(
        dataio,
        "load_dataset",
        lambda hf_id, split: FakeDataset([{"split": split}]),
        raising=True,
    )
    monkeypatch.setattr(
        dataio,
        "concatenate_datasets",
        lambda *_: (_ for _ in ()).throw(AssertionError("should not concatenate")),
        raising=True,
    )

    ds = load_composite("hf/some", ("train[:4]",))
    assert isinstance(ds, FakeDataset)
    assert ds[0]["split"] == "train[:4]"


def test_load_composite_multiple_parts_concatenates(fake_loader, fake_concat):
    ds = load_composite("hf/any", ("train[:2]", "train[:3]"))
    assert isinstance(ds, FakeDataset)
    assert getattr(ds, "concatenated", False) is True
    splits = [s for _, s in fake_loader]
    assert splits[:2] == ["train[:2]", "train[:3]"]


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
