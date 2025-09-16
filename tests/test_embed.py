# tests/test_embed.py
import numpy as np
import pytest
from PIL import Image
import torch

import inatinqperf.utils.embed as embed


# -----------------------
# Fake classes for mocking
# -----------------------
class DummyModel:
    def __init__(self):
        self.called = {"image": 0, "text": 0}

    def to(self, device):
        return self

    def get_image_features(self, **inputs):
        self.called["image"] += 1
        bsz = inputs["pixel_values"].shape[0]
        return torch.ones((bsz, 4))  # shape (batch, dim)

    def get_text_features(self, **inputs):
        self.called["text"] += 1
        bsz = len(inputs["input_ids"])
        return torch.ones((bsz, 4))

    @property
    def config(self):
        return {}


class DummyProcessor:
    def __init__(self):
        self.called = {"image": 0, "text": 0}

    def to(self, device):
        return self

    def __call__(self, **kwargs):
        class DummyInputs(dict):
            def to(self, device):
                return self

        if "images" in kwargs:
            self.called["image"] += 1
            arr = np.zeros((len(kwargs["images"]), 3, 8, 8), dtype=np.float32)
            return DummyInputs({"pixel_values": arr})
        if "text" in kwargs:
            self.called["text"] += 1
            return DummyInputs({"input_ids": [1] * len(kwargs["text"])})
        return DummyInputs()


# -----------------------
# Tests
# -----------------------
def test_pilify_numpy_and_pil():
    arr = np.ones((5, 5, 3), dtype=np.uint8)
    out1 = embed.pilify(arr)
    assert isinstance(out1, Image.Image)
    img = Image.new("RGB", (5, 5), color=(10, 20, 30))
    out2 = embed.pilify(img)
    assert out2.mode == "RGB"


def test_embed_images_and_to_hf_dataset(monkeypatch, tmp_path):
    # Fake dataset: two records with images + labels
    class FakeDataset(list):
        def __getitem__(self, i):
            return super().__getitem__(i)

    ds = FakeDataset([{"image": Image.new("RGB", (8, 8), color=(i, i, i)), "label": i} for i in range(4)])
    monkeypatch.setattr(embed, "load_from_disk", lambda path: ds)
    monkeypatch.setattr(embed, "CLIPModel", type("M", (), {"from_pretrained": lambda _: DummyModel()}))
    monkeypatch.setattr(
        embed, "CLIPProcessor", type("P", (), {"from_pretrained": lambda _: DummyProcessor()})
    )

    ds_out, X, ids, labels = embed.embed_images("anypath", "dummy-model", batch=2)
    assert ds_out is ds
    assert X.shape[0] == 4
    assert all(isinstance(i, int) for i in ids)
    assert labels == [0, 1, 2, 3]

    # Convert to HF dataset structure
    hf_ds = embed.to_hf_dataset(X, ids, labels)
    assert set(hf_ds.column_names) == {"id", "label", "embedding"}
    assert len(hf_ds) == 4
    assert isinstance(hf_ds[0]["embedding"], list)


def test_embed_text(monkeypatch):
    monkeypatch.setattr(embed, "CLIPModel", type("M", (), {"from_pretrained": lambda _: DummyModel()}))
    monkeypatch.setattr(
        embed, "CLIPProcessor", type("P", (), {"from_pretrained": lambda _: DummyProcessor()})
    )

    X = embed.embed_text(["hello", "world"], "dummy-model")
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 2
