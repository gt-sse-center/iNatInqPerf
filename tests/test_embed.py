# tests/test_embed.py
import numpy as np
import pytest
import torch
from PIL import Image

from inatinqperf.utils import embed


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


def test_pilify_invalid_type():
    with pytest.raises(TypeError):
        embed.pilify("not-an-image")


def test_embed_images_empty_dataset(monkeypatch):
    monkeypatch.setattr(embed, "load_from_disk", lambda _: [])
    monkeypatch.setattr(
        embed, "CLIPModel", type("M", (), {"from_pretrained": lambda *args, **kwargs: DummyModel()})
    )
    monkeypatch.setattr(
        embed, "CLIPProcessor", type("P", (), {"from_pretrained": lambda *args, **kwargs: DummyProcessor()})
    )

    dataset_with_embeddings = embed.embed_images("anypath", "dummy-model", batch=2)
    X = dataset_with_embeddings.embeddings
    ids = dataset_with_embeddings.ids
    labels = dataset_with_embeddings.labels

    assert X.shape[0] == 0
    assert ids == []
    assert labels.tolist() == []


def test_embed_images_model_error(monkeypatch):
    class BrokenModel(DummyModel):
        def get_image_features(self, **inputs):
            raise RuntimeError("boom")

    monkeypatch.setattr(embed, "load_from_disk", lambda _: [{"image": Image.new("RGB", (8, 8)), "label": 0}])
    monkeypatch.setattr(
        embed, "CLIPModel", type("M", (), {"from_pretrained": lambda *args, **kwargs: BrokenModel()})
    )
    monkeypatch.setattr(
        embed, "CLIPProcessor", type("P", (), {"from_pretrained": lambda *args, **kwargs: DummyProcessor()})
    )

    with pytest.raises(RuntimeError):
        embed.embed_images("anypath", "dummy-model", batch=1)


def test_to_hf_dataset_structure():
    X = np.ones((3, 4))
    ids = [1, 2, 3]
    labels = np.asarray(["a", "b", "c"])
    dse = embed.ImageDatasetWithEmbeddings(X, ids, labels)

    ds = embed.to_huggingface_dataset(dse)
    assert set(ds.column_names) == {"id", "label", "embedding"}
    assert isinstance(ds[0]["embedding"], list)
    assert ds[0]["id"] == 1
    assert ds[0]["label"] == "a"


def test_embed_images_and_to_hf_dataset(monkeypatch, tmp_path):
    """Test embedding images and saving to HuggingFace dataset format."""

    # Fake dataset: two records with images + labels
    class FakeDataset(list):
        def __getitem__(self, i):
            return super().__getitem__(i)

    ds = FakeDataset([{"image": Image.new("RGB", (8, 8), color=(i, i, i)), "label": i} for i in range(4)])
    monkeypatch.setattr(embed, "load_from_disk", lambda path: ds)
    monkeypatch.setattr(
        embed, "CLIPModel", type("M", (), {"from_pretrained": lambda *args, **kwargs: DummyModel()})
    )
    monkeypatch.setattr(
        embed, "CLIPProcessor", type("P", (), {"from_pretrained": lambda *args, **kwargs: DummyProcessor()})
    )

    dataset_with_embeddings = embed.embed_images("anypath", "dummy-model", batch=2)
    X = dataset_with_embeddings.embeddings
    ids = dataset_with_embeddings.ids
    labels = dataset_with_embeddings.labels

    assert X.shape[0] == 4
    assert all(isinstance(i, int) for i in ids)
    assert labels.tolist() == [0, 1, 2, 3]

    # Convert to HF dataset structure
    hf_ds = embed.to_huggingface_dataset(dataset_with_embeddings)
    assert set(hf_ds.column_names) == {"id", "label", "embedding"}
    assert len(hf_ds) == 4
    assert isinstance(hf_ds[0]["embedding"], list)


def test_embed_text(monkeypatch):
    monkeypatch.setattr(
        embed, "CLIPModel", type("M", (), {"from_pretrained": lambda *args, **kwargs: DummyModel()})
    )
    monkeypatch.setattr(
        embed, "CLIPProcessor", type("P", (), {"from_pretrained": lambda *args, **kwargs: DummyProcessor()})
    )

    X = embed.embed_text(["hello", "world"], "dummy-model")
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 2


def test_embed_text_empty(monkeypatch):
    monkeypatch.setattr(
        embed, "CLIPModel", type("M", (), {"from_pretrained": lambda *args, **kwargs: DummyModel()})
    )
    monkeypatch.setattr(
        embed, "CLIPProcessor", type("P", (), {"from_pretrained": lambda *args, **kwargs: DummyProcessor()})
    )

    X = embed.embed_text([], "dummy-model")
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 0
