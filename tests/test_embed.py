"""Tests for the `embed` module."""

import numpy as np
import pytest
from PIL import Image

from inatinqperf.utils import embed


@pytest.fixture(autouse=True)
def clip_model_fixture(mocker):
    class DummyModel:
        """Dummy embedding model to use as a mock."""

        def __init__(self, model_id: str):
            self.model_id = model_id
            self.embedding_dim: int = 512

        def __call__(
            self,
            images: list | None = None,
            text: list | None = None,
        ) -> np.ndarray:
            if images:
                return np.zeros((len(images), self.embedding_dim))
            if text:
                return np.zeros((len(text), self.embedding_dim))

    mocker.patch("inatinqperf.utils.embed.PretrainedCLIPModel", DummyModel)


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

    dataset_with_embeddings = embed.embed_images("anypath", "dummy-model", batch_size=2)
    X = dataset_with_embeddings.embeddings
    ids = dataset_with_embeddings.ids
    labels = dataset_with_embeddings.labels

    assert X.shape[0] == 0
    assert ids == []
    assert labels.tolist() == []


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

    class FakeDataset(list):
        """Fake dataset: two records with images + labels."""

        def __getitem__(self, i):
            return super().__getitem__(i)

    N = 4
    batch_size = 2

    ds = FakeDataset([{"image": Image.new("RGB", (8, 8), color=(i, i, i)), "label": i} for i in range(N)])
    monkeypatch.setattr(embed, "load_from_disk", lambda path: ds)

    dataset_with_embeddings = embed.embed_images("anypath", "dummy-model", batch_size=batch_size)
    X = dataset_with_embeddings.embeddings
    ids = dataset_with_embeddings.ids
    labels = dataset_with_embeddings.labels

    assert X.shape[0] == N
    assert all(isinstance(i, int) for i in ids)
    assert labels.tolist() == [0, 1, 2, 3]

    # Convert to HF dataset structure
    hf_ds = embed.to_huggingface_dataset(dataset_with_embeddings)
    assert set(hf_ds.column_names) == {"id", "label", "embedding"}
    assert len(hf_ds) == 4
    assert isinstance(hf_ds[0]["embedding"], list)


def test_embed_text():
    X = embed.embed_text(["hello", "world"], "dummy-model")
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 2


def test_embed_text_empty():
    X = embed.embed_text([], "dummy-model")
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 0
