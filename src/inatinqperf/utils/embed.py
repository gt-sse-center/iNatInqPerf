"""Utilities for embedding images and text using CLIP models."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, Features, Value, load_from_disk
from datasets import Sequence as HFSequence
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

_EMBED_MATRIX_NDIM = 2


def pilify(img: Image.Image | np.ndarray) -> Image.Image:
    """Convert inputs to a PIL RGB image when possible."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    msg = "Expected PIL.Image or numpy.ndarray"
    raise TypeError(msg)


@dataclass
class ImageDatasetWithEmbeddings:
    """An image dataset with embeddings, IDs and labels."""

    embeddings: np.ndarray
    ids: Sequence[int] | np.ndarray
    labels: Sequence[int | str] | np.ndarray


class PretrainedCLIPModel:
    """Helper class for loading and running a pretrained CLIP model."""

    def __init__(self, model_id: str) -> None:
        self.device = self.get_device()
        self.proc = CLIPProcessor.from_pretrained(model_id, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        # Get the embedding dimension from the model config
        self.embedding_dim: int = self.model.config.projection_dim

    @staticmethod
    def get_device() -> str:
        """Return the accelerator device which is available."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.mps.is_available():
            return "mps"

        return "cpu"

    def __call__(
        self,
        images: list[Image.Image] | None = None,
        text: list[str] | None = None,
    ) -> np.ndarray:
        """Forward pass of either image or text data."""
        if images is not None:
            inputs = self.proc(images=images, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_image_features(**inputs)

        elif text is not None:
            inputs = self.proc(text=text, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_text_features(**inputs)

        else:
            msg = "Neither image nor text data provided."
            raise ValueError(msg)

        normalized_feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return normalized_feats.cpu().numpy().astype(np.float32)


def embed_images(raw_dir: Path, model_id: str, batch_size: int) -> ImageDatasetWithEmbeddings:
    """Embed images from a dataset on disk using a CLIP model."""
    ds = load_from_disk(raw_dir)

    model = PretrainedCLIPModel(model_id=model_id)

    imgs = [pilify(r["image"]) for r in ds]

    embeddings, ids, labels = [], [], []
    for i in tqdm(range(0, len(imgs), batch_size)):
        batch_imgs = imgs[i : i + batch_size]
        with torch.no_grad():
            feats = model(images=batch_imgs)
            embeddings.append(feats)

        ids.extend([i + j for j in range(len(batch_imgs))])
        labels.extend(
            [int(ds[i + j].get("label", ds[i + j].get("label", 0))) for j in range(len(batch_imgs))]
        )

    if embeddings:
        x = np.concatenate(embeddings, axis=0)

    else:
        config = getattr(model, "config", None)
        dim = 0
        if config is not None:
            if isinstance(config, dict):
                dim = int(config.get("projection_dim") or config.get("hidden_size") or 0)
            else:
                dim = int(getattr(config, "projection_dim", 0) or getattr(config, "hidden_size", 0))
        x = np.empty((0, max(dim, 0)), dtype=np.float32)

    return ImageDatasetWithEmbeddings(x, ids, np.asarray(labels))


def embed_text(queries: list[str], model_id: str, batch_size: int = 128) -> np.ndarray:
    """Embed text queries using a CLIP model."""
    model = PretrainedCLIPModel(model_id=model_id)

    feats = np.empty((len(queries), model.embedding_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i : i + batch_size]
            batch_feats = model(text=batch_queries)
            feats[i : i + batch_feats.shape[0]] = batch_feats

    return feats


def to_huggingface_dataset(
    dataset: ImageDatasetWithEmbeddings,
) -> Dataset:
    """Convert embeddings and metadata to a HuggingFace dataset."""
    emb_dim = (
        dataset.embeddings.shape[1]
        if dataset.embeddings.ndim == _EMBED_MATRIX_NDIM and dataset.embeddings.shape[1:]
        else 0
    )

    # Convert to np array
    dataset.labels = np.asarray(dataset.labels)

    try:
        label_values = dataset.labels.astype(np.int32)
        label_feature = Value("int32")
    except (TypeError, ValueError):
        label_values = dataset.labels.astype(str)
        label_feature = Value("string")

    feats = Features(
        {
            "id": Value("int64"),
            "label": label_feature,
            "embedding": HFSequence(Value("float32"), length=emb_dim if emb_dim else -1),
        },
    )
    return Dataset.from_dict(
        {
            "id": dataset.ids,
            "label": label_values,
            "embedding": dataset.embeddings,
        },
        features=feats,
    )
