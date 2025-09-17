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


def get_device() -> str:
    """Return the accelerator device which is available."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"

    return "cpu"


@dataclass
class ImageDatasetWithEmbeddings:
    """An image dataset with embeddings, IDs and labels."""

    embeddings: np.ndarray
    ids: Sequence[int] | np.ndarray
    labels: Sequence[int | str] | np.ndarray


def embed_images(raw_dir: Path, model_id: str, batch: int) -> ImageDatasetWithEmbeddings:
    """Embed images from a dataset on disk using a CLIP model."""
    ds = load_from_disk(raw_dir)
    device = get_device()
    proc = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)

    imgs = [pilify(r["image"]) for r in ds]

    all_emb, ids, labels = [], [], []
    for i in tqdm(range(0, len(imgs), batch)):
        batch_imgs = imgs[i : i + batch]
        with torch.no_grad():
            inputs = proc(images=batch_imgs, return_tensors="pt", padding=True).to(device)
            feats = model.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            all_emb.append(feats.cpu().numpy().astype(np.float32))
        ids.extend([i + j for j in range(len(batch_imgs))])
        labels.extend(
            [int(ds[i + j].get("labels", ds[i + j].get("label", 0))) for j in range(len(batch_imgs))]
        )

    if all_emb:
        x = np.concatenate(all_emb, axis=0)
    else:
        config = getattr(model, "config", None)
        dim = 0
        if config is not None:
            if isinstance(config, dict):
                dim = int(config.get("projection_dim") or config.get("hidden_size") or 0)
            else:
                dim = int(getattr(config, "projection_dim", 0) or getattr(config, "hidden_size", 0))
        x = np.empty((0, max(dim, 0)), dtype=np.float32)

    return ImageDatasetWithEmbeddings(x, ids, labels)


def to_hf_dataset(
    embeddings: np.ndarray,
    ids: Sequence[int] | np.ndarray,
    labels: Sequence[int | str] | np.ndarray,
) -> Dataset:
    """Convert embeddings and metadata to a HuggingFace dataset."""
    emb_dim = embeddings.shape[1] if embeddings.ndim == _EMBED_MATRIX_NDIM and embeddings.shape[1:] else 0

    try:
        label_values = [int(y) for y in labels]
        label_feature = Value("int32")
    except (TypeError, ValueError):
        label_values = [str(y) for y in labels]
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
            "id": [int(i) for i in ids],
            "label": label_values,
            "embedding": [d.tolist() for d in embeddings],
        },
        features=feats,
    )


def embed_text(queries: list[str], model_id: str) -> np.ndarray:
    """Embed text queries using a CLIP model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    proc = CLIPProcessor.from_pretrained(model_id)
    with torch.no_grad():
        inputs = proc(text=queries, return_tensors="pt", padding=True).to(device)
        feats = model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
    return feats.cpu().numpy().astype(np.float32)
