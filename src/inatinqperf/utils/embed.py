"""Utilities for embedding images and text using CLIP models."""

from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetInfo, Features, Value, load_from_disk
from datasets import List as HFList
from loguru import logger
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


class PretrainedCLIPModel:
    """Helper class for loading and running a pretrained CLIP model."""

    def __init__(self, model_id: str) -> None:
        self.device = self.get_device()
        logger.info(f"Using device: {self.device}")
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


def embed_images(raw_dir: Path, model_id: str, batch_size: int) -> Dataset:
    """Embed images from a dataset on disk using a CLIP model."""
    ds = load_from_disk(raw_dir)

    if len(ds) == 0:
        raise ValueError("Dataset is empty")

    model = PretrainedCLIPModel(model_id=model_id)

    def processor():  # noqa: ANN202
        """A generator to process the dataset images."""
        num_batches = int(np.ceil(len(ds) / batch_size))
        idx = 0

        for batch in tqdm(ds.iter(batch_size=batch_size), total=num_batches):
            sz = len(batch["image"])
            batch_imgs = [pilify(img) for img in batch["image"]]
            with torch.inference_mode():
                feats = model(images=batch_imgs)

            ids = batch["id"] if "id" in batch else list(range(idx, idx + sz))
            labels = batch["label"] if "label" in batch else [0] * sz

            # Iterate over the batch since the generator is only supposed to yield a single datum
            for i in range(sz):
                yield {"id": ids[i], "embedding": feats[i], "label": labels[i]}

            idx += sz

    features = Features(
        {
            # `id` column to be of type int64
            "id": Value("int64"),
            # `embedding` column is of type datasets.List[float32]
            "embedding": HFList(feature=Value("float32"), length=model.embedding_dim),
            "label": Value("int32"),
        }
    )
    ds_with_embeddings = Dataset.from_generator(
        processor,
        features=features,
    )

    # Update the datset info with the new features
    ds_info = vars(ds.info)
    ds_info.pop("features")  # remove the old features attribute
    dataset_info = DatasetInfo(features=features, **ds_info)
    # directly access attribute since no setter for `info`
    ds_with_embeddings._info = dataset_info  # noqa: SLF001

    return ds_with_embeddings


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
