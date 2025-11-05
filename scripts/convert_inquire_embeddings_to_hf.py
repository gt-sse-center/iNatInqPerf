#!/bin/env/python
"""Script to convert the embeddings downloaded from the INQUIRE repository to HuggingFace Dataset format.

This script will also upload the generated dataset to `gt-csse/iNat24-vit-b-16`.

More info: https://github.com/inquire-benchmark/INQUIRE/tree/main/data#embeddings
"""

import os
from collections.abc import Generator
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from datasets import Dataset, Features, Split, Value
from datasets import List as HFList
from dotenv import load_dotenv
from huggingface_hub import login as hf_login
from loguru import logger
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

app = typer.Typer()


def data_generator(file_paths: list[str]) -> Generator:
    """The generator for lazily loading the npy files."""
    idx = 0  # running ID counter

    for path in file_paths:
        arr = np.load(path, mmap_mode="r")  # memory-map each file

        for row in arr:
            yield {"id": idx, "embedding": row}

            idx += 1


def create_dataset(
    num_files: int,
    dim: int,
    embeddings_dir: Path,
    dataset_dir: Path,
    *,
    test_version: bool,
) -> Dataset:
    """Create the dataset from the npy files in `file_paths`."""

    # NOTE: This is very specific to the inquire-inat dataset and may fail if using another dataset.
    file_paths = [embeddings_dir / f"img_emb_{i}.npy" for i in tqdm(range(num_files))]

    logger.info(f"Generating 🤗 dataset with dim:{dim}")
    features = Features(
        {
            # `id` column to be of type int64
            "id": Value("int64"),
            # `embedding` column is of type datasets.List[float32]
            "embedding": HFList(feature=Value("float32"), length=dim),
        },
    )
    if test_version:
        embeddings = np.load(file_paths[0])[:1000]
        dataset = Dataset.from_dict(
            {"id": np.arange(1000).tolist(), "embedding": embeddings}, features=features, split=Split.TRAIN
        )

    else:
        dataset = Dataset.from_generator(
            lambda: data_generator(file_paths),
            features=features,
            split=Split.TRAIN,
        )
    logger.info(f"Full dataset size: {dataset.num_rows}")

    logger.info(f"Saving to: {dataset_dir}")
    # Ensure the dataset directory exists
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save the dataset
    dataset.save_to_disk(dataset_dir)

    return dataset


@app.command("run")
def main(
    embeddings_dir: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=True, help="The directory with all the embeddings as .npy files."
        ),
    ],
    dataset_dir: Annotated[
        Path,
        typer.Argument(exists=False, dir_okay=True, help="The directory where the dataset is saved."),
    ],
    num_files: Annotated[int, typer.Option(help="The number of files to load.")] = 5,
    dim: Annotated[int, typer.Option(help="The embedding size")] = 512,
    force: Annotated[  # noqa: FBT002
        bool, typer.Option("-f", help="Force dataset creation even if the dataset directory already exists.")
    ] = False,
    test_version: Annotated[  # noqa: FBT002
        bool, typer.Option("-t", help="Generate test version of the dataset with 1000 data points.")
    ] = False,
) -> None:
    """Main runner."""
    if (access_token := os.getenv("HUGGINGFACE_ACCESS_TOKEN")) is None:
        raise RuntimeError("Please add HUGGINGFACE_ACCESS_TOKEN=<TOKEN> in .env file")

    if dataset_dir.exists() and not force:
        logger.info("Dataset already exists. Loading...")
        dataset = Dataset.load_from_disk(dataset_path=dataset_dir)
    else:
        dataset = create_dataset(
            num_files, dim, embeddings_dir=embeddings_dir, dataset_dir=dataset_dir, test_version=test_version
        )

    logger.info("Logging into 🤗")
    hf_login(token=access_token)

    hub = "gt-csse/iNat24-vit-b-16"
    if test_version:
        hub = f"{hub}-test"

    logger.info(f"Pushing to 🤗 dataset {hub}")
    dataset.push_to_hub(hub)


if __name__ == "__main__":
    app()
