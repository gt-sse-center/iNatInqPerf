"""Modules with classes for loading configurations with Pydantic validation."""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, PositiveInt, StringConstraints

from inatinqperf.adaptors.enums import Metric

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class DatasetConfig(BaseModel):
    """Dataset definition within the main configuration."""

    dataset_id: NonEmptyStr
    # For multiple splits, HuggingFace accepts a single string with splits concatenated with the `+` symbol.
    splits: str
    directory: Path
    export_images: bool


class EmbeddingParams(BaseModel):
    """Configuration for the embedding model."""

    model_id: NonEmptyStr
    batch_size: PositiveInt
    directory: Path


class VectorDatabaseParams(BaseModel):
    """Configuration for parameters initializing a Vector Database."""

    url: NonEmptyStr = "localhost"
    port: NonEmptyStr  # The vector db clients expect the port as a string
    collection_name: NonEmptyStr = "benchmark"

    metric: Metric
    index_type: NonEmptyStr
    nlist: int | None = None
    m: int | None = None
    nbits: int | None = None
    nprobe: int | None = None
    ef: int = 32
    batch_size: int = 1000

    def to_dict(self) -> dict[str, Any]:
        """Return parameters, including extra fields, omitting unset values."""
        return {key: value for key, value in self.model_dump().items() if value is not None}


class VectorDatabaseConfig(BaseModel):
    """Configuration for Vector Database."""

    type: NonEmptyStr
    params: VectorDatabaseParams


class SearchParams(BaseModel):
    """Configuration for search parameters."""

    topk: int
    queries_file: Path


class ContainerConfig(BaseModel):
    """Configuration for setting up a docker container of the vector database."""

    image: NonEmptyStr
    name: NonEmptyStr
    hostname: str = ""
    ports: dict[str | PositiveInt, PositiveInt]
    environment: dict[str, str] = {}
    volumes: list[str] = []
    command: str | list = ""
    security_opt: list[str] = []
    healthcheck: dict[str, int | str | list]
    network: str = ""


class Config(BaseModel):
    """Class encapsulating benchmark configuration with data validation."""

    dataset: DatasetConfig
    embedding: EmbeddingParams
    containers: Sequence[ContainerConfig] = []
    container_network: str = ""
    vectordb: VectorDatabaseConfig
    search: SearchParams
    update: dict[str, PositiveInt]
