"""Modules with classes for loading configurations with Pydantic validation."""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field, PositiveInt, StringConstraints
from simpleeval import simple_eval

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


class ContainerHealthCheck(BaseModel):
    """The health-check parameters for a docker container.

    This has added validators which help make the config file more readable.
    """

    @staticmethod
    def ensure_int(value: str | int) -> int:
        """Helper method to ensure the passed in value is an int before we perform validation.

        Allows passing in expressions in the yaml file for better readability.
        """
        try:
            # In case `value` is an expression string, e.g. '3 * 10**9'
            return simple_eval(value)
        except Exception:
            # If it is an int or simple string, then just run `int` on it
            return int(value)

    test: str | list = ""
    interval: Annotated[int, Field(default=3 * 10**9), BeforeValidator(ensure_int)]
    timeout: Annotated[int, Field(default=2 * 10**9), BeforeValidator(ensure_int)]
    retries: int = 3
    start_period: int = 0


class ContainerConfig(BaseModel):
    """Configuration for setting up a docker container of the vector database."""

    image: NonEmptyStr
    name: NonEmptyStr
    hostname: str | None = None
    ports: dict[str | PositiveInt, PositiveInt]
    environment: dict[str, str] = {}
    volumes: list[str] = []
    command: str | list = ""
    security_opt: list[str] = []
    healthcheck: ContainerHealthCheck
    network: str = ""

    def __init__(self, **data) -> None:
        """Constructor to convert `healthcheck` to dict after validation."""
        super().__init__(**data)
        # After validation (which happens in super().__init__),
        # convert the model to a dictionary and store it.
        self.healthcheck = self.healthcheck.model_dump()


class Config(BaseModel):
    """Class encapsulating benchmark configuration with data validation."""

    dataset: DatasetConfig
    embedding: EmbeddingParams
    containers: Sequence[ContainerConfig] = []
    container_network: str = ""
    vectordb: VectorDatabaseConfig
    search: SearchParams
    update: dict[str, PositiveInt]
