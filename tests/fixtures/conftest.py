"""conftest file for test data fixtures."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture(name="benchmark_yaml")
def benchmark_config_fixture(config_yaml: Path):
    with config_yaml.open() as f:
        return yaml.safe_load(f)


@pytest.fixture(name="milvus_yaml")
def milvus_config_fixture(fixtures_dir):
    config_yaml = fixtures_dir / "inquire_milvus.yaml"
    with config_yaml.open() as f:
        return yaml.safe_load(f)


@pytest.fixture(name="qdrant_yaml")
def qdrant_config_fixture(fixtures_dir):
    config_yaml = fixtures_dir / "inquire_qdrant.yaml"
    with config_yaml.open() as f:
        return yaml.safe_load(f)


@pytest.fixture(name="weaviate_yaml")
def weaviate_config_fixture(fixtures_dir):
    config_yaml = fixtures_dir / "inquire_weaviate.yaml"
    with config_yaml.open() as f:
        return yaml.safe_load(f)
