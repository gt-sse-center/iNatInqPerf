"""conftest file for test data fixtures."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture(name="benchmark_yaml")
def benchmark_config_fixture(config_yaml: Path):
    with config_yaml.open() as f:
        return yaml.safe_load(f)
