"""Pytest configuration for shared test setup."""

import os
import sys
from functools import partialmethod
from pathlib import Path

import pytest
from loguru import logger
from tqdm import tqdm

# Disable tqdm bars in tests
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


@pytest.fixture(name="source_dir")
def source_dir_fixture():
    """A fixture for the source directory."""
    # Add the source directory to the fake filesystem so everything can download correctly.
    source_dir = Path(__file__).parent.parent
    return source_dir


@pytest.fixture(name="fixtures_dir")
def fixtures_dir_fixture(source_dir):
    return source_dir / "tests" / "fixtures"


@pytest.fixture(name="config_yaml")
def config_yaml_fixture(source_dir, fixtures_dir):
    """The config as a yaml file within a fake source directory."""
    config_file = fixtures_dir / "inquire_benchmark_small.yaml"

    return config_file


pytest_plugins = ["fixtures.conftest"]

# Set logging level to CRITICAL so it doesn't show
# in test output but is still captured for testing.
logger.remove()
logger.add(sys.stderr, level="CRITICAL")


# Keep thread counts low and avoid at-fork init issues that can trip FAISS/Torch on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("FAISS_DISABLE_GPU", "1")
