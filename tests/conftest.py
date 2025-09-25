# tests/conftest.py
"""Pytest configuration for shared test setup."""

import importlib
import logging
import os

import pytest
from loguru import logger

# Keep thread counts low and avoid at-fork init issues that can trip FAISS/Torch on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("FAISS_DISABLE_GPU", "1")


def pytest_configure(config):
    """Eagerly (re)load key modules after coverage has started."""
    for name in (
        "inatinqperf.utils.embed",
        "inatinqperf.utils.dataio",
        "inatinqperf.utils.profiler",
        "inatinqperf.adaptors.base",
    ):
        module = importlib.import_module(name)
        importlib.reload(module)
