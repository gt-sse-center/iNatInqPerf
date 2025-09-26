"""Pytest configuration for shared test setup."""

import os
import sys
from loguru import logger
import tqdm
import pytest

# Set logging level to ERROR so it doesn't show
# in test output but is still captured for testing.
logger.remove()
logger.add(sys.stderr, level="ERROR")


@pytest.fixture(autouse=True)
def notqdm(monkeypatch):
    """
    Replacement for tqdm that just passes back the iterable.
    Useful to silence `tqdm` in tests.
    """

    def tqdm_replacement(iterable, *args, **kwargs):
        return iterable

    monkeypatch.setattr(tqdm, "tqdm", tqdm_replacement)


# Keep thread counts low and avoid at-fork init issues that can trip FAISS/Torch on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("FAISS_DISABLE_GPU", "1")


def pytest_sessionstart(session):
    """Clamp FAISS OMP threads early in the session."""
    try:
        import faiss  # type: ignore

        if hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(1)
    except Exception:
        # It's okay if FAISS isn't importable here.
        pass
