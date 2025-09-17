"""Common fixtures for vector DB tests."""

import pytest


@pytest.fixture
def collection_name():
    """Provide a default collection name for tests."""
    return "test_collection"
