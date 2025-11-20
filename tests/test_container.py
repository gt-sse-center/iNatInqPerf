"""Unit tests for the container utils."""

from docker.errors import APIError
import pytest

from inatinqperf.benchmark.container import container_context


def _run_container_config(config, expected_count):
    try:
        with container_context(config) as containers:
            assert len(containers) == expected_count
            for container in containers:
                assert container.status == "created"
    except APIError as exc:
        if "port is already allocated" in str(exc).lower():
            pytest.skip(f"Docker port busy: {exc}")
        raise


def test_milvus_container(milvus_yaml):
    _run_container_config(milvus_yaml, expected_count=3)


def test_qdrant_container(qdrant_yaml):
    _run_container_config(qdrant_yaml, expected_count=1)


def test_weaviate_container(weaviate_yaml):
    _run_container_config(weaviate_yaml, expected_count=1)
