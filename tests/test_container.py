"""Unit tests for the container utils."""

from inatinqperf.benchmark.container import container_context


def test_milvus_container(milvus_yaml):
    with container_context(milvus_yaml) as containers:
        assert len(containers) == 3
        for container in containers:
            assert container.status == "created"


def test_qdrant_container(qdrant_yaml):
    with container_context(qdrant_yaml) as containers:
        assert len(containers) == 1
        for container in containers:
            assert container.status == "created"


def test_weaviate_container(weaviate_yaml):
    with container_context(weaviate_yaml) as containers:
        assert len(containers) == 1
        for container in containers:
            assert container.status == "created"
