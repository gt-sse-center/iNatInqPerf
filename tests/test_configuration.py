from pathlib import Path

from inatinqperf.benchmark.configuration import (
    Config,
    DatasetConfig,
    EmbeddingParams,
    SearchParams,
    VectorDatabaseConfig,
    VectorDatabaseParams,
    ContainerConfig,
)


def test_dataset_config(benchmark_yaml):
    config = DatasetConfig(**benchmark_yaml["dataset"])
    assert config.dataset_id == "sagecontinuum/INQUIRE-Benchmark-small"
    assert config.splits == "validation[0:256]"
    assert config.directory == Path("data/inquire_benchmark/raw")
    assert config.export_images is True


def test_embedding_params(benchmark_yaml):
    params = EmbeddingParams(**benchmark_yaml["embedding"])
    assert params.model_id == "openai/clip-vit-base-patch32"
    assert params.batch_size == 64
    assert params.directory == Path("data/inquire_benchmark/emb")


def test_vectordatabase_params(benchmark_yaml):
    params = VectorDatabaseParams(**benchmark_yaml["vectordb"]["params"])
    assert params.metric == "ip"
    assert params.index_type == "IVFPQ"
    assert params.nlist == 32768
    assert params.m == 16
    assert params.nbits == 2
    assert params.nprobe == 4


def test_vectordatabase_config(benchmark_yaml):
    config = VectorDatabaseConfig(**benchmark_yaml["vectordb"])
    assert config.type == "faiss"
    assert isinstance(config.params, VectorDatabaseParams)


def test_search_params(benchmark_yaml):
    params = SearchParams(**benchmark_yaml["search"])
    assert params.topk == 10
    assert params.queries_file == Path("benchmark/queries.txt")


def test_container_config_list(benchmark_yaml):
    containers = [ContainerConfig(**cc) for cc in benchmark_yaml["containers"]]
    assert len(containers) == 1
    assert containers[0].image == "qdrant/qdrant"
    assert len(containers[0].ports) == 1
    assert containers[0].healthcheck["test"] == "healthcheck test"


def test_config(benchmark_yaml):
    config = Config(**benchmark_yaml)
    assert isinstance(config.dataset, DatasetConfig)
    assert isinstance(config.embedding, EmbeddingParams)
    assert isinstance(config.containers, list)
    assert isinstance(config.containers[0], ContainerConfig)
    assert isinstance(config.vectordb, VectorDatabaseConfig)
    assert isinstance(config.search, SearchParams)
    assert isinstance(config.update, dict)
