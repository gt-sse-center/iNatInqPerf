from inatinqperf.benchmark.configuration import (
    DatasetConfig,
    Config,
    EmbeddingParams,
    SearchParams,
    VectorDatabaseConfig,
    VectorDatabaseParams,
)
from pathlib import Path


def test_dataset_config(benchmark_yaml):
    config = DatasetConfig(**benchmark_yaml["dataset"])
    assert config.dataset_id == "sagecontinuum/INQUIRE-Benchmark-small"
    assert config.splits == "validation[0:200]"
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
    assert params.nlist == 32768
    assert params.m == 64
    assert params.nbits == 8
    assert params.nprobe == 32


def test_vectordatabase_config(benchmark_yaml):
    config = VectorDatabaseConfig(**benchmark_yaml["vectordb"])
    assert config.type == "faiss.ivfpq"
    assert isinstance(config.params, VectorDatabaseParams)


def test_search_params(benchmark_yaml):
    params = SearchParams(**benchmark_yaml["search"])
    assert params.topk == 10
    assert params.queries_file == Path("benchmark/queries.txt")


def test_config(benchmark_yaml):
    config = Config(**benchmark_yaml)
    assert isinstance(config.dataset, DatasetConfig)
    assert isinstance(config.embedding, EmbeddingParams)
    assert isinstance(config.vectordb, VectorDatabaseConfig)
    assert isinstance(config.search, SearchParams)
    assert isinstance(config.update, dict)
