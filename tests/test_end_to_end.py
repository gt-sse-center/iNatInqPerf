"""Unit tests to verify end to end running of the benchmark."""

import pytest

from inatinqperf.benchmark import Benchmarker


@pytest.mark.parametrize(
    "config_filename",
    [
        "inquire_qdrant.yaml",
        "inquire_milvus.yaml",
        "inquire_weaviate.yaml",
    ],
)
def test_full_run(fixtures_dir, config_filename):
    config_file = fixtures_dir / config_filename

    benchmarker = Benchmarker(config_file, base_path=fixtures_dir.parent.parent)
    benchmarker.run()
