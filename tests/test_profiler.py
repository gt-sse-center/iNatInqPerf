# tests/test_profiler.py
import json
import time
import uuid

import pytest

from inatinqperf.configuration import ContainerConfig
from inatinqperf.utils import profiler


@pytest.fixture(scope="module", name="docker_client")
def docker_client_fixture():
    """Provide a docker client for tests, skipping if docker SDK is unavailable."""
    docker_mod = pytest.importorskip("docker")
    profiler.docker = docker_mod
    client = docker_mod.from_env()
    try:
        yield client
    finally:
        client.close()


@pytest.fixture(name="running_test_containers")
def running_containers_fixture(docker_client):
    """Run two short-lived containers for profiling tests."""
    docker_mod = profiler.docker
    base_image = "busybox:latest"
    docker_client.images.pull(base_image)

    containers = []
    configs = []
    tagged_images: list[str] = []

    for idx in range(2):
        name = f"inatinqperf-test-{uuid.uuid4().hex[:6]}"

        image_ref = base_image
        if idx == 1:
            suffix = uuid.uuid4().hex[:6]
            repository = f"inatinqperf/test-profiler-{suffix}"
            docker_client.images.get(base_image).tag(repository, tag="latest")
            image_ref = f"{repository}:latest"
            tagged_images.append(image_ref)

        container = docker_client.containers.run(
            image_ref,
            name=name,
            command=["sh", "-c", "while true; do sleep 1; done"],
            detach=True,
            remove=True,
        )
        # Allow container to reach running state before sampling
        container.reload()
        containers.append(container)

        config_name = name if idx == 0 else None
        configs.append(
            ContainerConfig(
                name=config_name,
                image=image_ref,
                ports={},
                healthcheck={"test": "echo ok"},
            )
        )

    try:
        yield configs, containers
    finally:
        for container in containers:
            try:
                container.stop(timeout=5)
            except docker_mod.errors.NotFound:
                continue
            except docker_mod.errors.APIError:
                try:
                    container.kill()
                except docker_mod.errors.APIError:
                    pass
        for image in tagged_images:
            try:
                docker_client.images.remove(image, force=True)
            except docker_mod.errors.APIError:
                pass


def test_profiler_writes_metrics_and_json(tmp_path):
    # Patch results_dir so output is written into tmp_path
    step_name = "unit"
    results_dir = tmp_path / ".results"
    p = profiler.Profiler(step=step_name, results_dir=results_dir)

    with p as prof:
        # Do some work and sample memory
        _ = [i for i in range(1000)]
        prof.sample()
        time.sleep(0.01)

    # 1) Metrics are available and contain expected keys
    metrics = p.metrics
    assert metrics["step"] == step_name
    assert "wall_time_sec" in metrics
    assert "cpu_time_sec" in metrics
    assert "py_heap_peak_mb" in metrics
    assert "rss_avg_mb" in metrics
    assert "rss_max_mb" in metrics
    assert metrics["profiler"] == "builtin"

    # 2) JSON file written
    files = list(results_dir.glob(f"step-{step_name}-*.json"))
    assert len(files) == 1
    with open(files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == metrics  # contents round-trip

    # 3) Sanity: metrics values are non-negative
    assert all(v >= 0 for k, v in metrics.items() if isinstance(v, (int, float)))


def test_profiler_sample_handles_exceptions(monkeypatch, tmp_path):
    # Patch proc.memory_info to raise
    step_name = "failcase"
    results_dir = tmp_path / ".results"
    p = profiler.Profiler(step=step_name, results_dir=results_dir)
    monkeypatch.setattr(p.proc, "memory_info", lambda: (_ for _ in ()).throw(RuntimeError("bad")))

    with p:
        p.sample()  # should not raise despite error


def test_profiler_multiple_steps_create_distinct_files(tmp_path):
    results_dir = tmp_path / ".results"

    # First step
    p1 = profiler.Profiler("step1", results_dir=results_dir)
    with p1:
        time.sleep(0.001)

    # Second step
    p2 = profiler.Profiler("step2", results_dir=results_dir)
    with p2:
        time.sleep(0.001)

    files = list(results_dir.glob("*.json"))
    assert len(files) == 2
    names = [f.name for f in files]
    assert any("step1" in n for n in names)
    assert any("step2" in n for n in names)
    # Metrics differ at least in step name
    assert p1.metrics["step"] == "step1"
    assert p2.metrics["step"] == "step2"


def test_extract_metrics_returns_expected_values():
    mb = profiler._MB

    sample1 = {
        "cpu_stats": {
            "cpu_usage": {"total_usage": 200_000_000, "percpu_usage": [100_000_000, 100_000_000]},
            "system_cpu_usage": 400_000_000,
            "online_cpus": 2,
        },
        "precpu_stats": {
            "cpu_usage": {"total_usage": 100_000_000, "percpu_usage": [50_000_000, 50_000_000]},
            "system_cpu_usage": 300_000_000,
            "online_cpus": 2,
        },
        "memory_stats": {"usage": 150 * mb, "limit": 800 * mb, "stats": {"cache": 20 * mb}},
        "networks": {"eth0": {"rx_bytes": 1_000_000, "tx_bytes": 500_000}},
        "blkio_stats": {
            "io_service_bytes_recursive": [{"op": "Read", "value": 100_000}, {"op": "Write", "value": 50_000}]
        },
        "pids_stats": {"current": 5},
    }

    sample2 = {
        "cpu_stats": {
            "cpu_usage": {"total_usage": 360_000_000, "percpu_usage": [180_000_000, 180_000_000]},
            "system_cpu_usage": 520_000_000,
            "online_cpus": 2,
        },
        "precpu_stats": sample1["cpu_stats"],
        "memory_stats": {"usage": 200 * mb, "limit": 800 * mb, "stats": {"cache": 25 * mb}},
        "networks": {"eth0": {"rx_bytes": 2_500_000, "tx_bytes": 1_100_000}},
        "blkio_stats": {
            "io_service_bytes_recursive": [
                {"op": "Read", "value": 260_000},
                {"op": "Write", "value": 120_000},
            ]
        },
        "pids_stats": {"current": 6},
    }

    sample3 = {
        "cpu_stats": {
            "cpu_usage": {"total_usage": 520_000_000, "percpu_usage": [260_000_000, 260_000_000]},
            "system_cpu_usage": 640_000_000,
            "online_cpus": 2,
        },
        "precpu_stats": sample2["cpu_stats"],
        "memory_stats": {"usage": 220 * mb, "limit": 800 * mb, "stats": {"cache": 30 * mb}},
        "networks": {"eth0": {"rx_bytes": 3_400_000, "tx_bytes": 1_900_000}},
        "blkio_stats": {
            "io_service_bytes_recursive": [
                {"op": "Read", "value": 420_000},
                {"op": "Write", "value": 210_000},
            ]
        },
        "pids_stats": {"current": 7},
    }

    metrics_a = profiler.ContainerStatsCollector._extract_metrics(sample1)
    metrics_b = profiler.ContainerStatsCollector._extract_metrics(sample2)
    metrics_c = profiler.ContainerStatsCollector._extract_metrics(sample3)

    assert metrics_a["cpu_percent"] >= 0.0
    assert metrics_b["cpu_percent"] >= 0.0
    assert metrics_c["cpu_percent"] >= 0.0

    assert metrics_a["mem_usage_bytes"] == 150 * mb - 20 * mb
    assert metrics_b["mem_usage_bytes"] == 200 * mb - 25 * mb
    assert metrics_c["mem_usage_bytes"] == 220 * mb - 30 * mb

    assert metrics_c["net_rx_bytes"] > metrics_b["net_rx_bytes"]
    assert metrics_c["net_tx_bytes"] > metrics_a["net_tx_bytes"]


def test_container_collector_accepts_mapping():
    cfg = {
        "image": "busybox:latest",
        "name": "sample",
        "ports": {},
        "healthcheck": {"test": "echo ok"},
    }

    collector = profiler.ContainerStatsCollector([cfg])
    assert isinstance(collector.configs[0], ContainerConfig)


def test_container_collector_profiles_running_containers(running_test_containers):
    configs, containers = running_test_containers

    collector = profiler.ContainerStatsCollector(configs)
    collector.start()

    # Capture multiple samples to populate averages and deltas
    time.sleep(0.2)
    collector.sample()
    time.sleep(0.2)
    collector.sample()

    metrics = collector.finalize()

    assert len(metrics) == len(containers)

    # First config resolves by name; second resolves via image lookup.
    expected_labels = {container.name for container in containers}
    assert set(metrics.keys()) == expected_labels

    for label, data in metrics.items():
        assert data["samples"] >= 2
        assert data["cpu_percent_avg"] >= 0
        assert data["mem_usage_mb_avg"] >= 0
