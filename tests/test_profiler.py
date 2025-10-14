"""Tests for the lightweight profiler utility, including container tracking."""

import json
import time

from inatinqperf.utils import profiler


def test_profiler_writes_metrics_and_json(tmp_path):
    """Profiler captures process metrics and persists them to disk."""
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
    """A failure while sampling RSS should not bubble up to the caller."""
    # Patch proc.memory_info to raise
    step_name = "failcase"
    results_dir = tmp_path / ".results"
    p = profiler.Profiler(step=step_name, results_dir=results_dir)
    monkeypatch.setattr(p.proc, "memory_info", lambda: (_ for _ in ()).throw(RuntimeError("bad")))

    with p:
        p.sample()  # should not raise despite error

    assert p.metrics["step"] == step_name
    assert "wall_time_sec" in p.metrics


def test_profiler_multiple_steps_create_distinct_files(tmp_path):
    """Each profiled step writes a distinct metrics file."""
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


class _StubContainer:
    """Minimal container stub returning deterministic stats payloads."""

    def __init__(self) -> None:
        self.id = "stub123"
        self.name = "stub"

    def stats(self, stream: bool = False):  # noqa: FBT001, FBT002
        return {
            "memory_stats": {"usage": 21 * 1024 * 1024},
            "cpu_stats": {
                "cpu_usage": {
                    "total_usage": 200_000_000,
                    "percpu_usage": [100_000_000, 100_000_000],
                },
                "system_cpu_usage": 2_000_000_000,
                "online_cpus": 2,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 100_000_000},
                "system_cpu_usage": 1_000_000_000,
            },
        }


def test_profiler_tracks_container_metrics(tmp_path):
    """Profiler records container CPU and memory stats when a container object is supplied."""
    results_dir = tmp_path / ".results"
    container = _StubContainer()
    p = profiler.Profiler("step-cont", results_dir=results_dir, containers=[container])

    with p as prof:
        prof.sample()
        time.sleep(0.001)

    container_metrics = p.metrics.get("containers")
    assert container_metrics is not None
    assert len(container_metrics) == 1
    metrics = container_metrics[0]
    assert metrics["id"] == container.id
    assert metrics["name"] == container.name
    assert metrics["samples"] >= 2
    assert metrics["cpu_avg_percent"] > 0
    assert metrics["mem_max_mb"] >= metrics["mem_avg_mb"] > 0


class _StubDockerClient:
    """Docker client stub returning a pre-registered container."""

    def __init__(self, container: _StubContainer) -> None:
        self._container = container
        self.containers = self

    def get(self, identifier: str):  # noqa: D401 - simple passthrough
        if identifier != self._container.id:
            raise RuntimeError("unknown container")
        return self._container


def test_profiler_registers_container_by_identifier(tmp_path):
    """Profiler resolves container identifiers via the docker client."""
    container = _StubContainer()
    docker_client = _StubDockerClient(container)
    results_dir = tmp_path / ".results"

    p = profiler.Profiler(
        "step-ident",
        results_dir=results_dir,
        containers=[container.id],
        docker_client=docker_client,
    )

    with p:
        time.sleep(0.001)

    metrics = p.metrics.get("containers")
    assert metrics is not None
    assert metrics[0]["id"] == container.id
    assert metrics[0]["name"] == container.name
    assert metrics[0]["samples"] >= 1


class _NoIdContainer:
    """Container-like stub lacking an identifier attribute."""

    name = "nameless"

    def stats(self, stream: bool = False):  # noqa: FBT001, FBT002
        return _StubContainer().stats(stream=stream)


def test_profiler_register_container_skips_objects_without_stats(tmp_path):
    """Objects without a stats attribute are ignored."""
    p = profiler.Profiler("step-unsupported", results_dir=tmp_path / ".results")
    p.register_container(object())
    assert not p._container_trackers


def test_profiler_handles_container_without_identifier(tmp_path):
    """Containers that resolve but expose no identifier should be skipped."""
    container = _NoIdContainer()

    class _NoIdDockerClient:
        def __init__(self, ref):
            self._ref = ref
            self.containers = self

        def get(self, identifier: str):
            return self._ref

    docker_client = _NoIdDockerClient(container)
    p = profiler.Profiler(
        "step-noid",
        results_dir=tmp_path / ".results",
        containers=["missing-id"],
        docker_client=docker_client,
    )

    with p:
        time.sleep(0.001)

    assert "containers" not in p.metrics
    assert not p._container_trackers
    assert not p._pending_containers


class _FlakyDockerClient:
    """Docker client that never resolves the requested container."""

    def __init__(self) -> None:
        self.containers = self

    def get(self, identifier: str):
        raise RuntimeError(f"missing container {identifier}")


def test_profiler_skips_unresolvable_containers(tmp_path):
    """Profiler skips containers that cannot be resolved."""
    results_dir = tmp_path / ".results"
    docker_client = _FlakyDockerClient()

    p = profiler.Profiler(
        "step-missing",
        results_dir=results_dir,
        containers=["ghost-container"],
        docker_client=docker_client,
    )

    with p:
        time.sleep(0.001)

    assert "containers" not in p.metrics


class _EventualDockerClient:
    """Docker client that succeeds after an initial failure."""

    def __init__(self, container: _StubContainer) -> None:
        self._container = container
        self._tries = 0
        self.containers = self

    def get(self, identifier: str):
        self._tries += 1
        if self._tries < 2:
            raise RuntimeError("container not ready yet")
        if identifier != self._container.id:
            raise RuntimeError("unknown container")
        return self._container


def test_profiler_resolves_pending_container(tmp_path):
    """Profiler should retry pending identifiers and record stats once available."""
    container = _StubContainer()
    docker_client = _EventualDockerClient(container)
    results_dir = tmp_path / ".results"

    p = profiler.Profiler(
        "step-eventual",
        results_dir=results_dir,
        containers=[container.id],
        docker_client=docker_client,
    )

    with p as prof:
        prof.sample()

    metrics = p.metrics.get("containers")
    assert metrics is not None
    assert metrics[0]["id"] == container.id
    assert metrics[0]["samples"] >= 1


def test_profiler_ensure_docker_client_handles_initialisation_failure(monkeypatch, tmp_path):
    """Failed docker client initialisation should disable container tracking gracefully."""

    class _DummyDocker:
        def from_env(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(profiler, "docker", _DummyDocker())
    results_dir = tmp_path / ".results"
    p = profiler.Profiler(
        "step-nodocker",
        results_dir=results_dir,
        containers=["whatever"],
    )

    with p:
        time.sleep(0.001)

    assert "containers" not in p.metrics
    assert p._docker_client is None


def test_container_tracker_cpu_percent_handles_missing_fields():
    """CPU percentage helper returns None for incomplete or zero-delta snapshots."""
    tracker = profiler._ContainerTracker(container=_StubContainer())

    # Missing cpu deltas -> expect None
    stats = {
        "cpu_stats": {"cpu_usage": {"total_usage": 100}},
        "precpu_stats": {"cpu_usage": {"total_usage": 100}},
    }
    assert tracker._cpu_percent(stats) is None  # type: ignore[arg-type]

    # Zero delta -> expect None
    stats = {
        "cpu_stats": {
            "cpu_usage": {"total_usage": 200, "percpu_usage": [100, 100]},
            "system_cpu_usage": 2000,
            "online_cpus": 2,
        },
        "precpu_stats": {"cpu_usage": {"total_usage": 200}, "system_cpu_usage": 2000},
    }
    assert tracker._cpu_percent(stats) is None  # type: ignore[arg-type]

    # Valid deltas with implicit CPU count -> expect positive percentage
    stats = {
        "cpu_stats": {
            "cpu_usage": {"total_usage": 300, "percpu_usage": [150, 150]},
            "system_cpu_usage": 3000,
        },
        "precpu_stats": {"cpu_usage": {"total_usage": 100}, "system_cpu_usage": 1000},
    }
    cpu_percent = tracker._cpu_percent(stats)  # type: ignore[arg-type]
    assert cpu_percent is not None
    assert cpu_percent > 0


def test_container_tracker_summary_defaults():
    """Summary should return zeroed metrics when no samples were collected."""
    tracker = profiler._ContainerTracker(container=_StubContainer())
    summary = tracker.summary()
    assert summary["samples"] == 0
    assert summary["cpu_avg_percent"] == 0
    assert summary["mem_max_mb"] == 0
