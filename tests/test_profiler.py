# tests/test_profiler.py
import json
import time

from inatinqperf.utils import profiler


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
