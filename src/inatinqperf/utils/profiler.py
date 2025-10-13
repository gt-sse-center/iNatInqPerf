"""Lightweight in-process profiler."""

import datetime
import json
import os
import time
import tracemalloc
from pathlib import Path
from types import TracebackType

import psutil
from loguru import logger

# Set logger to record the calling function in the profiler
logger = logger.opt(depth=1)


class Profiler:
    """Lightweight in-process profiler.

      - wall_time_sec, cpu_time_sec
      - Python heap peak (tracemalloc)
      - rss_avg_mb, rss_max_mb (process Resident Set Size aka `RSS` snapshots)

    For CPU flamegraphs, run the command via py-spy externally.
    """

    def __init__(self, step: str, results_dir: Path = Path(".results")) -> None:
        """Initialize profiler."""
        self.step = step
        self.results_dir = results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        self.proc = psutil.Process(os.getpid())
        self.rss_samples = []

        self._t0 = None
        self._cpu0 = None
        self.metrics = {}

    def __enter__(self) -> "Profiler":
        """Start profiling context."""
        self._t0 = time.perf_counter()
        self._cpu0 = time.process_time()
        tracemalloc.start()
        return self

    def sample(self) -> None:
        """Sample current RSS memory usage. Can be called multiple times during the profiled block."""
        try:
            self.rss_samples.append(self.proc.memory_info().rss)
        except Exception:
            logger.info("[PROFILE] Warning: failed to sample RSS")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        wall = time.perf_counter() - self._t0
        cpu = time.process_time() - self._cpu0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rss_avg = (sum(self.rss_samples) / len(self.rss_samples) / (1024 * 1024)) if self.rss_samples else 0.0
        rss_max = (max(self.rss_samples) / (1024 * 1024)) if self.rss_samples else 0.0

        self.metrics = {
            "step": self.step,
            "wall_time_sec": round(wall, 4),
            "cpu_time_sec": round(cpu, 4),
            "py_heap_peak_mb": round(peak / (1024 * 1024), 3),
            "rss_avg_mb": round(rss_avg, 3),
            "rss_max_mb": round(rss_max, 3),
            "profiler": "builtin",
        }

        ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        path = self.results_dir / f"step-{self.step}-{ts}.json"

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"[PROFILE] {self.metrics}")  # noqa: G004
