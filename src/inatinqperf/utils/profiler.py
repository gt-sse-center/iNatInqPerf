"""Lightweight in-process profiler."""

from __future__ import annotations

import datetime
import json
import os
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

import psutil
from loguru import logger

# Set logger to record the calling function in the profiler
logger = logger.opt(depth=1)

try:  # Optional dependency used only when container tracking is enabled.
    import docker
    from docker.models.containers import Container
except ImportError:  # pragma: no cover - exercised when docker is unavailable
    docker = None
    Container = Any  # type: ignore[misc]


@dataclass(slots=True)
class _ContainerTracker:
    """Collect CPU and memory metrics for a Docker container."""

    container: Container
    cpu_samples: list[float] = field(default_factory=list)
    memory_samples: list[float] = field(default_factory=list)

    def sample(self) -> None:
        """Capture a stats snapshot, ignoring transient failures."""
        try:
            stats = self.container.stats(stream=False)
        except Exception:  # pragma: no cover - relies on docker API behaviour
            logger.opt(exception=True).debug("[PROFILE] Failed to gather container stats")
            return

        try:
            memory_usage = stats.get("memory_stats", {}).get("usage")
            if memory_usage is not None:
                self.memory_samples.append(memory_usage / (1024 * 1024))

            cpu_percent = self._cpu_percent(stats)
            if cpu_percent is not None:
                self.cpu_samples.append(cpu_percent)
        except Exception:  # pragma: no cover - defensive against schema drift
            logger.opt(exception=True).debug("[PROFILE] Unexpected container stats payload")

    def summary(self) -> dict[str, Any]:
        """Return aggregate metrics for the tracked container."""
        samples = max(len(self.memory_samples), len(self.cpu_samples))
        return {
            "id": getattr(self.container, "id", "unknown"),
            "name": getattr(self.container, "name", "unknown"),
            "samples": samples,
            "cpu_avg_percent": round(mean(self.cpu_samples), 3) if self.cpu_samples else 0.0,
            "cpu_max_percent": round(max(self.cpu_samples), 3) if self.cpu_samples else 0.0,
            "mem_avg_mb": round(mean(self.memory_samples), 3) if self.memory_samples else 0.0,
            "mem_max_mb": round(max(self.memory_samples), 3) if self.memory_samples else 0.0,
        }

    @staticmethod
    def _cpu_percent(stats: dict[str, Any]) -> float | None:
        """Calculate CPU percentage using Docker's documented formula."""
        cpu_stats = stats.get("cpu_stats", {})
        precpu_stats = stats.get("precpu_stats", {})

        total_usage = cpu_stats.get("cpu_usage", {}).get("total_usage")
        pre_total_usage = precpu_stats.get("cpu_usage", {}).get("total_usage")
        system_usage = cpu_stats.get("system_cpu_usage")
        pre_system_usage = precpu_stats.get("system_cpu_usage")

        if None in (total_usage, system_usage) or pre_total_usage is None or pre_system_usage is None:
            return None

        cpu_delta = total_usage - pre_total_usage
        system_delta = system_usage - pre_system_usage
        if cpu_delta <= 0 or system_delta <= 0:
            return None

        online_cpus = cpu_stats.get("online_cpus")
        if not online_cpus:
            percpu = cpu_stats.get("cpu_usage", {}).get("percpu_usage")
            online_cpus = len(percpu) if percpu else 1

        return (cpu_delta / system_delta) * online_cpus * 100.0

try:  # Optional dependency used only when container tracking is enabled.
    import docker
    from docker.models.containers import Container
except ImportError:  # pragma: no cover - exercised when docker is unavailable
    docker = None
    Container = Any  # type: ignore[misc]


@dataclass(slots=True)
class _ContainerTracker:
    """Collect CPU and memory metrics for a Docker container."""

    container: Container
    cpu_samples: list[float] = field(default_factory=list)
    memory_samples: list[float] = field(default_factory=list)

    def sample(self) -> None:
        """Capture a stats snapshot, ignoring transient failures."""
        try:
            stats = self.container.stats(stream=False)
        except Exception:  # pragma: no cover - relies on docker API behaviour
            logger.opt(exception=True).debug("[PROFILE] Failed to gather container stats")
            return

        try:
            memory_usage = stats.get("memory_stats", {}).get("usage")
            if memory_usage is not None:
                self.memory_samples.append(memory_usage / (1024 * 1024))

            cpu_percent = self._cpu_percent(stats)
            if cpu_percent is not None:
                self.cpu_samples.append(cpu_percent)
        except Exception:  # pragma: no cover - defensive against schema drift
            logger.opt(exception=True).debug("[PROFILE] Unexpected container stats payload")

    def summary(self) -> dict[str, Any]:
        """Return aggregate metrics for the tracked container."""
        samples = max(len(self.memory_samples), len(self.cpu_samples))
        return {
            "id": getattr(self.container, "id", "unknown"),
            "name": getattr(self.container, "name", "unknown"),
            "samples": samples,
            "cpu_avg_percent": round(mean(self.cpu_samples), 3) if self.cpu_samples else 0.0,
            "cpu_max_percent": round(max(self.cpu_samples), 3) if self.cpu_samples else 0.0,
            "mem_avg_mb": round(mean(self.memory_samples), 3) if self.memory_samples else 0.0,
            "mem_max_mb": round(max(self.memory_samples), 3) if self.memory_samples else 0.0,
        }

    @staticmethod
    def _cpu_percent(stats: dict[str, Any]) -> float | None:
        """Calculate CPU percentage using Docker's documented formula."""
        cpu_stats = stats.get("cpu_stats", {})
        precpu_stats = stats.get("precpu_stats", {})

        total_usage = cpu_stats.get("cpu_usage", {}).get("total_usage")
        pre_total_usage = precpu_stats.get("cpu_usage", {}).get("total_usage")
        system_usage = cpu_stats.get("system_cpu_usage")
        pre_system_usage = precpu_stats.get("system_cpu_usage")

        if None in (total_usage, system_usage) or pre_total_usage is None or pre_system_usage is None:
            return None

        cpu_delta = total_usage - pre_total_usage
        system_delta = system_usage - pre_system_usage
        if cpu_delta <= 0 or system_delta <= 0:
            return None

        online_cpus = cpu_stats.get("online_cpus")
        if not online_cpus:
            percpu = cpu_stats.get("cpu_usage", {}).get("percpu_usage")
            online_cpus = len(percpu) if percpu else 1

        return (cpu_delta / system_delta) * online_cpus * 100.0


class Profiler:
    """Lightweight in-process profiler.

      - wall_time_sec, cpu_time_sec
      - Python heap peak (tracemalloc)
      - rss_avg_mb, rss_max_mb (process Resident Set Size aka `RSS` snapshots)

    For CPU flamegraphs, run the command via py-spy externally.
    """

    def __init__(
        self,
        step: str,
        results_dir: Path = Path(".results"),
        *,
        containers: Sequence[Container | str] | None = None,
        docker_client: docker.DockerClient | None = None,
    ) -> None:
        """Initialize profiler."""
        self.step = step
        self.results_dir = results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        self.proc = psutil.Process(os.getpid())
        self.rss_samples = []
        self._docker_client = docker_client
        self._container_trackers: dict[str, _ContainerTracker] = {}
        self._pending_containers: set[str] = set()

        self._t0 = None
        self._cpu0 = None
        self.metrics = {}

        if containers:
            for container in containers:
                self.register_container(container)

    def __enter__(self) -> Self:
        """Start profiling context."""
        self._t0 = time.perf_counter()
        self._cpu0 = time.process_time()
        tracemalloc.start()
        # Capture an initial baseline for containers if configured.
        self._sample_containers()
        return self

    def sample(self) -> None:
        """Sample current RSS memory usage. Can be called multiple times during the profiled block."""
        try:
            self.rss_samples.append(self.proc.memory_info().rss)
        except Exception:
            logger.info("[PROFILE] Warning: failed to sample RSS")
        self._sample_containers()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Ensure we have an up-to-date snapshot before computing aggregates.
        self._sample_containers()
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

        container_metrics = self._summarise_containers()
        if container_metrics:
            self.metrics["containers"] = container_metrics

        ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        path = self.results_dir / f"step-{self.step}-{ts}.json"

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"[PROFILE] {self.metrics}")  # noqa: G004

    def register_container(self, container: Container | str) -> None:
        """Begin tracking resource usage for the specified container."""
        if isinstance(container, str):
            container_obj = self._resolve_container(container)
            if container_obj is None:
                self._pending_containers.add(container)
                return
        else:
            container_obj = self._resolve_container(container)
            if container_obj is None:
                return

        container_id = getattr(container_obj, "id", None)
        if container_id is None:
            logger.info(f"[PROFILE] Skipping container without identifier: {container_obj}")
            return

        if container_id not in self._container_trackers:
            self._container_trackers[container_id] = _ContainerTracker(container_obj)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sample_containers(self) -> None:
        if self._pending_containers:
            for ref in list(self._pending_containers):
                container_obj = self._resolve_container(ref)
                if container_obj is None:
                    continue
                container_id = getattr(container_obj, "id", None)
                if container_id is None:
                    logger.info(f"[PROFILE] Skipping container without identifier: {container_obj}")
                    self._pending_containers.discard(ref)
                    continue
                self._container_trackers[container_id] = _ContainerTracker(container_obj)
                self._pending_containers.discard(ref)

        if not self._container_trackers:
            return
        for tracker in self._container_trackers.values():
            tracker.sample()

    def _summarise_containers(self) -> list[dict[str, Any]]:
        if not self._container_trackers:
            return []
        return [tracker.summary() for tracker in self._container_trackers.values()]

    def _resolve_container(self, container: Container | str) -> Container | None:
        if getattr(container, "stats", None):
            return container  # Already a container-like object injected by caller.

        if isinstance(container, str):
            client = self._ensure_docker_client()
            if client is None:
                logger.info(f"[PROFILE] Docker client unavailable; skipping container {container}")
                return None
            try:
                return client.containers.get(container)
            except Exception:
                logger.opt(exception=True).debug(f"[PROFILE] Failed to resolve container {container}")
                return None

        logger.info(f"[PROFILE] Unsupported container handle: {container}")
        return None

    def _ensure_docker_client(self) -> docker.DockerClient | None:
        if self._docker_client is not None:
            return self._docker_client
        if docker is None:
            return None
        try:
            self._docker_client = docker.from_env()
        except Exception:
            logger.opt(exception=True).debug("[PROFILE] Failed to initialise docker client")
            self._docker_client = None
        return self._docker_client
