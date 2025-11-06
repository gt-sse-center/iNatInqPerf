"""Lightweight in-process profiler."""

import datetime
import json
import os
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

import psutil
from loguru import logger
from pydantic import ValidationError

from inatinqperf.benchmark.configuration import ContainerConfig

try:  # pragma: no cover - import guard only
    import docker
except Exception:  # pragma: no cover - gracefully handle absent docker
    docker = None

# Set logger to record the calling function in the profiler
logger = logger.opt(depth=1)

_MB = 1024 * 1024


class ContainerStatsCollector:
    """Collect metrics for one or more Docker containers."""

    def __init__(self, containers: Sequence[ContainerConfig | Mapping[str, Any]] | None) -> None:
        """Set up the collector with optional container descriptions."""
        # We accept zero or more container configs; absence disables container profiling.
        if not containers:
            logger.info("Container profiling not enabled (no container provided)")
            self.configs: list[ContainerConfig] = []
        else:
            parsed: list[ContainerConfig] = []
            for config in containers:
                if isinstance(config, ContainerConfig):
                    parsed.append(config)
                else:
                    try:
                        data = dict(config)
                        healthcheck = data.get("healthcheck")
                        if isinstance(healthcheck, str):
                            data["healthcheck"] = {"test": healthcheck}
                        parsed.append(ContainerConfig(**data))
                    except (TypeError, ValidationError) as exc:
                        msg = f"Invalid container configuration: {config!r}"
                        raise ValueError(msg) from exc
            self.configs = parsed

        self.docker_client = None
        self._containers: dict[str, Any] = {}
        self._container_labels: dict[str, str] = {}
        self._samples: dict[str, list[dict[str, float]]] = {}

    def start(self) -> None:
        """Open a Docker client and resolve the container reference."""
        if not self.configs or docker is None:
            return
        try:
            self.docker_client = docker.from_env()
        except docker.errors.DockerException as exc:
            logger.warning(f"Container profiling not enabled (Docker error: {exc!s})")
            self.docker_client = None
            return

        for config in self.configs:
            container = self._resolve_container(config)
            if container is None:
                continue

            # Skip duplicate references so we only poll each container once.
            key = container.id
            label = container.name or container.id[:12]
            if key in self._containers:
                logger.info(
                    f"Container {label} already tracked; ignoring duplicate configuration",
                )
                continue

            self._containers[key] = container
            self._container_labels[key] = label
            self._samples[key] = []

        if not self._containers:
            logger.info("Container profiling not enabled (no containers resolved)")

    def sample(self) -> None:
        """Record a single metrics snapshot for the resolved container."""
        if not self._containers:
            return

        # Pull one snapshot of stats from Docker and stash simplified metrics.
        for key, container in list(self._containers.items()):
            try:
                stats = container.stats(stream=False)
            except Exception as exc:  # pragma: no cover - docker stats failure
                label = self._container_labels.get(key, key[:12])
                logger.warning(f"Failed to sample container {label} ({exc!s}); skipping")
                continue
            self._samples[key].append(self._extract_metrics(stats))

    def finalize(self) -> dict[str, dict[str, float]]:
        """Aggregate all sampled metrics and tear down Docker resources."""
        # Aggregate samples (if any) and ensure Docker resources are released.
        result: dict[str, dict[str, float]] = {}
        for key, samples in self._samples.items():
            if not samples:
                continue
            label = self._container_labels.get(key, key[:12])
            result[label] = self._aggregate_samples(samples)

        if self.docker_client is not None:
            self.docker_client.close()
            self.docker_client = None
        self._containers.clear()
        self._container_labels.clear()
        self._samples.clear()

        return result

    def _resolve_container(self, config: "ContainerConfig") -> object | None:
        """Return the running Docker container described by ``config``.

        We look up by explicit name when provided, otherwise fall back to
        the container image. On lookup failure or client errors we emit a
        warning and return ``None`` so profiling can continue without that
        container.
        """
        if self.docker_client is None:
            return None

        try:
            if config.name:
                matches = self.docker_client.containers.list(filters={"name": config.name})
                container = matches[0] if matches else None
                if container is None:
                    logger.warning(
                        f"Container profiling not enabled (no running container named {config.name})",
                    )
                    return None
            else:
                matches = self.docker_client.containers.list(filters={"ancestor": config.image})
                container = matches[0] if matches else None
                if container is None:
                    logger.warning(
                        f"Container profiling not enabled (no running container for image {config.image})",
                    )
                    return None
        except docker.errors.DockerException as exc:
            identifier = config.name or config.image
            logger.warning(
                f"Container profiling not enabled (Docker error resolving {identifier}: {exc!s})",
            )
            return None

        return container

    @staticmethod
    def _extract_metrics(stats: Mapping[str, Any]) -> dict[str, float]:
        """Convert a Docker stats payload into user-facing metrics."""

        def dig(mapping: Mapping[str, Any], *keys: str, default: object = 0) -> object:
            """Walk nested mappings safely, returning a default if any step fails."""
            current: Any = mapping
            for key in keys:
                if not isinstance(current, Mapping):
                    return default
                current = current.get(key)
                if current is None:
                    return default
            return current

        # CPU deltas are computed relative to the previous snapshot provided by Docker.
        cpu_delta = dig(stats, "cpu_stats", "cpu_usage", "total_usage") - dig(
            stats, "precpu_stats", "cpu_usage", "total_usage"
        )
        system_delta = dig(stats, "cpu_stats", "system_cpu_usage") - dig(
            stats, "precpu_stats", "system_cpu_usage"
        )
        online_cpus = (
            dig(stats, "cpu_stats", "online_cpus")
            or len(dig(stats, "cpu_stats", "cpu_usage", "percpu_usage", default=[]))
            or 1
        )
        if cpu_delta > 0 and system_delta > 0:
            cpu_percent = cpu_delta / system_delta * online_cpus * 100.0
        else:
            cpu_percent = 0.0

        # Memory usage is reported in raw bytes, subtracting cache to align with resident set usage.
        usage = dig(stats, "memory_stats", "usage")
        cache = dig(stats, "memory_stats", "stats", "cache")
        mem_usage = max(usage - cache, 0)
        mem_limit = dig(stats, "memory_stats", "limit")
        mem_percent = (mem_usage / mem_limit * 100.0) if mem_limit else 0.0

        # Network and block IO bytes are accumulated counters, so we capture absolute totals.
        networks = dig(stats, "networks", default={})
        rx_bytes = sum(net.get("rx_bytes", 0) for net in networks.values())
        tx_bytes = sum(net.get("tx_bytes", 0) for net in networks.values())

        blkio_entries = dig(stats, "blkio_stats", "io_service_bytes_recursive", default=[]) or []
        reads = sum(entry.get("value", 0) for entry in blkio_entries if entry.get("op", "").lower() == "read")
        writes = sum(
            entry.get("value", 0) for entry in blkio_entries if entry.get("op", "").lower() == "write"
        )

        pids = dig(stats, "pids_stats", "current")

        return {
            "cpu_percent": float(cpu_percent),
            "mem_usage_bytes": float(mem_usage),
            "mem_limit_bytes": float(mem_limit),
            "mem_percent": float(mem_percent),
            "net_rx_bytes": float(rx_bytes),
            "net_tx_bytes": float(tx_bytes),
            "block_read_bytes": float(reads),
            "block_write_bytes": float(writes),
            "pids": float(pids),
        }

    @staticmethod
    def _aggregate_samples(samples: Sequence[dict[str, float]]) -> dict[str, float]:
        """Summarise collected container metrics.

        Each sample is a flat mapping produced by ``_extract_metrics`` for a single poll.
        The return payload exposes counts, averages, maxima, and start/end deltas that
        feed directly into the JSON profiler report.
        """
        count = len(samples)
        if count == 0:
            return {}

        def avg(key: str) -> float:
            return sum(sample.get(key, 0.0) for sample in samples) / count

        def mx(key: str) -> float:
            return max(sample.get(key, 0.0) for sample in samples)

        first = samples[0]
        last = samples[-1]

        return {
            "samples": float(count),
            "cpu_percent_avg": round(avg("cpu_percent"), 3),
            "cpu_percent_max": round(mx("cpu_percent"), 3),
            "mem_usage_mb_avg": round(avg("mem_usage_bytes") / _MB, 3),
            "mem_usage_mb_max": round(mx("mem_usage_bytes") / _MB, 3),
            "mem_percent_avg": round(avg("mem_percent"), 3),
            "mem_percent_max": round(mx("mem_percent"), 3),
            "pids_avg": round(avg("pids"), 3),
            "pids_max": round(mx("pids"), 3),
            "net_rx_mb_delta": round(
                (last.get("net_rx_bytes", 0.0) - first.get("net_rx_bytes", 0.0)) / _MB,
                3,
            ),
            "net_tx_mb_delta": round(
                (last.get("net_tx_bytes", 0.0) - first.get("net_tx_bytes", 0.0)) / _MB,
                3,
            ),
            "block_read_mb_delta": round(
                (last.get("block_read_bytes", 0.0) - first.get("block_read_bytes", 0.0)) / _MB,
                3,
            ),
            "block_write_mb_delta": round(
                (last.get("block_write_bytes", 0.0) - first.get("block_write_bytes", 0.0)) / _MB,
                3,
            ),
        }


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
        containers: Sequence[ContainerConfig | Mapping[str, Any]] | None = None,
    ) -> None:
        """Initialize profiler."""
        # Stash core execution context and ensure the results directory exists for JSON dumps.
        self.step = step
        self.results_dir = results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        self.proc = psutil.Process(os.getpid())
        self.rss_samples = []

        self._t0 = None
        self._cpu0 = None
        self.metrics = {}
        # Container metrics are optional; delegate that responsibility to the collector.
        self._container_collector = ContainerStatsCollector(containers)

    def __enter__(self) -> "Profiler":
        """Start profiling context."""
        # Capture starting timestamps and enable tracemalloc for heap measurements.
        self._t0 = time.perf_counter()
        self._cpu0 = time.process_time()
        tracemalloc.start()
        # Kick off container tracking immediately so the first sample aligns with the block start.
        self._container_collector.start()
        self._container_collector.sample()
        return self

    def sample(self) -> None:
        """Sample current RSS memory usage. Can be called multiple times during the profiled block."""
        try:
            # RSS snapshots are stored in bytes and summarised on exit.
            self.rss_samples.append(self.proc.memory_info().rss)
        except Exception:
            logger.info("Warning: failed to sample RSS")
        self._container_collector.sample()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Compute wall/CPU duration and collect peak heap stats.
        wall = time.perf_counter() - self._t0
        cpu = time.process_time() - self._cpu0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Summarise RSS samples (if any) into MB for readability.
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

        self._container_collector.sample()
        # Container stats get appended when available; absent collectors simply return {}.
        container_metrics = self._container_collector.finalize()
        if container_metrics:
            self.metrics["containers"] = container_metrics

        # Persist the metrics to disk and emit a log so they appear in captured output.
        ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        path = self.results_dir / f"step-{self.step}-{ts}.json"

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"{self.metrics}")
