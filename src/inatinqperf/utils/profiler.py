import os, time, json, psutil, tracemalloc, datetime

class Profiler:
    """
    Lightweight in-process profiler:
      - wall_time_sec, cpu_time_sec
      - Python heap peak (tracemalloc)
      - rss_avg_mb, rss_max_mb (process RSS snapshots)

    For CPU flamegraphs, run the command via py-spy externally.
    """
    def __init__(self, step: str, results_dir: str = ".results"):
        self.step = step
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.proc = psutil.Process(os.getpid())
        self.rss_samples = []

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._cpu0 = time.process_time()
        tracemalloc.start()
        return self

    def sample(self):
        try:
            self.rss_samples.append(self.proc.memory_info().rss)
        except Exception:
            pass

    def __exit__(self, exc_type, exc, tb):
        wall = time.perf_counter() - self._t0
        cpu = time.process_time() - self._cpu0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rss_avg = (sum(self.rss_samples)/len(self.rss_samples)/(1024*1024)) if self.rss_samples else 0.0
        rss_max = (max(self.rss_samples)/(1024*1024)) if self.rss_samples else 0.0

        self.metrics = {
            "step": self.step,
            "wall_time_sec": round(wall, 4),
            "cpu_time_sec": round(cpu, 4),
            "py_heap_peak_mb": round(peak/(1024*1024), 3),
            "rss_avg_mb": round(rss_avg, 3),
            "rss_max_mb": round(rss_max, 3),
            "profiler": "builtin",
        }
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(self.results_dir, f"step-{self.step}-{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"[PROFILE] {self.metrics}")