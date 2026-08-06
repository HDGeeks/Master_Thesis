"""Small helpers for measuring runtime, throughput, and memory during a
local model run. Kept separate from the experiment scripts so the same
instrumentation can be reused by any local runner (hf, cpp, ...), not
just experiment_qwen3.5_hf_1.py.
"""

import resource
import statistics
import subprocess
import sys
import time


def peak_ram_mb():
    """Peak resident memory used by this process so far, in MB.

    ru_maxrss is reported in different units depending on the OS:
    kilobytes on Linux, bytes on macOS.
    """

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    return peak / 1024


def peak_gpu_mb():
    """Peak GPU memory allocated by PyTorch specifically, in MB. 0 if
    torch isn't installed, has no GPU, or the backend doesn't use torch's
    allocator at all (e.g. llama-cpp-python manages its own CUDA memory,
    so this will read 0 for that backend, use gpu_memory_used_mb() instead
    for backends that don't go through torch)."""

    try:
        import torch
    except ImportError:
        return 0

    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0


def gpu_memory_used_mb():
    """Current GPU memory in use, in MB, read directly from nvidia-smi.

    Works regardless of which library actually allocated it (torch,
    llama.cpp, ...), unlike peak_gpu_mb(). Downsides: it's a snapshot at
    call time, not a tracked peak, and on a shared GPU it includes other
    processes' usage too, not just this one. 0 if nvidia-smi isn't
    available (no GPU, or not on this machine)."""

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(output.strip().splitlines()[0])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0


class RunMetrics:
    """Collects per-call timing/token metrics over a run, and prints a
    summary at the end, same idea as the print_summary() helper the
    experiment scripts already use for the matching categories."""

    def __init__(self):
        self.model_load_seconds = None
        self.call_seconds = []
        self.output_token_counts = []
        self.truncated_count = 0
        self._run_start = None

    def start_run(self):
        self._run_start = time.time()

    def record_load(self, seconds):
        self.model_load_seconds = seconds

    def record_call(self, seconds, output_tokens, truncated):
        self.call_seconds.append(seconds)
        self.output_token_counts.append(output_tokens)
        if truncated:
            self.truncated_count += 1

    def as_dict(self):
        total_runtime = time.time() - self._run_start
        num_calls = len(self.call_seconds)
        total_output_tokens = sum(self.output_token_counts)
        total_call_time = sum(self.call_seconds)

        return {
            "model_load_seconds": self.model_load_seconds,
            "total_runtime_seconds": total_runtime,
            "num_calls": num_calls,
            "avg_latency_seconds": total_call_time / num_calls if num_calls else None,
            "median_latency_seconds": statistics.median(self.call_seconds) if num_calls else None,
            "tokens_per_second": total_output_tokens / total_call_time if total_call_time > 0 else None,
            "truncated_count": self.truncated_count,
            "truncated_rate": self.truncated_count / num_calls if num_calls else None,
            "peak_ram_mb": peak_ram_mb(),
            "peak_gpu_mb": peak_gpu_mb(),
            "gpu_memory_used_mb": gpu_memory_used_mb(),
        }

    def print_summary(self):
        stats = self.as_dict()

        print("\n--- Run metrics ---")
        print(f"Model load time: {stats['model_load_seconds']:.1f}s")
        print(f"Total runtime: {stats['total_runtime_seconds']:.1f}s "
              f"({stats['total_runtime_seconds'] / 60:.1f} min)")
        print(f"Documents processed: {stats['num_calls']}")
        if stats["avg_latency_seconds"] is not None:
            print(f"Avg latency per call: {stats['avg_latency_seconds']:.2f}s")
            print(f"Median latency per call: {stats['median_latency_seconds']:.2f}s")
        if stats["tokens_per_second"] is not None:
            print(f"Throughput: {stats['tokens_per_second']:.1f} tokens/sec")
        if stats["truncated_rate"] is not None:
            print(f"Truncated answers (hit max_new_tokens): "
                  f"{stats['truncated_count']} ({stats['truncated_rate']:.1%})")
        print(f"Peak RAM: {stats['peak_ram_mb']:.0f} MB")
        print(f"Peak GPU memory (torch-tracked): {stats['peak_gpu_mb']:.0f} MB")
        print(f"GPU memory in use (nvidia-smi, end of run): {stats['gpu_memory_used_mb']:.0f} MB")
