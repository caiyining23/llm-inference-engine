"""Reproducible attention microbenchmark (CPU or CUDA, no model downloads).

Examples:
  python benchmarks/attention.py --device cpu --iterations 30
  python benchmarks/attention.py --device cuda --iterations 100 --warmup 20

Results are microkernel timings only, NOT LLM tokens/sec or end-to-end latency.
"""
import argparse
import json
import platform
import statistics
import time

import torch

from inference_engine.attention import reference_attention, sdpa_attention


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_kernel(fn, q, k, v, *, warmup, iterations):
    with torch.inference_mode():
        for _ in range(warmup):
            fn(q, k, v)
        synchronize(q.device)
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn(q, k, v)
            synchronize(q.device)
            samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": samples[min(len(samples) - 1, int(0.95 * len(samples)))],
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--query-length", type=int, default=1)
    parser.add_argument("--kv-length", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("iterations must be positive; warmup must be nonnegative")
    if args.batch < 1 or args.heads < 1 or args.head_dim < 1:
        parser.error("batch, heads and head-dim must be positive")
    if args.query_length < 1 or args.kv_length < args.query_length:
        parser.error("kv-length must be >= query-length >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but NVIDIA CUDA device is unavailable")

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.manual_seed(42)
    shape = (args.batch, args.heads)
    q = torch.randn((*shape, args.query_length, args.head_dim), device=device, dtype=dtype)
    k = torch.randn((*shape, args.kv_length, args.head_dim), device=device, dtype=dtype)
    v = torch.randn((*shape, args.kv_length, args.head_dim), device=device, dtype=dtype)
    with torch.inference_mode():
        expected = reference_attention(q, k, v)
        actual = sdpa_attention(q, k, v)
        torch.testing.assert_close(actual, expected,
                                   rtol=1e-2 if dtype == torch.float16 else 1e-4,
                                   atol=1e-2 if dtype == torch.float16 else 1e-4)
    report = {
        "benchmark": "attention_microbenchmark",
        "not_end_to_end_llm": True,
        "pytorch": torch.__version__,
        "python_platform": platform.platform(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": str(dtype),
        "shape": vars(args),
        "reference": time_kernel(reference_attention, q, k, v,
                                 warmup=args.warmup, iterations=args.iterations),
        "sdpa": time_kernel(sdpa_attention, q, k, v,
                            warmup=args.warmup, iterations=args.iterations),
    }
    report["reference_to_sdpa_median_ratio"] = (
        report["reference"]["median_ms"] / report["sdpa"]["median_ms"])
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
