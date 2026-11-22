# LLM Inference Engine — Educational MVP

A runnable Python prototype implementing **continuous-batching request scheduling**, a **paged KV block allocator with real tensor storage**, optional Hugging Face Transformers generation backends, **length-grouped real tensor batching**, and attention correctness/benchmark baselines.

> **Implementation status:** This repository is an educational prototype, **not** a production GPU inference engine. The optional tensor block store is connected to the length-grouped backend, but reconstructs contiguous KV caches for ordinary attention. There are no custom CUDA kernels, GPU paged attention, prefix sharing, speculative decoding, or validated Qwen3-8B performance claims in this version.

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/caiyining23/llm-inference-engine.git
cd llm-inference-engine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/demo.py
python -m unittest discover -s tests -v
```

The demo uses a deterministic fake token generator and runs without a GPU.

### Optional real language model

This example downloads a tiny causal language model from Hugging Face and can run on CPU (slowly):

```bash
python -m pip install -e ".[model]"
python examples/real_model.py
```

The Transformers backend **recomputes all input tokens at each step** instead of using an optimized incremental KV cache.

### Incremental model backend (new)

A per-request `IncrementalTransformersBackend` now uses real Hugging Face `past_key_values`: a full-prompt prefill occurs once, and later steps feed only the newest token while retaining the attention cache. Caches are released when requests complete or are cancelled.

```bash
python -m pip install -e ".[model]"
python examples/incremental_model.py
```

**Limitations:** This is per-request incremental decoding, **not** true GPU batched forward execution. Dynamic request admission occurs at the scheduler level. The original `TransformersBackend` still uses full-context recomputation. The real-model example requires downloading model weights and has not been validated with a GPU in this environment.

### Length-grouped batched model forwards (v0.2)

The `LengthGroupedTransformersBackend` combines multiple requests into a **single model forward** when their prefill lengths or processed KV-cache lengths are equal. It concatenates their actual model KV tensors along the batch dimension, decodes, then splits the resulting caches back into per-request tensors. Requests of **different** lengths are executed in separate groups: this is **not** arbitrary-length continuous batching or paged attention.

```bash
python -m pip install -e ".[model]"
python examples/batched_model.py
```

The optional dependency constrains Transformers to `>=4.45,<4.50` for the legacy tuple KV-cache interface. The offline deterministic PyTorch tests validate tensor grouping and cache lifecycles; **real-model generation and GPU performance need environment-specific validation**.

Install PyTorch to run tensor tests (without torch those tests are skipped):

```bash
python -m unittest discover -s tests -v
```

### Tensor-backed paged KV blocks (v0.3)

`TensorPagedKVStore` now holds **real PyTorch K/V tensors** in bounded, fixed-size blocks and tracks per-request block tables. The length-grouped backend accepts `kv_store=TensorPagedKVStore(total_blocks=..., block_size=...)`, reads contiguous cache tuples from the block store for each model forward, and writes updated caches back. Completed and cancelled requests release their block reservations.

```python
from inference_engine import TensorPagedKVStore
from inference_engine.batched import LengthGroupedTransformersBackend

store = TensorPagedKVStore(total_blocks=256, block_size=16)
backend = LengthGroupedTransformersBackend(kv_store=store)
```

Run CPU tests without model downloads:

```bash
python -m pip install -e .
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
```

**Important limitations:** This is tensor-backed **block storage**, not vLLM-style paged attention. The backend still reconstructs *contiguous copies* of cached K/V before each ordinary model forward, so the block store may use **more** memory and compute than direct caching. Stores have fixed capacity and are not thread safe. NVIDIA CUDA correctness/performance and real model behavior are still unverified.

### Attention correctness and benchmarking (v0.4)

`src/inference_engine/attention.py` provides two independent Attention implementations:

- `reference_attention`: explicit FP32 score accumulation + masked softmax + value aggregation.
- `sdpa_attention`: PyTorch's `scaled_dot_product_attention`, with an explicit **bottom-right suffix-aligned causal mask** so incremental one-token and chunked decoding have correct causal semantics.

The implementations are compared in `tests/test_attention.py` on CPU for prefill, one-token decode, chunked decode, gradient parity, and future-token leakage.

Run a reproducible **attention-only** microbenchmark:

```bash
python -m pip install -e .
python -m pip install torch
python benchmarks/attention.py --device cpu --iterations 30
# Only if an NVIDIA CUDA GPU is installed:
python benchmarks/attention.py --device cuda --iterations 100 --warmup 20 --output attention-results.json
```

The benchmark synchronizes CUDA kernels and reports median/p95 per-call latencies in JSON alongside PyTorch version, device, dtype and shape. **These numbers are not model tokens/sec, and do not establish Qwen3-8B or custom-CUDA performance gains.** The reference is educational and is not guaranteed to be faster than PyTorch SDPA.

## Project layout

```text
src/inference_engine/
  engine.py       # synchronous batch admission, decode steps, cancel, completion
  backend.py      # deterministic fake backend and optional Transformers model
  kv_cache.py     # block ID allocator
  paged_tensors.py # bounded tensor-backed K/V block pool
  attention.py    # reference attention + torch SDPA parity
  incremental.py  # per-request real KV cache incremental decode
  batched.py      # grouped forwards for same-length requests
examples/
  demo.py         # no-download example
  real_model.py   # original recomputation example
  incremental_model.py # real per-request cached decoding
  batched_model.py # batched model forward demonstration
tests/
  test_engine.py  # 7 original unit tests
  test_incremental.py # 3 offline lifecycle tests
  test_batched.py # 5 tensor-level batch/cache tests (requires torch)
  test_paged_tensors.py # 6 block-storage and integration tests
  test_attention.py # 7 attention correctness/gradient tests
benchmarks/
  attention.py  # CPU/CUDA attention microbenchmark
```

## Technical details

- `ContinuousBatchEngine` admits pending requests whenever slots become available, including between generation steps.
- Supports per-request output length caps, duplicate ID detection, cancellation and EOS completion.
- `BlockAllocator` reserves and releases integer IDs; `TensorPagedKVStore` stores actual K/V tensors on its configured torch device, but does not implement paged attention.
- The synchronous scheduler is not thread-safe and does not expose a production HTTP service.




