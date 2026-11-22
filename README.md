# LLM Inference Engine — Educational MVP

A runnable Python prototype implementing **continuous-batching request scheduling**, a **paged KV block metadata allocator**, optional Hugging Face Transformers generation backends, and **length-grouped real tensor batching**.

> **Implementation status:** This repository is an educational prototype, **not** a production GPU inference engine. The metadata allocator is not connected to attention KV tensors. There are no custom CUDA kernels, GPU paged attention, prefix sharing, speculative decoding, or validated Qwen3-8B performance claims in this version.

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

## Project layout

```text
src/inference_engine/
  engine.py       # synchronous batch admission, decode steps, cancel, completion
  backend.py      # deterministic fake backend and optional Transformers model
  kv_cache.py     # standalone paged-block metadata allocator
  incremental.py  # per-request real KV cache incremental decode\n  batched.py      # grouped forwards for same-length requests
examples/
  demo.py         # no-download example
  real_model.py   # original recomputation example
  incremental_model.py # real per-request cached decoding\n  batched_model.py # batched model forward demonstration
tests/
  test_engine.py  # 7 original unit tests
  test_incremental.py # 3 offline lifecycle tests\n  test_batched.py # 5 tensor-level batch/cache tests (requires torch)
```

## Technical details

- `ContinuousBatchEngine` admits pending requests whenever slots become available, including between generation steps.
- Supports per-request output length caps, duplicate ID detection, cancellation and EOS completion.
- `BlockAllocator` reserves and releases integer block IDs in block tables. It does not hold actual GPU tensors or control model memory.
- The synchronous scheduler is not thread-safe and does not expose a production HTTP service.




