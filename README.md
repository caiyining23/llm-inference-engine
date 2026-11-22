# LLM Inference Engine

A modular LLM inference framework built with **Python and PyTorch**, featuring continuous request scheduling, transformer KV-cache management, batched generation, attention benchmarking, and greedy speculative decoding.

## Features

- **Continuous request scheduling:** FIFO admission, configurable batch size, per-request generation limits, cancellation, and completion tracking.
- **Incremental generation:** Transformer decoding with `past_key_values` and explicit cache lifecycle management.
- **Batched inference:** One model forward pass per group of requests sharing the same sequence length; per-request cache tensors are merged and separated along the batch dimension.
- **Block-based KV storage:** A bounded PyTorch K/V tensor pool with configurable block size, per-request block tables, allocation, and reclamation.
- **Attention implementations:** An explicit causal attention reference and a PyTorch scaled-dot-product attention implementation using suffix-aligned masks for incremental decoding.
- **Speculative decoding:** Draft-token proposals, batched target verification, acceptance/rejection, and greedy-output consistency.
- **Testing and profiling:** Deterministic unit tests, tensor-level integration tests, continuous integration, and CPU/CUDA attention microbenchmarks.

## Architecture

```text
                       Generation Requests
                                |
                      ContinuousBatchEngine
                       /        |         \
                Admission   Scheduling   Cancellation
                       \        |         /
                         Model Backend
                   /         |           \
              Incremental  Length-grouped  Greedy speculative
                decoding     batching         decoding
                                |
                          KV Tensor Store
                          Block Allocator
                                |
                       PyTorch Attention
                       Reference / SDPA
```

The scheduler and generation backends are designed as separate components. The tensor store uses fixed-size blocks and reconstructs ordinary contiguous K/V tensors for model forwards; attention execution uses PyTorch operations.

## Requirements

- Python 3.10+
- PyTorch for tensor operations and attention tests
- Hugging Face Transformers for model-backed generation

## Installation

```bash
git clone https://github.com/caiyining23/llm-inference-engine.git
cd llm-inference-engine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For model-backed generation:

```bash
python -m pip install -e ".[model]"
```

## Usage

### Request scheduling

```bash
python examples/demo.py
```

### Transformer decoding

```bash
python examples/real_model.py
python examples/incremental_model.py
python examples/batched_model.py
```

The model examples use a small Hugging Face causal language model. The incremental backend reuses per-request `past_key_values`; the length-grouped backend merges requests with compatible cached sequence lengths.

### KV block storage

```python
from inference_engine.paged_tensors import TensorPagedKVStore
from inference_engine.batched import LengthGroupedTransformersBackend

store = TensorPagedKVStore(total_blocks=256, block_size=16)
backend = LengthGroupedTransformersBackend(kv_store=store)
```

The block store manages actual PyTorch Key/Value tensors and releases request allocations when decoding completes or is cancelled.

### Greedy speculative decoding

```bash
python examples/speculative_demo.py
```

The speculative decoder checks draft proposals against target-model greedy predictions, accepts matching tokens and applies target corrections on mismatch. `HuggingFaceGreedyAdapter` supports full-context block verification with causal language models.

## Tests

Run the core unit tests:

```bash
python -m unittest discover -s tests -v
```

To include tensor-level tests, install PyTorch first:

```bash
python -m pip install torch
python -m unittest discover -s tests -v
```

GitHub Actions runs standard unit tests and CPU PyTorch tests on pushes and pull requests.

Tests cover request admission, cancellation, KV allocation and reclamation, tensor round trips, batched generation, attention parity, causal masking, gradients, and speculative greedy-output consistency.

## Attention Benchmark

```bash
python benchmarks/attention.py --device cpu --iterations 30
```

On an NVIDIA CUDA environment:

```bash
python benchmarks/attention.py --device cuda --warmup 20 --iterations 100 --output attention-results.json
```

The benchmark reports reference attention and PyTorch SDPA execution timings, including median and P95 latency, plus device, dtype, framework version, and input dimensions. These results represent **attention microbenchmarks**, not end-to-end model tokens/second.

## Project Structure

```text
src/inference_engine/
  engine.py             Request lifecycle and continuous scheduler
  backend.py            Fake and Transformers model backends
  incremental.py        Cached autoregressive decoding
  batched.py            Length-grouped batched generation
  kv_cache.py           Block allocation metadata
  paged_tensors.py      Fixed-block PyTorch KV tensor storage
  attention.py          Reference attention and PyTorch SDPA
  speculative.py        Greedy speculative decoding
examples/               Runnable generation examples
tests/                  Unit and tensor integration tests
benchmarks/             Attention timing and output reports
.github/workflows/      Continuous integration
```

## License

MIT.
