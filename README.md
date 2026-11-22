# LLM Inference Engine — Educational MVP

A tested, runnable Python prototype implementing **continuous-batching request scheduling**, a **paged KV block metadata allocator**, and an optional Hugging Face Transformers generation backend.

> **Implementation status:** This repository is an educational prototype, **not** a production GPU inference engine. The metadata allocator is not connected to attention KV tensors. There are no custom CUDA kernels, GPU paged attention, prefix sharing, speculative decoding, or validated Qwen3-8B performance claims in this version.

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/caiyining23/lm-inference-engine.git
cd lm-inference-engine
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

## Project layout

```text
src/inference_engine/
  engine.py       # synchronous batch admission, decode steps, cancel, completion
  backend.py      # deterministic fake backend and optional Transformers model
  kv_cache.py     # standalone paged-block metadata allocator
examples/
  demo.py         # no-download example
  real_model.py   # optional Hugging Face example
tests/
  test_engine.py  # 7 unit tests
```

## Technical details

- `ContinuousBatchEngine` admits pending requests whenever slots become available, including between generation steps.
- Supports per-request output length caps, duplicate ID detection, cancellation and EOS completion.
- `BlockAllocator` reserves and releases integer block IDs in block tables. It does not hold actual GPU tensors or control model memory.
- The synchronous scheduler is not thread-safe and does not expose a production HTTP service.

## Future work

- Build real incremental prefill/decode with per-request cached model states.
- Integrate KV-block allocation with actual attention memory.
- Add streaming responses, async admission and backpressure.
- Benchmark on specified hardware with reproducible tokens/sec, P50/P95 latency and memory results.
- Explore Qwen3-8B and speculative decoding after validating an actual GPU serving backend.

## Resume/portfolio accuracy

This repository **does not substantiate** the claimed throughput, latency or CUDA optimization figures that may appear in a separate resume. Only include features and measurements after implementing and reproducing them.

## License

MIT.
