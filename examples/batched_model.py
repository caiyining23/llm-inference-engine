"""Batched real-model example (downloads small GPT-2 weights)."""
from inference_engine.engine import ContinuousBatchEngine, Request
from inference_engine.batched import LengthGroupedTransformersBackend

backend = LengthGroupedTransformersBackend("sshleifer/tiny-gpt2")
engine = ContinuousBatchEngine(backend, max_batch_size=4)
# Equal-length prompts share a batched prefill and subsequent decode forwards.
engine.submit(Request("a", backend.encode("Hello world"), max_new_tokens=8))
engine.submit(Request("b", backend.encode("Great day"), max_new_tokens=8))
for result in engine.run_until_idle():
    print(result.request_id, backend.decode(result.output_tokens))
print("Model forward calls:", backend.forward_calls)
print("Unreleased request caches:", backend.cache_count)
