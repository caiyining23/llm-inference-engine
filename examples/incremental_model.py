"""Optional real incremental generation; install with pip install -e '.[model]'."""
from inference_engine.engine import ContinuousBatchEngine, Request
from inference_engine.incremental import IncrementalTransformersBackend

backend = IncrementalTransformersBackend("sshleifer/tiny-gpt2")
engine = ContinuousBatchEngine(backend, max_batch_size=2)
engine.submit(Request("example", backend.encode("The weather is"), max_new_tokens=8))
result = engine.run_until_idle()[0]
print(backend.decode(result.output_tokens))
print("Remaining cached model states:", backend.cache_count)
