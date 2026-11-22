"""Optional real-model example; downloads tiny model weights from Hugging Face."""
from inference_engine import ContinuousBatchEngine, TransformersBackend, Request

backend = TransformersBackend("sshleifer/tiny-gpt2")
engine = ContinuousBatchEngine(backend, max_batch_size=2)
engine.submit(Request("example", backend.encode("The weather is"), max_new_tokens=8))
result = engine.run_until_idle()[0]
print(backend.decode(result.output_tokens))
