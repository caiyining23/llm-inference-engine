from inference_engine import ContinuousBatchEngine, FakeBackend, Request

engine = ContinuousBatchEngine(FakeBackend(), max_batch_size=2)
engine.submit(Request("alice", [1, 2], max_new_tokens=3))
engine.submit(Request("bob", [8], max_new_tokens=2))
engine.step()
engine.submit(Request("carol", [50], max_new_tokens=2))
for result in engine.run_until_idle():
    print(result)
