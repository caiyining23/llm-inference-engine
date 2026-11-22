"""Offline scheduling tests for incremental backend integration."""
import unittest
from inference_engine.engine import ContinuousBatchEngine, Request
from inference_engine.incremental import IncrementalTransformersBackend


class TrackingBackend:
    eos_token_id = None

    def __init__(self):
        self.histories = {}
        self.released = []

    def next_tokens_by_id(self, members):
        result = []
        for request_id, sequence in members:
            self.histories.setdefault(request_id, []).append(sequence.copy())
            result.append(sequence[-1] + 1)
        return result

    def release(self, request_id):
        self.released.append(request_id)


class IncrementalSchedulingTests(unittest.TestCase):
    def test_admission_and_cleanup(self):
        backend = TrackingBackend()
        engine = ContinuousBatchEngine(backend, 2)
        engine.submit(Request("first", [1], 3))
        engine.step()
        engine.submit(Request("second", [5], 2))
        result = engine.run_until_idle()
        self.assertEqual({r.request_id for r in result}, {"first", "second"})
        self.assertEqual(backend.histories["first"], [[1], [1, 2], [1, 2, 3]])
        self.assertEqual(backend.histories["second"], [[5], [5, 6]])
        self.assertEqual(set(backend.released), {"first", "second"})

    def test_cancel_releases_backend(self):
        backend = TrackingBackend()
        engine = ContinuousBatchEngine(backend)
        engine.submit(Request("x", [1], 10))
        engine.step()
        self.assertTrue(engine.cancel("x"))
        self.assertEqual(backend.released, ["x"])
        self.assertEqual(engine.completed[0].reason, "cancelled")

    def test_optional_backend_is_importable_without_model_dependencies(self):
        self.assertTrue(callable(IncrementalTransformersBackend))


if __name__ == "__main__":
    unittest.main()
