import unittest
from inference_engine import ContinuousBatchEngine, Request, FakeBackend, BlockAllocator


class EngineTests(unittest.TestCase):
    def test_continuous_admission_and_completion(self):
        engine = ContinuousBatchEngine(FakeBackend(), 2)
        engine.submit(Request("a", [1], 3))
        engine.step()
        engine.submit(Request("b", [10], 1))
        results = engine.run_until_idle()
        self.assertEqual({r.request_id for r in results}, {"a", "b"})
        self.assertEqual(next(r for r in results if r.request_id == "a").output_tokens, [2, 3, 4])
        self.assertEqual(next(r for r in results if r.request_id == "b").output_tokens, [11])

    def test_capacity_and_pending(self):
        engine = ContinuousBatchEngine(FakeBackend(), 1)
        engine.submit(Request("a", [1], 2))
        engine.submit(Request("b", [1], 1))
        engine.step()
        self.assertEqual(list(engine.active), ["a"])
        self.assertEqual(len(engine.pending), 1)
        self.assertEqual(len(engine.run_until_idle()), 2)

    def test_cancel_and_unique_ids(self):
        engine = ContinuousBatchEngine(FakeBackend())
        engine.submit(Request("a", [1], 2))
        with self.assertRaises(ValueError):
            engine.submit(Request("a", [2]))
        self.assertTrue(engine.cancel("a"))
        self.assertFalse(engine.cancel("missing"))
        self.assertEqual(engine.completed[0].reason, "cancelled")

    def test_backend_shape_validation(self):
        class Bad(FakeBackend):
            def next_tokens(self, sequences):
                return []
        engine = ContinuousBatchEngine(Bad())
        engine.submit(Request("a", [1]))
        with self.assertRaises(RuntimeError):
            engine.step()

    def test_eos(self):
        class Eos(FakeBackend):
            eos_token_id = 99
            def next_tokens(self, sequences):
                return [99] * len(sequences)
        engine = ContinuousBatchEngine(Eos())
        engine.submit(Request("a", [1], 10))
        self.assertEqual(engine.run_until_idle()[0].reason, "eos")


class AllocatorTests(unittest.TestCase):
    def test_capacity_and_cleanup(self):
        allocator = BlockAllocator(3, 4)
        self.assertEqual(len(allocator.reserve("x", 5)), 2)
        self.assertAlmostEqual(allocator.utilization, 2 / 3)
        with self.assertRaises(MemoryError):
            allocator.reserve("y", 8)
        self.assertEqual(len(allocator.free), 1)
        allocator.release("x")
        self.assertEqual(len(allocator.free), 3)

    def test_growth(self):
        allocator = BlockAllocator(3, 4)
        first = allocator.reserve("x", 2)
        self.assertEqual(allocator.reserve("x", 4), first)
        self.assertEqual(len(allocator.reserve("x", 9)), 3)
        allocator.release("x")
        self.assertEqual(allocator.utilization, 0)


if __name__ == "__main__":
    unittest.main()
