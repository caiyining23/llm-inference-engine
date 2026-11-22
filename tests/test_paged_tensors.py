"""CPU-only torch KV block tests. No model downloads needed."""
import unittest
try:
    import torch
except ImportError:
    torch = None
from inference_engine.paged_tensors import TensorPagedKVStore
from inference_engine.engine import ContinuousBatchEngine, Request


def cache(values, layers=2):
    seq = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1, 1)
    return tuple((seq + i, seq + 100 + i) for i in range(layers))


@unittest.skipUnless(torch is not None, "requires torch")
class PagedTensorTests(unittest.TestCase):
    def test_multiblock_roundtrip(self):
        store = TensorPagedKVStore(5, 2)
        expected = cache([3, 5, 7, 9, 11])
        store.write("a", expected)
        for original, restored in zip(expected, store.read("a")):
            for x, y in zip(original, restored):
                torch.testing.assert_close(x, y)
        self.assertAlmostEqual(store.utilization, 3 / 5)

    def test_release_reuses_blocks_without_cross_request_corruption(self):
        store = TensorPagedKVStore(3, 2)
        store.write("a", cache([1, 2]))
        store.write("b", cache([8, 9, 10]))
        with self.assertRaises(MemoryError):
            store.write("c", cache([42]))
        store.release("b")
        store.write("c", cache([42, 43, 44]))
        self.assertEqual(store.read("a")[0][0].flatten().tolist(), [1, 2])
        self.assertEqual(store.read("c")[0][0].flatten().tolist(), [42, 43, 44])
        store.release("a")
        store.release("c")
        self.assertEqual(store.utilization, 0)
        self.assertEqual(store.active_requests, 0)

    def test_cache_growth(self):
        store = TensorPagedKVStore(3, 2)
        store.write("a", cache([1]))
        store.write("a", cache([1, 2, 3]))
        self.assertEqual(len(store.allocator.tables["a"]), 2)
        self.assertEqual(store.read("a")[0][0].flatten().tolist(), [1, 2, 3])

    def test_failed_resize_preserves_old_cache(self):
        store = TensorPagedKVStore(1, 2)
        store.write("a", cache([1, 2]))
        with self.assertRaises(MemoryError):
            store.write("a", cache([1, 2, 3]))
        self.assertEqual(store.read("a")[0][0].flatten().tolist(), [1, 2])

    def test_invalid_cache_rejected(self):
        store = TensorPagedKVStore(3, 2)
        store.write("a", cache([1, 2, 3]))
        with self.assertRaises(ValueError):
            store.write("a", cache([1]))
        with self.assertRaises(ValueError):
            store.write("b", ((torch.ones(1, 2), torch.ones(1, 2)),))

    def test_backend_end_to_end_cleanup(self):
        from test_batched import DeterministicModel, Tokenizer
        from inference_engine.batched import LengthGroupedTransformersBackend
        store = TensorPagedKVStore(8, 2)
        model = DeterministicModel()
        backend = LengthGroupedTransformersBackend(
            model=model, tokenizer=Tokenizer(), device="cpu", kv_store=store)
        engine = ContinuousBatchEngine(backend, max_batch_size=2)
        engine.submit(Request("a", [1, 2], 3))
        engine.submit(Request("b", [5, 6], 3))
        results = engine.run_until_idle()
        self.assertEqual([r.output_tokens for r in results], [[3, 4, 5], [7, 8, 9]])
        self.assertEqual(model.batch_sizes, [2, 2, 2])
        self.assertEqual(store.active_requests, 0)
        self.assertEqual(store.utilization, 0)


if __name__ == "__main__":
    unittest.main()
