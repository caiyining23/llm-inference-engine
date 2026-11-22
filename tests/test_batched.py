"""Tensor-level tests: no model downloads or network required."""
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

from inference_engine.engine import ContinuousBatchEngine, Request
from inference_engine.batched import LengthGroupedTransformersBackend


class Tokenizer:
    eos_token_id = None
    def encode(self, text):
        return [ord(x) % 7 + 1 for x in text]
    def decode(self, ids, **kwargs):
        return ",".join(map(str, ids))


class DeterministicModel:
    def __init__(self):
        self.batch_sizes = []
    def to(self, device):
        return self
    def eval(self):
        return self
    def __call__(self, *, input_ids, attention_mask, past_key_values, use_cache):
        self.batch_sizes.append(input_ids.shape[0])
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        assert attention_mask.shape[1] == past_len + input_ids.shape[1]
        latest = input_ids[:, -1]
        logits = torch.full((len(latest), 1, 32), -100.0)
        for i, token in enumerate(latest.tolist()):
            logits[i, 0, (token + 1) % 32] = 100.0
        old = None if past_key_values is None else past_key_values[0][0]
        new = input_ids[:, None, :, None].float()
        key = new if old is None else torch.cat((old, new), dim=2)
        return SimpleNamespace(logits=logits, past_key_values=((key, key.clone()),))


@unittest.skipUnless(torch is not None, "requires torch")
class LengthGroupTests(unittest.TestCase):
    def make_engine(self, limit=4):
        model = DeterministicModel()
        backend = LengthGroupedTransformersBackend(model=model, tokenizer=Tokenizer(), device="cpu")
        return ContinuousBatchEngine(backend, max_batch_size=limit), backend, model

    def test_two_requests_share_model_forward(self):
        engine, backend, model = self.make_engine()
        engine.submit(Request("a", [1, 2], 3))
        engine.submit(Request("b", [7, 8], 3))
        results = engine.run_until_idle()
        self.assertEqual(model.batch_sizes, [2, 2, 2])
        self.assertEqual(backend.forward_calls, 3)
        self.assertEqual([r.output_tokens for r in results], [[3, 4, 5], [9, 10, 11]])
        self.assertEqual(backend.cache_count, 0)

    def test_mixed_lengths_split_safely(self):
        engine, backend, model = self.make_engine()
        engine.submit(Request("a", [1], 2))
        engine.submit(Request("b", [2, 3], 2))
        engine.run_until_idle()
        self.assertEqual(model.batch_sizes, [1, 1, 1, 1])
        self.assertEqual(backend.cache_count, 0)

    def test_cancel_releases_cache(self):
        engine, backend, model = self.make_engine()
        engine.submit(Request("a", [1], 5))
        engine.step()
        self.assertEqual(backend.cache_count, 1)
        self.assertTrue(engine.cancel("a"))
        self.assertEqual(backend.cache_count, 0)

    def test_new_admission_reuses_available_slots(self):
        engine, backend, model = self.make_engine(limit=2)
        engine.submit(Request("a", [1], 3))
        engine.submit(Request("b", [2], 1))
        engine.step()
        engine.submit(Request("c", [4], 1))
        results = engine.run_until_idle()
        self.assertEqual({r.request_id for r in results}, {"a", "b", "c"})
        self.assertEqual(backend.cache_count, 0)

    def test_reject_history_change(self):
        engine, backend, model = self.make_engine()
        backend.next_tokens_by_id([("x", [1, 2])])
        with self.assertRaises(ValueError):
            backend.next_tokens_by_id([("x", [7, 2, 3])])


if __name__ == "__main__":
    unittest.main()
