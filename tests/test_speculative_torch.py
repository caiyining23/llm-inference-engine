"""Verify Hugging Face adapter causal-logit indexing with a tiny mock model."""
import unittest
from types import SimpleNamespace
try:
    import torch
except ImportError:
    torch = None

from inference_engine.speculative import (
    HuggingFaceGreedyAdapter, greedy_decode, speculative_greedy
)


@unittest.skipUnless(torch is not None, "requires PyTorch")
class HfAdapterTests(unittest.TestCase):
    def test_one_target_forward_per_block(self):
        class MockModel:
            def eval(self):
                return self
            def to(self, device):
                return self
            def __call__(self, input_ids, use_cache):
                self_outer.assertFalse(use_cache)
                b, length = input_ids.shape
                logits = torch.full((b, length, 20), -100.0)
                for i, token in enumerate(input_ids[0].tolist()):
                    logits[0, i, (token + 1) % 20] = 100
                return SimpleNamespace(logits=logits)
        self_outer = self
        target = HuggingFaceGreedyAdapter(MockModel(), "cpu")
        draft = HuggingFaceGreedyAdapter(MockModel(), "cpu")
        result = speculative_greedy([1], target, draft, 8, lookahead=3)
        self.assertEqual(result.output_tokens, [2,3,4,5,6,7,8,9])
        self.assertEqual(target.forward_calls, result.verification_rounds)
        self.assertEqual(result.verification_rounds, 2)
        self.assertEqual(draft.forward_calls, result.proposed_draft_tokens)


if __name__ == "__main__":
    unittest.main()
