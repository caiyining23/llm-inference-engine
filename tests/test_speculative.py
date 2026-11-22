"""Greedy speculative decoding parity tests; no torch or model downloads."""
import unittest
from inference_engine.speculative import greedy_decode, speculative_greedy


class ToyTarget:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def next_token(self, seq):
        return self.fn(seq)

    def verify_block(self, prefix, proposals):
        self.calls += 1
        history = list(prefix)
        results = []
        for proposal in proposals:
            results.append(self.fn(history))
            history.append(proposal)
        results.append(self.fn(history))
        return results


class ToyDraft:
    def __init__(self, fn):
        self.fn = fn

    def next_token(self, seq):
        return self.fn(seq)


class SpeculativeTests(unittest.TestCase):
    def test_perfect_draft(self):
        target = ToyTarget(lambda s: (s[-1] + 1) % 19)
        result = speculative_greedy([1], target, ToyDraft(target.next_token), 11, 3)
        self.assertEqual(result.output_tokens, greedy_decode([1], target, 11))
        self.assertEqual(result.accepted_draft_tokens, 9)
        self.assertEqual(result.verification_rounds, 3)
        self.assertAlmostEqual(result.acceptance_rate, 1.0)

    def test_all_rejected(self):
        target = ToyTarget(lambda s: (s[-1] + 1) % 19)
        result = speculative_greedy([1], target, ToyDraft(lambda s: 0), 8, 4)
        self.assertEqual(result.output_tokens, greedy_decode([1], target, 8))
        self.assertEqual(result.accepted_draft_tokens, 0)

    def test_partial_acceptance(self):
        target = ToyTarget(lambda s: (s[-1] + 1) % 19)
        draft = ToyDraft(lambda s: (s[-1] + (2 if len(s) % 3 == 0 else 1)) % 19)
        result = speculative_greedy([1], target, draft, 17, 5)
        self.assertEqual(result.output_tokens, greedy_decode([1], target, 17))
        self.assertGreater(result.accepted_draft_tokens, 0)

    def test_target_eos(self):
        target = ToyTarget(lambda s: 9 if s[-1] >= 4 else s[-1] + 1)
        result = speculative_greedy([2], target, ToyDraft(lambda s: 17), 12, 4, 9)
        self.assertEqual(result.output_tokens, greedy_decode([2], target, 12, 9))
        self.assertEqual(result.reason, "eos")

    def test_accepted_draft_eos(self):
        target = ToyTarget(lambda s: 9 if s[-1] >= 4 else s[-1] + 1)
        result = speculative_greedy([3], target, ToyDraft(target.next_token), 10, 4, 9)
        self.assertEqual(result.output_tokens, [4, 9])

    def test_max_budget(self):
        target = ToyTarget(lambda s: s[-1] + 1)
        result = speculative_greedy([3], target, ToyDraft(target.next_token), 1, 5)
        self.assertEqual(result.output_tokens, [4])

    def test_zero_budget(self):
        target = ToyTarget(lambda s: 1)
        result = speculative_greedy([3], target, ToyDraft(target.next_token), 0)
        self.assertEqual(result.output_tokens, [])
        self.assertEqual(result.verification_rounds, 0)

    def test_invalid_arguments(self):
        target = ToyTarget(lambda s: 1)
        draft = ToyDraft(target.next_token)
        for prefix, length, lookahead in [([], 2, 3), ([1], -1, 3), ([1], 2, 0)]:
            with self.assertRaises(ValueError):
                speculative_greedy(prefix, target, draft, length, lookahead)

    def test_invalid_verifier(self):
        class Broken(ToyTarget):
            def verify_block(self, prefix, proposals):
                return []
        target = Broken(lambda s: 1)
        with self.assertRaises(ValueError):
            speculative_greedy([1], target, ToyDraft(target.next_token), 5)

    def test_rejected_draft_eos(self):
        target = ToyTarget(lambda s: s[-1] + 1)
        result = speculative_greedy([1], target, ToyDraft(lambda s: 9), 3, 4, 9)
        self.assertEqual(result.output_tokens, [2, 3, 4])

    def test_prefix_sensitive_function(self):
        target = ToyTarget(lambda s: sum(s) % 11)
        draft = ToyDraft(lambda s: (sum(s) + 1) % 11)
        result = speculative_greedy([1, 2], target, draft, 20, 6)
        self.assertEqual(result.output_tokens, greedy_decode([1, 2], target, 20))


if __name__ == "__main__":
    unittest.main()
