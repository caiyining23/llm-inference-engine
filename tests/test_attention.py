"""Attention parity and masking tests; PyTorch CPU, no model downloads."""
import unittest

try:
    import torch
except ImportError:
    torch = None

from inference_engine.attention import (
    reference_attention, sdpa_attention, suffix_causal_mask
)


@unittest.skipUnless(torch is not None, "requires torch")
class AttentionTests(unittest.TestCase):
    def make_tensors(self, query_len, key_len, dtype=None):
        g = torch.Generator().manual_seed(42)
        dtype = dtype or torch.float32
        q = torch.randn((2, 3, query_len, 8), generator=g).to(dtype)
        k = torch.randn((2, 3, key_len, 8), generator=g).to(dtype)
        v = torch.randn((2, 3, key_len, 8), generator=g).to(dtype)
        return q, k, v

    def test_full_prefill_matches_reference(self):
        q, k, v = self.make_tensors(7, 7)
        torch.testing.assert_close(sdpa_attention(q, k, v),
                                   reference_attention(q, k, v), rtol=1e-5, atol=1e-5)

    def test_one_token_decode_matches_reference(self):
        q, k, v = self.make_tensors(1, 19)
        torch.testing.assert_close(sdpa_attention(q, k, v),
                                   reference_attention(q, k, v), rtol=1e-5, atol=1e-5)

    def test_chunked_decode_matches_reference(self):
        q, k, v = self.make_tensors(4, 12)
        torch.testing.assert_close(sdpa_attention(q, k, v),
                                   reference_attention(q, k, v), rtol=1e-5, atol=1e-5)

    def test_suffix_causal_mask(self):
        mask = suffix_causal_mask(2, 5, "cpu").tolist()
        self.assertEqual(mask, [[True, True, True, True, False],
                                [True, True, True, True, True]])

    def test_no_future_token_leakage(self):
        q, k, v = self.make_tensors(3, 5)
        y1 = sdpa_attention(q, k, v)
        v[:, :, -1, :] += 1000
        y2 = sdpa_attention(q, k, v)
        torch.testing.assert_close(y1[:, :, 0, :], y2[:, :, 0, :])
        self.assertFalse(torch.allclose(y1[:, :, -1, :], y2[:, :, -1, :]))

    def test_invalid_shapes(self):
        q, k, v = self.make_tensors(4, 3)
        with self.assertRaises(ValueError):
            sdpa_attention(q, k, v)
        q, k, v = self.make_tensors(2, 4)
        with self.assertRaises(ValueError):
            reference_attention(q, k, v[:, :, :, :4])

    def test_gradient_correctness(self):
        q, k, v = self.make_tensors(2, 4)
        q1 = q.clone().requires_grad_()
        q2 = q.clone().requires_grad_()
        reference_attention(q1, k, v).sum().backward()
        sdpa_attention(q2, k, v).sum().backward()
        torch.testing.assert_close(q1.grad, q2.grad, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
