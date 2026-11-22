"""Token generation backends.

FakeBackend enables deterministic CPU-only tests. TransformersBackend loads a
real causal language model; this is not a custom CUDA or paged-attention engine.
"""
from typing import Protocol


class TokenBackend(Protocol):
    eos_token_id: int | None
    def next_tokens(self, sequences: list[list[int]]) -> list[int]: ...


class FakeBackend:
    eos_token_id = None

    def next_tokens(self, sequences: list[list[int]]) -> list[int]:
        return [(seq[-1] + 1) % 256 if seq else 1 for seq in sequences]


class TransformersBackend:
    """Optional real model; recomputes context every decoding step."""
    def __init__(self, model_name: str = "sshleifer/tiny-gpt2", device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text)
        return ids or [self.tokenizer.eos_token_id]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def next_tokens(self, sequences: list[list[int]]) -> list[int]:
        if not sequences:
            return []
        torch = self.torch
        width = max(map(len, sequences))
        pad = self.tokenizer.pad_token_id
        ids = [[pad] * (width - len(s)) + s for s in sequences]
        mask = [[0] * (width - len(s)) + [1] * len(s) for s in sequences]
        with torch.inference_mode():
            out = self.model(input_ids=torch.tensor(ids, device=self.device),
                             attention_mask=torch.tensor(mask, device=self.device),
                             use_cache=False)
        return out.logits[:, -1, :].argmax(dim=-1).cpu().tolist()
