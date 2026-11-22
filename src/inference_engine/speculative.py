"""Exact greedy speculative decoding with draft proposals and target block verification.

Standalone reference implementation. No persistent KV cache, sampling support,
or claim of end-to-end speedup. Target verifies k+1 positions in one forward.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeculativeResult:
    output_tokens: list[int]
    accepted_draft_tokens: int
    proposed_draft_tokens: int
    verification_rounds: int
    reason: str

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_draft_tokens / self.proposed_draft_tokens if self.proposed_draft_tokens else 0.0


def greedy_decode(prompt, target, max_new_tokens, eos_token_id=None):
    if not prompt or max_new_tokens < 0:
        raise ValueError("prompt must be nonempty; max_new_tokens >= 0")
    history = list(prompt)
    output = []
    for _ in range(max_new_tokens):
        token = int(target.next_token(history))
        output.append(token)
        history.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
    return output


def speculative_greedy(prompt, target, draft, max_new_tokens, lookahead=4, eos_token_id=None):
    """Produce the exact target greedy output, accepting draft tokens only when matching.

    target.verify_block(prefix, proposals) must return k+1 greedy predictions
    for prefix, prefix+d1, ... prefix+d1+...+dk, in that order.
    """
    if not prompt or max_new_tokens < 0 or lookahead < 1:
        raise ValueError("prompt nonempty; max_new_tokens >= 0; lookahead >= 1")
    history = list(prompt)
    output = []
    accepted = proposed_count = rounds = 0
    reason = "length"
    while len(output) < max_new_tokens:
        proposals = []
        for _ in range(min(lookahead, max_new_tokens - len(output))):
            token = int(draft.next_token(history + proposals))
            proposals.append(token)
            if eos_token_id is not None and token == eos_token_id:
                break
        predictions = target.verify_block(history, proposals)
        if len(predictions) != len(proposals) + 1:
            raise ValueError("target must return k+1 predictions")
        proposed_count += len(proposals)
        rounds += 1
        rejected = False
        for i, token in enumerate(proposals):
            expected = int(predictions[i])
            if token != expected:
                output.append(expected)
                history.append(expected)
                rejected = True
                if eos_token_id is not None and expected == eos_token_id:
                    reason = "eos"
                break
            accepted += 1
            output.append(token)
            history.append(token)
            if eos_token_id is not None and token == eos_token_id:
                reason = "eos"
                break
        if reason == "eos":
            break
        if not rejected and len(output) < max_new_tokens:
            token = int(predictions[len(proposals)])
            output.append(token)
            history.append(token)
            if eos_token_id is not None and token == eos_token_id:
                reason = "eos"
                break
    return SpeculativeResult(output, accepted, proposed_count, rounds, reason)


class HuggingFaceGreedyAdapter:
    """Real causal-LM draft/target adapter with full-context forwards.

    A target verification round takes one model forward on the proposed block.
    Draft still runs autoregressively. No caching across rounds.
    """
    def __init__(self, model, device="cpu"):
        import torch
        self.torch = torch
        self.model = model.eval().to(device)
        self.device = device
        self.forward_calls = 0

    def _logits(self, history):
        ids = self.torch.tensor([history], dtype=self.torch.long, device=self.device)
        with self.torch.inference_mode():
            result = self.model(input_ids=ids, use_cache=False)
        self.forward_calls += 1
        return result.logits[0]

    def next_token(self, history):
        return int(self._logits(history)[-1].argmax().item())

    def verify_block(self, prefix, proposals):
        if not prefix or not proposals:
            raise ValueError("prefix and proposals must be nonempty")
        logits = self._logits(prefix + proposals)
        start = len(prefix) - 1
        return [int(x) for x in logits[start:start + len(proposals) + 1].argmax(-1).tolist()]
