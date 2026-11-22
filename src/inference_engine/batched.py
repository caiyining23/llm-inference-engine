"""Batched model forwards for requests with matching KV lengths.

The backend groups equally long prefill prompts or equally long cached decode
sequences into a common forward. Heterogeneous lengths use separate forwards.
It uses legacy Hugging Face KV tuples; no paged attention or custom CUDA.
"""
from collections import defaultdict


class LengthGroupedTransformersBackend:
    """Real batched tensor forwards within groups of identical context length."""

    def __init__(self, model_name="sshleifer/tiny-gpt2", device=None, *, model=None, tokenizer=None, kv_store=None):
        import torch
        self.torch = torch
        if model is None or tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.eos_token_id = tokenizer.eos_token_id
        self._states = {}
        self.kv_store = kv_store
        self.forward_calls = 0

    def encode(self, text):
        ids = self.tokenizer.encode(text)
        if ids:
            return ids
        if self.eos_token_id is None:
            raise ValueError("empty prompt and no fallback EOS token")
        return [self.eos_token_id]

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def cache_count(self):
        return len(self._states)

    def release(self, request_id):
        self._states.pop(request_id, None)
        if self.kv_store is not None:
            self.kv_store.release(request_id)

    @staticmethod
    def _legacy_cache(cache):
        if hasattr(cache, "to_legacy_cache"):
            cache = cache.to_legacy_cache()
        if not isinstance(cache, (list, tuple)):
            raise TypeError("Expected legacy tuple past_key_values; use compatible Transformers version")
        return tuple(tuple(kv for kv in layer) for layer in cache)

    def next_tokens_by_id(self, members):
        if len({request_id for request_id, _ in members}) != len(members):
            raise ValueError("duplicate request IDs in batch")
        groups = defaultdict(list)
        for i, (request_id, seq) in enumerate(members):
            if not seq:
                raise ValueError("empty token sequence")
            state = self._states.get(request_id)
            if state is not None and (len(seq) != state["length"] + 1 or
                                      seq[:-1] != state["history"]):
                raise ValueError("input history does not match cached request")
            key = ("prefill", len(seq)) if state is None else ("decode", state["length"])
            groups[key].append((i, request_id, seq, state))

        results = [None] * len(members)
        for (phase, _), batch in groups.items():
            if phase == "prefill":
                token_rows = [seq for _, _, seq, _ in batch]
                past = None
            else:
                token_rows = [[seq[-1]] for _, _, seq, _ in batch]
                pasts = [self.kv_store.read(request_id) if self.kv_store is not None
                         else state["past"] for _, request_id, _, state in batch]
                past = tuple(
                    tuple(self.torch.cat([item[layer][part] for item in pasts], dim=0)
                          for part in range(2))
                    for layer in range(len(pasts[0]))
                )
            ids = self.torch.tensor(token_rows, dtype=self.torch.long, device=self.device)
            mask = self.torch.ones((len(batch), len(batch[0][2])), dtype=self.torch.long, device=self.device)
            with self.torch.inference_mode():
                output = self.model(input_ids=ids, attention_mask=mask,
                                    past_key_values=past, use_cache=True)
            self.forward_calls += 1
            cache = self._legacy_cache(output.past_key_values)
            tokens = output.logits[:, -1, :].argmax(dim=-1).tolist()
            if len(tokens) != len(batch):
                raise RuntimeError("model output batch size mismatch")
            for row, (i, request_id, seq, _) in enumerate(batch):
                results[i] = int(tokens[row])
                single_cache = tuple(tuple(kv[row:row + 1].contiguous() for kv in layer)
                                     for layer in cache)
                if self.kv_store is not None:
                    self.kv_store.write(request_id, single_cache)
                self._states[request_id] = {
                    "history": list(seq), "length": len(seq),
                    "past": None if self.kv_store is not None else single_cache,
                }
        return results
