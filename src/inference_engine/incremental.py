"""Per-request Hugging Face incremental decoding with real past_key_values.

Each request has its own cache and executes a separate model forward pass.
This is NOT GPU continuous batching or paged attention.
"""


class IncrementalTransformersBackend:
    def __init__(self, model_name="sshleifer/tiny-gpt2", device=None):
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
        self._states = {}

    def encode(self, prompt):
        ids = self.tokenizer.encode(prompt)
        if not ids:
            if self.eos_token_id is None:
                raise ValueError("empty prompt without EOS fallback")
            return [self.eos_token_id]
        return ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def release(self, request_id):
        self._states.pop(request_id, None)

    @property
    def cache_count(self):
        return len(self._states)

    def next_tokens_by_id(self, members):
        tokens = []
        for request_id, sequence in members:
            if not sequence:
                raise ValueError("empty sequence")
            state = self._states.get(request_id)
            if state is None:
                model_input = sequence
                past = None
            else:
                if len(sequence) != state["length"] + 1:
                    raise ValueError("token history changed or advanced by more than one token")
                model_input = [sequence[-1]]
                past = state["past"]
            input_ids = self.torch.tensor([model_input], dtype=self.torch.long, device=self.device)
            attention_mask = self.torch.ones((1, len(sequence)), dtype=self.torch.long, device=self.device)
            with self.torch.inference_mode():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                    past_key_values=past, use_cache=True)
            tokens.append(int(output.logits[:, -1, :].argmax(dim=-1).item()))
            self._states[request_id] = {"past": output.past_key_values, "length": len(sequence)}
        return tokens
