"""Run a speculative decoding example without model weights or GPU."""
from inference_engine.speculative import speculative_greedy, greedy_decode


class ToyGreedyModel:
    def __init__(self, offset=1):
        self.offset = offset
    def next_token(self, tokens):
        return (tokens[-1] + self.offset) % 100
    def verify_block(self, prefix, proposed):
        seq = list(prefix)
        predicted = []
        for token in proposed:
            predicted.append(self.next_token(seq))
            seq.append(token)
        predicted.append(self.next_token(seq))
        return predicted


target = ToyGreedyModel(offset=1)
draft = ToyGreedyModel(offset=1)
baseline = greedy_decode([1], target, max_new_tokens=16)
result = speculative_greedy([1], target, draft, max_new_tokens=16, lookahead=4)
print("Greedy baseline:", baseline)
print("Speculative:", result.output_tokens)
print("Identical:", baseline == result.output_tokens)
print("Draft acceptance rate:", round(result.acceptance_rate, 3))
print("Target verification rounds:", result.verification_rounds)
print("This counts verification rounds, NOT measured latency or speedup.")
