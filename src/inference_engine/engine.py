"""Synchronous continuous-batching step scheduler, not a network service."""
from collections import deque
from dataclasses import dataclass, field
from .backend import TokenBackend


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt_tokens: list[int]
    max_new_tokens: int = 16

    def __post_init__(self):
        if not self.request_id or not self.prompt_tokens or self.max_new_tokens < 1:
            raise ValueError("request needs ID, prompt tokens, and positive output budget")


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    output_tokens: list[int]
    reason: str


@dataclass
class _Active:
    request: Request
    tokens: list[int] = field(default_factory=list)


class ContinuousBatchEngine:
    def __init__(self, backend: TokenBackend, max_batch_size: int = 8):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.pending: deque[Request] = deque()
        self.active: dict[str, _Active] = {}
        self.completed: list[GenerationResult] = []
        self._ids: set[str] = set()
        self.steps = 0

    def submit(self, request: Request) -> None:
        if request.request_id in self._ids:
            raise ValueError(f"duplicate request id: {request.request_id}")
        self._ids.add(request.request_id)
        self.pending.append(request)

    def _release_backend(self, request_id: str) -> None:
        release = getattr(self.backend, "release", None)
        if callable(release):
            release(request_id)

    def cancel(self, request_id: str) -> bool:
        active = self.active.pop(request_id, None)
        if active is not None:
            self._release_backend(request_id)
            self.completed.append(GenerationResult(request_id, active.tokens.copy(), "cancelled"))
            return True
        for req in self.pending:
            if req.request_id == request_id:
                self.pending.remove(req)
                self._release_backend(request_id)
                self.completed.append(GenerationResult(request_id, [], "cancelled"))
                return True
        return False

    def step(self) -> list[GenerationResult]:
        while self.pending and len(self.active) < self.max_batch_size:
            req = self.pending.popleft()
            self.active[req.request_id] = _Active(req)
        if not self.active:
            return []
        members = list(self.active.items())
        sequences = [state.request.prompt_tokens + state.tokens for _, state in members]
        by_id = getattr(self.backend, "next_tokens_by_id", None)
        if callable(by_id):
            generated = by_id([
                (request_id, sequence)
                for (request_id, _), sequence in zip(members, sequences)
            ])
        else:
            generated = self.backend.next_tokens(sequences)
        if len(generated) != len(members):
            raise RuntimeError("backend returned incorrect batch size")
        finished = []
        for (request_id, state), token in zip(members, generated):
            state.tokens.append(int(token))
            reason = None
            if token == self.backend.eos_token_id:
                reason = "eos"
            elif len(state.tokens) >= state.request.max_new_tokens:
                reason = "length"
            if reason:
                finished.append(GenerationResult(request_id, state.tokens.copy(), reason))
                del self.active[request_id]
                self._release_backend(request_id)
        self.completed.extend(finished)
        self.steps += 1
        return finished

    def run_until_idle(self, max_steps: int = 10000) -> list[GenerationResult]:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        for _ in range(max_steps):
            if not self.active and not self.pending:
                return list(self.completed)
            self.step()
        raise TimeoutError("engine did not finish within max_steps")
