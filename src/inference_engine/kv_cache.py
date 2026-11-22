"""Metadata-only paged block allocator; not integrated with model KV tensors."""
from dataclasses import dataclass, field


@dataclass
class BlockAllocator:
    total_blocks: int
    block_size: int = 16
    free: set[int] = field(init=False)
    tables: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self):
        if self.total_blocks <= 0 or self.block_size <= 0:
            raise ValueError("total_blocks and block_size must be positive")
        self.free = set(range(self.total_blocks))

    def reserve(self, request_id: str, token_count: int) -> list[int]:
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        existing = self.tables.get(request_id, [])
        required = (token_count + self.block_size - 1) // self.block_size
        additional = max(0, required - len(existing))
        if additional > len(self.free):
            raise MemoryError("KV metadata block pool exhausted")
        new = sorted(self.free)[:additional]
        self.free.difference_update(new)
        self.tables[request_id] = existing + new
        return list(self.tables[request_id])

    def release(self, request_id: str) -> None:
        self.free.update(self.tables.pop(request_id, []))

    @property
    def utilization(self) -> float:
        return 1.0 - len(self.free) / self.total_blocks
