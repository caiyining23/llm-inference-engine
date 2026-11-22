"""Fixed-size torch tensor storage for legacy attention KV caches.

Provides block tables and tensor storage, but attention reconstructs a
contiguous KV tuple before a normal Transformers forward. Not paged attention.
"""
from .kv_cache import BlockAllocator


class TensorPagedKVStore:
    """Store per-request [1, heads, tokens, head_dim] KV tensors in fixed blocks.

    Lazily allocates a torch pool on the first write. A request's full KV
    snapshot may be replaced after a model step. Caller owns synchronization.
    """

    def __init__(self, total_blocks: int, block_size: int = 16):
        self.allocator = BlockAllocator(total_blocks, block_size)
        self.keys = None
        self.values = None
        self._spec = None
        self._lengths = {}

    @property
    def utilization(self):
        return self.allocator.utilization

    @property
    def active_requests(self):
        return len(self._lengths)

    def _validate(self, cache):
        import torch
        if not isinstance(cache, (tuple, list)) or not cache:
            raise ValueError("expected a nonempty tuple of (key, value) layers")
        first = cache[0][0]
        if first.ndim != 4 or first.shape[0] != 1 or first.shape[2] == 0:
            raise ValueError("KV tensors must be [1, heads, positive_tokens, dim]")
        length = first.shape[2]
        spec = (len(cache), first.shape[1], first.shape[3], first.dtype, first.device)
        for layer in cache:
            if len(layer) != 2:
                raise ValueError("each layer needs (key, value)")
            for tensor in layer:
                if not isinstance(tensor, torch.Tensor) or tensor.shape != first.shape:
                    raise ValueError("all KV tensors must have the same shape")
                if tensor.device != first.device or tensor.dtype != first.dtype:
                    raise ValueError("all KV tensors must have same dtype and device")
        if self._spec is not None and spec != self._spec:
            raise ValueError("KV tensor specification changed")
        return spec, length

    def write(self, request_id: str, cache):
        """Replace a request's full KV snapshot, reserving blocks first."""
        import torch
        spec, length = self._validate(cache)
        if request_id in self._lengths and length < self._lengths[request_id]:
            raise ValueError("KV sequence length must be nondecreasing")
        table = self.allocator.reserve(request_id, length)
        if self._spec is None:
            layers, heads, dim, dtype, device = spec
            shape = (layers, self.allocator.total_blocks, heads,
                     self.allocator.block_size, dim)
            self.keys = torch.empty(shape, dtype=dtype, device=device)
            self.values = torch.empty(shape, dtype=dtype, device=device)
            self._spec = spec
        with torch.no_grad():
            for index, block_id in enumerate(table):
                start = index * self.allocator.block_size
                end = min(start + self.allocator.block_size, length)
                width = end - start
                for layer, (key, value) in enumerate(cache):
                    self.keys[layer, block_id, :, :width, :].copy_(key[0, :, start:end, :])
                    self.values[layer, block_id, :, :width, :].copy_(value[0, :, start:end, :])
        self._lengths[request_id] = length

    def read(self, request_id: str):
        """Reassemble ordinary contiguous KV tuples (not zero-copy attention)."""
        import torch
        if request_id not in self._lengths:
            raise KeyError(request_id)
        length = self._lengths[request_id]
        table = self.allocator.tables[request_id]
        layers = len(self.keys)
        result = []
        for layer in range(layers):
            tensors = []
            for pool in (self.keys, self.values):
                slices = []
                for index, block_id in enumerate(table):
                    start = index * self.allocator.block_size
                    width = min(self.allocator.block_size, length - start)
                    slices.append(pool[layer, block_id, :, :width, :])
                tensors.append(torch.cat(slices, dim=1).unsqueeze(0))
            result.append(tuple(tensors))
        return tuple(result)

    def release(self, request_id: str):
        self._lengths.pop(request_id, None)
        self.allocator.release(request_id)

    def contains(self, request_id: str):
        return request_id in self._lengths
