"""Physical KV-cache blocks and the allocator behind PagedAttention.

The KV cache is one big pre-allocated tensor split into `num_blocks` blocks of
`block_size` token-slots each. A sequence never owns a contiguous range: it
owns a *list* of block ids (its "block table"), exactly like page tables in an
OS. Logical token position `p` for a sequence lives at:

    physical block = block_table[p // block_size]
    slot in block  = p % block_size

This is what lets vLLM pack many variable-length sequences into memory with
almost no fragmentation, and what makes eviction cheap (just free the blocks).
"""

import math
from typing import List

import numpy as np

from .config import EngineConfig, ModelConfig
from .sequence import Sequence


class KVCache:
    """The physical key/value storage for every layer."""

    def __init__(self, mc: ModelConfig, ec: EngineConfig):
        self.block_size = ec.block_size
        shape = (mc.n_layer, ec.num_blocks, ec.block_size, mc.n_head, mc.head_dim)
        self.k = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)


class BlockManager:
    """Hands out and reclaims physical blocks; owns per-sequence block tables."""

    def __init__(self, ec: EngineConfig):
        self.block_size = ec.block_size
        self.num_blocks = ec.num_blocks
        # A simple free list. Real vLLM uses a ref-counted pool to share blocks
        # across sequences (prefix caching); we keep one owner per block.
        self.free_blocks: List[int] = list(range(ec.num_blocks))

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    def _blocks_for(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size)

    def can_allocate(self, num_tokens: int) -> bool:
        return self.num_free >= self._blocks_for(num_tokens)

    def allocate(self, seq: Sequence, num_tokens: int) -> None:
        """Grow `seq`'s block table so it can hold `num_tokens` tokens total."""
        need = self._blocks_for(num_tokens) - len(seq.block_table)
        assert need <= self.num_free, "allocate() called without capacity"
        for _ in range(max(0, need)):
            seq.block_table.append(self.free_blocks.pop())

    def free(self, seq: Sequence) -> None:
        """Return all of a sequence's blocks to the pool."""
        self.free_blocks.extend(seq.block_table)
        seq.block_table = []

    # -- helpers used by the model runner -----------------------------------
    def slot_of(self, seq: Sequence, position: int):
        """(physical_block, slot) where token `position` of `seq` is stored."""
        block = seq.block_table[position // self.block_size]
        slot = position % self.block_size
        return block, slot

    def gather_indices(self, seq: Sequence, ctx_len: int):
        """Vectorised (block_ids, slot_ids) for positions 0..ctx_len-1.

        Used to pull a sequence's scattered KV blocks into a contiguous tensor
        for the attention matmul.
        """
        positions = np.arange(ctx_len)
        table = np.asarray(seq.block_table, dtype=np.int64)
        blocks = table[positions // self.block_size]
        slots = positions % self.block_size
        return blocks, slots
