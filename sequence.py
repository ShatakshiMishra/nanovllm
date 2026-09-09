"""Per-request state tracked by the engine."""

from enum import Enum, auto
from typing import List

from .sampling import SamplingParams


class SeqStatus(Enum):
    WAITING = auto()    # queued, no KV blocks allocated
    RUNNING = auto()    # in the active batch, holds KV blocks
    FINISHED = auto()   # done (hit max_tokens / max_model_len)


class Sequence:
    """One generation request and everything the engine needs to run it.

    `num_processed` is the number of leading tokens whose keys/values already
    live in the KV cache. Tokens `[num_processed:]` are the ones that still
    need a forward pass -- on the first step that is the whole prompt (prefill),
    afterwards it is a single freshly-sampled token (decode).
    """

    _next_id = 0

    def __init__(self, prompt_ids: List[int], params: SamplingParams):
        self.seq_id = Sequence._next_id
        Sequence._next_id += 1

        self.token_ids: List[int] = list(prompt_ids)
        self.prompt_len = len(prompt_ids)
        self.params = params

        self.status = SeqStatus.WAITING
        self.num_processed = 0            # tokens with KV committed to cache
        self.block_table: List[int] = []  # physical block ids, in logical order

    # -- lengths -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.token_ids) - self.prompt_len

    @property
    def num_uncached(self) -> int:
        """How many query tokens the next forward pass must process."""
        return len(self.token_ids) - self.num_processed

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    @property
    def output_ids(self) -> List[int]:
        return self.token_ids[self.prompt_len:]

    def is_finished(self, max_model_len: int) -> bool:
        return (self.num_generated >= self.params.max_tokens
                or len(self.token_ids) >= max_model_len)
