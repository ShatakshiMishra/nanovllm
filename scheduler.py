"""Continuous-batching scheduler with recompute-based preemption.

Every engine step we call schedule(), which returns the set of sequences to run
*this* step. Unlike static batching, sequences join (from `waiting`) and leave
(when finished) independently, so a 5-token request never waits behind a
500-token one sharing the batch.

Memory pressure is handled by preemption: if a running sequence needs another
KV block and none are free, we evict the most-recently-admitted running
sequence back to `waiting` and free its blocks. When re-admitted it re-prefills
from scratch (the "recompute" policy -- vLLM also supports swapping to CPU).
"""

from collections import deque
from typing import List

from .block_manager import BlockManager
from .config import EngineConfig
from .sequence import Sequence, SeqStatus


class Scheduler:
    def __init__(self, ec: EngineConfig, bm: BlockManager):
        self.ec = ec
        self.bm = bm
        self.waiting: deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self.preemptions = 0     # counter, for demo/telemetry

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _evict(self, victim: Sequence) -> None:
        self.running.remove(victim)
        self.bm.free(victim)
        victim.num_processed = 0                  # recompute from scratch later
        victim.status = SeqStatus.WAITING
        self.waiting.appendleft(victim)           # re-admit ASAP (front of queue)
        self.preemptions += 1

    def _preempt(self, protect: Sequence) -> bool:
        """Evict the newest running sequence other than `protect`.

        Returns True if something was evicted, False if `protect` is the only
        running sequence (caller must then evict `protect` itself).
        """
        for seq in reversed(self.running):        # newest first
            if seq is not protect:
                self._evict(seq)
                return True
        return False

    def schedule(self) -> List[Sequence]:
        # 1) Admit waiting sequences (prefill) while the batch and KV memory allow.
        while self.waiting and len(self.running) < self.ec.max_num_seqs:
            seq = self.waiting[0]
            if not self.bm.can_allocate(len(seq)):
                break
            self.waiting.popleft()
            self.bm.allocate(seq, len(seq))       # blocks for the whole prompt
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)

        # 2) Ensure every running sequence has a block for its next token.
        #    Decodes grow by one token and may cross a block boundary.
        for seq in list(self.running):
            if seq.status is not SeqStatus.RUNNING:
                continue                          # got preempted earlier this round
            need_tokens = seq.num_processed + seq.num_uncached
            while not self.bm.can_allocate(need_tokens):
                if not self._preempt(protect=seq):
                    # `seq` is the only one left and still doesn't fit: evict it
                    # too and let it recompute once memory frees up.
                    self._evict(seq)
                    break
            if seq.status is SeqStatus.RUNNING:
                self.bm.allocate(seq, need_tokens)

        return list(self.running)

    def free_finished(self) -> None:
        """Drop finished sequences from the batch and reclaim their blocks."""
        still = []
        for seq in self.running:
            if seq.status is SeqStatus.FINISHED:
                self.bm.free(seq)
            else:
                still.append(seq)
        self.running = still
