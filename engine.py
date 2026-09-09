"""The serving engine: turns a queue of requests into generated tokens.

One `step()` = one transformer forward over a dynamic batch:
    schedule -> build flattened inputs -> forward -> sample -> commit tokens.

`generate()` just drives `step()` until every request is finished, mirroring
vLLM's `LLMEngine.step()` / `LLM.generate()` split.
"""

from typing import Dict, List

import numpy as np

from .block_manager import BlockManager, KVCache
from .config import EngineConfig, ModelConfig
from .model import GPT, SeqRun
from .sampling import SamplingParams, sample
from .scheduler import Scheduler
from .sequence import Sequence, SeqStatus


class LLMEngine:
    def __init__(self, model_config: ModelConfig = None, engine_config: EngineConfig = None):
        self.mc = model_config or ModelConfig()
        self.ec = engine_config or EngineConfig()
        # Keep the two length caps consistent.
        self.ec.max_model_len = min(self.ec.max_model_len, self.mc.max_model_len)

        self.model = GPT(self.mc)
        self.kv = KVCache(self.mc, self.ec)
        self.bm = BlockManager(self.ec)
        self.scheduler = Scheduler(self.ec, self.bm)
        self._rngs: Dict[int, np.random.Generator] = {}

    # -- public API ----------------------------------------------------------
    def add_request(self, prompt_ids: List[int], params: SamplingParams = None) -> int:
        params = params or SamplingParams()
        if len(prompt_ids) >= self.ec.max_model_len:
            raise ValueError("prompt longer than max_model_len")
        seq = Sequence(prompt_ids, params)
        self._rngs[seq.seq_id] = np.random.default_rng(params.seed)
        self.scheduler.add(seq)
        return seq.seq_id

    def generate(self, prompts: List[List[int]], params=None):
        """Convenience: add many prompts, run to completion, return outputs.

        Returns a list of dicts (in request order) with the generated token ids.
        """
        if params is None:
            params = [SamplingParams()] * len(prompts)
        elif isinstance(params, SamplingParams):
            params = [params] * len(prompts)

        order = [self.add_request(p, sp) for p, sp in zip(prompts, params)]
        done: Dict[int, Sequence] = {}
        while self.scheduler.has_work():
            for seq in self.step():
                done[seq.seq_id] = seq
        return [{"seq_id": sid, "output_ids": done[sid].output_ids} for sid in order]

    # -- one engine step -----------------------------------------------------
    def step(self) -> List[Sequence]:
        seqs = self.scheduler.schedule()
        if not seqs:
            return []

        input_ids, positions, write_block, write_slot, runs = self._build_inputs(seqs)
        logits = self.model.forward(
            np.asarray(input_ids, dtype=np.int64),
            np.asarray(positions, dtype=np.int64),
            runs, self.kv, self.bm,
            np.asarray(write_block, dtype=np.int64),
            np.asarray(write_slot, dtype=np.int64),
        )

        finished: List[Sequence] = []
        for seq, row in zip(seqs, logits):
            token = sample(row, seq.params, self._rngs[seq.seq_id])
            # every query token in this step is now cached
            seq.num_processed = len(seq.token_ids)
            seq.append_token(token)
            if seq.is_finished(self.ec.max_model_len):
                seq.status = SeqStatus.FINISHED
                finished.append(seq)

        self.scheduler.free_finished()
        for seq in finished:
            self._rngs.pop(seq.seq_id, None)
        return finished

    def _build_inputs(self, seqs: List[Sequence]):
        """Flatten the batch into per-token arrays the model can consume."""
        input_ids, positions, write_block, write_slot, runs = [], [], [], [], []
        for seq in seqs:
            start = seq.num_processed
            q_len = seq.num_uncached            # prefill: whole prompt; decode: 1
            ctx_len = len(seq.token_ids)
            for pos in range(start, ctx_len):
                input_ids.append(seq.token_ids[pos])
                positions.append(pos)
                block, slot = self.bm.slot_of(seq, pos)
                write_block.append(block)
                write_slot.append(slot)
            runs.append(SeqRun(q_len=q_len, ctx_len=ctx_len,
                               block_table=list(seq.block_table)))
        return input_ids, positions, write_block, write_slot, runs
