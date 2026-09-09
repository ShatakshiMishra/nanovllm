"""Configuration objects for the model and the serving engine."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Hyper-parameters of the toy decoder-only transformer.

    Defaults describe a ~1M-parameter byte-level model: vocab is 256 (raw
    bytes) so any text encodes without a tokenizer download.
    """

    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    max_model_len: int = 512      # size of the positional embedding table
    seed: int = 0                 # deterministic weight initialisation

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0, "n_embd must divide by n_head"
        return self.n_embd // self.n_head


@dataclass
class EngineConfig:
    """Serving-time knobs -- these control the scheduler and KV memory.

    In real vLLM these are `--max-num-seqs`, `--block-size`, and the number of
    KV blocks derived from `--gpu-memory-utilization`. Here we set the block
    count directly so it is easy to make the cache small enough to force
    preemption in a demo.
    """

    block_size: int = 16          # KV-cache tokens per physical block
    num_blocks: int = 64          # total physical KV blocks (the memory budget)
    max_num_seqs: int = 8         # max sequences running concurrently (batch)
    max_model_len: int = 512      # hard cap on prompt+generated length
