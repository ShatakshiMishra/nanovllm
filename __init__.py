"""nano-vLLM: a tiny, from-scratch reimplementation of vLLM's serving engine.

Implements the three ideas that make vLLM fast, in pure NumPy:
  * PagedAttention   -- KV cache stored in fixed-size blocks, addressed via
                        per-sequence block tables (nanovllm/block_manager.py)
  * Continuous batch -- a scheduler that admits and retires sequences every
                        engine step, so short requests don't wait for long
                        ones (nanovllm/scheduler.py)
  * Preemption       -- when KV memory runs out, evict a running sequence and
                        recompute it later (nanovllm/scheduler.py)

The transformer (nanovllm/model.py) uses small random weights so the package
runs with no model download. Output text is therefore gibberish -- the point
is the *engine*, and test_correctness.py proves the paged/batched path yields
token-for-token identical results to a naive full-recompute reference.
"""

from .config import ModelConfig, EngineConfig
from .sampling import SamplingParams
from .engine import LLMEngine

__all__ = ["ModelConfig", "EngineConfig", "SamplingParams", "LLMEngine"]
