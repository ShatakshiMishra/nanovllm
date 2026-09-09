"""Sampling parameters and the token sampler (greedy / temperature / top-k / top-p)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SamplingParams:
    max_tokens: int = 32
    temperature: float = 1.0      # 0.0 => greedy (argmax)
    top_k: int = 0                # 0 => disabled
    top_p: float = 1.0            # 1.0 => disabled
    seed: int = 0                 # per-request RNG seed for reproducibility

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


def sample(logits: np.ndarray, params: SamplingParams, rng: np.random.Generator) -> int:
    """Pick one token id from a 1-D logits vector according to `params`."""
    if params.greedy:
        return int(np.argmax(logits))

    logits = logits.astype(np.float64) / max(params.temperature, 1e-6)

    if params.top_k and params.top_k < logits.shape[0]:
        # Keep only the top_k largest logits; mask the rest to -inf.
        kth = np.partition(logits, -params.top_k)[-params.top_k]
        logits = np.where(logits < kth, -np.inf, logits)

    # Softmax (numerically stable).
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()

    if params.top_p < 1.0:
        # Nucleus: keep the smallest set of tokens whose cumulative prob >= top_p.
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        cutoff = np.searchsorted(cumulative, params.top_p) + 1
        keep = order[:cutoff]
        mask = np.zeros_like(probs)
        mask[keep] = probs[keep]
        probs = mask / mask.sum()

    return int(rng.choice(probs.shape[0], p=probs))
