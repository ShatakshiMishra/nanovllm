"""A tiny decoder-only transformer whose attention reads/writes a paged KV cache.

Design mirrors a real vLLM "model runner":
  * Linear layers (embedding, QKV, MLP, LM head) run over ALL query tokens in
    the step, flattened into one [num_tokens, n_embd] matrix -- one big matmul,
    regardless of how many sequences are in the batch or how long each is.
  * Attention is computed per sequence, gathering that sequence's keys/values
    out of its (possibly scattered) KV blocks. This is the CPU/NumPy stand-in
    for vLLM's fused PagedAttention CUDA kernel.

Weights are small and random (seeded), so this runs instantly with no download.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from .block_manager import BlockManager, KVCache
from .config import ModelConfig


@dataclass
class SeqRun:
    """What one sequence contributes to a forward pass."""
    q_len: int        # number of query tokens this step (prefill: prompt len; decode: 1)
    ctx_len: int      # total tokens whose KV is in cache after this step
    block_table: list # physical block ids for gathering context KV


def _layernorm(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def _softmax_lastdim(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


class GPT:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        C, L = cfg.n_embd, cfg.n_layer

        def randn(*shape, scale):
            return (rng.standard_normal(shape) * scale).astype(np.float32)

        self.wte = randn(cfg.vocab_size, C, scale=0.02)          # token embedding
        self.wpe = randn(cfg.max_model_len, C, scale=0.02)       # positional embedding
        self.layers = []
        for _ in range(L):
            self.layers.append({
                "ln1_g": np.ones(C, np.float32), "ln1_b": np.zeros(C, np.float32),
                "qkv_w": randn(C, 3 * C, scale=0.02), "qkv_b": np.zeros(3 * C, np.float32),
                "proj_w": randn(C, C, scale=0.02), "proj_b": np.zeros(C, np.float32),
                "ln2_g": np.ones(C, np.float32), "ln2_b": np.zeros(C, np.float32),
                "fc_w": randn(C, 4 * C, scale=0.02), "fc_b": np.zeros(4 * C, np.float32),
                "fc2_w": randn(4 * C, C, scale=0.02), "fc2_b": np.zeros(C, np.float32),
            })
        self.lnf_g = np.ones(C, np.float32)
        self.lnf_b = np.zeros(C, np.float32)
        # LM head is tied to the token embedding, as in GPT-2.

    def forward(self, input_ids, positions, runs: List[SeqRun],
                kv: KVCache, bm: BlockManager,
                write_block, write_slot) -> np.ndarray:
        """Run one engine step.

        Args:
            input_ids:  [T] flattened query-token ids across all sequences.
            positions:  [T] logical position of each query token in its sequence.
            runs:       one SeqRun per sequence, in the same order tokens appear.
            write_block/write_slot: [T] where to store each token's K/V.

        Returns:
            logits [num_seqs, vocab] -- only for the LAST query token of each
            sequence (the position we sample the next token from).
        """
        cfg = self.cfg
        H, D = cfg.n_head, cfg.head_dim
        T = input_ids.shape[0]

        x = self.wte[input_ids] + self.wpe[positions]            # [T, C]

        for li, p in enumerate(self.layers):
            h = _layernorm(x, p["ln1_g"], p["ln1_b"])
            qkv = h @ p["qkv_w"] + p["qkv_b"]                    # [T, 3C]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(T, H, D)
            k = k.reshape(T, H, D)
            v = v.reshape(T, H, D)

            # --- write this step's K/V into the paged cache -----------------
            kv.k[li, write_block, write_slot] = k
            kv.v[li, write_block, write_slot] = v

            # --- attention, one sequence at a time --------------------------
            attn = np.empty((T, H, D), dtype=np.float32)
            t0 = 0
            for run in runs:
                ql, cl = run.q_len, run.ctx_len
                qs = q[t0:t0 + ql]                              # [ql, H, D]

                # Gather full context K/V for this sequence from its blocks.
                positions_ctx = np.arange(cl)
                table = np.asarray(run.block_table, dtype=np.int64)
                blk = table[positions_ctx // bm.block_size]
                slt = positions_ctx % bm.block_size
                Kc = kv.k[li, blk, slt]                         # [cl, H, D]
                Vc = kv.v[li, blk, slt]

                # scores[h, i, j] = q_i . k_j  over head h
                scores = np.einsum("qhd,khd->hqk", qs, Kc) / np.sqrt(D)

                # Causal mask: query i (absolute pos = cl-ql+i) sees keys 0..pos.
                q_abs = np.arange(cl - ql, cl)[:, None]         # [ql, 1]
                k_abs = np.arange(cl)[None, :]                  # [1, cl]
                mask = k_abs > q_abs                            # [ql, cl] True = block
                scores = np.where(mask[None], -np.inf, scores)

                probs = _softmax_lastdim(scores)               # [H, ql, cl]
                out = np.einsum("hqk,khd->qhd", probs, Vc)      # [ql, H, D]
                attn[t0:t0 + ql] = out
                t0 += ql

            attn = attn.reshape(T, cfg.n_embd)
            x = x + attn @ p["proj_w"] + p["proj_b"]            # attn residual

            h2 = _layernorm(x, p["ln2_g"], p["ln2_b"])
            mlp = _gelu(h2 @ p["fc_w"] + p["fc_b"]) @ p["fc2_w"] + p["fc2_b"]
            x = x + mlp                                         # MLP residual

        x = _layernorm(x, self.lnf_g, self.lnf_b)

        # Keep only the last query token of each sequence, then project to vocab.
        last = np.cumsum([r.q_len for r in runs]) - 1          # [num_seqs]
        logits = x[last] @ self.wte.T                          # [num_seqs, vocab]
        return logits
