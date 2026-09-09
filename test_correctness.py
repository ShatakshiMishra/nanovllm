"""Prove the paged / continuously-batched engine is correct.

The reference is a deliberately dumb generator: at every step it re-runs a
dense, no-cache forward pass over the entire sequence so far. If nano-vLLM's
PagedAttention + KV cache + continuous batching are correct, greedy decoding
must produce token-for-token identical output -- even when many sequences of
different lengths share the batch and even when preemption forces recompute.
"""

import numpy as np

from nanovllm import EngineConfig, LLMEngine, ModelConfig, SamplingParams
from nanovllm.model import GPT, _gelu, _layernorm, _softmax_lastdim


def reference_logits(model: GPT, tokens):
    """Dense forward over a full token list, no cache. Returns last-token logits."""
    cfg = model.cfg
    H, D = cfg.n_head, cfg.head_dim
    T = len(tokens)
    ids = np.asarray(tokens, dtype=np.int64)
    pos = np.arange(T)
    x = model.wte[ids] + model.wpe[pos]

    causal = np.triu(np.ones((T, T), dtype=bool), k=1)   # True above diagonal
    for p in model.layers:
        h = _layernorm(x, p["ln1_g"], p["ln1_b"])
        qkv = h @ p["qkv_w"] + p["qkv_b"]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(T, H, D); k = k.reshape(T, H, D); v = v.reshape(T, H, D)
        scores = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(D)
        scores = np.where(causal[None], -np.inf, scores)
        probs = _softmax_lastdim(scores)
        out = np.einsum("hqk,khd->qhd", probs, v).reshape(T, cfg.n_embd)
        x = x + out @ p["proj_w"] + p["proj_b"]
        h2 = _layernorm(x, p["ln2_g"], p["ln2_b"])
        x = x + (_gelu(h2 @ p["fc_w"] + p["fc_b"]) @ p["fc2_w"] + p["fc2_b"])
    x = _layernorm(x, model.lnf_g, model.lnf_b)
    return x[-1] @ model.wte.T


def reference_greedy(model, prompt, max_tokens):
    tokens = list(prompt)
    for _ in range(max_tokens):
        tokens.append(int(np.argmax(reference_logits(model, tokens))))
    return tokens[len(prompt):]


def main():
    mc = ModelConfig(seed=1)
    # Small KV budget on purpose so this batch triggers preemption/recompute.
    ec = EngineConfig(block_size=8, num_blocks=9, max_num_seqs=4, max_model_len=256)

    prompts = [
        [10, 20, 30],
        [5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17],  # long -> more blocks
        [100],
        [200, 201, 202, 203],
        [1, 2, 3, 4, 5, 6, 7, 8],
    ]
    max_tokens = 20
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)  # greedy

    engine = LLMEngine(mc, ec)
    engine.model = GPT(mc)  # same seed -> identical weights to the reference
    outs = engine.generate(prompts, sp)

    all_ok = True
    for prompt, out in zip(prompts, outs):
        ref = reference_greedy(engine.model, prompt, max_tokens)
        ok = out["output_ids"] == ref
        all_ok &= ok
        print(f"prompt(len={len(prompt):2d})  match={ok}")
        if not ok:
            print("  engine:", out["output_ids"])
            print("  ref   :", ref)

    print(f"\npreemptions during run: {engine.scheduler.preemptions}")
    print("ALL MATCH ✓" if all_ok else "MISMATCH ✗")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
