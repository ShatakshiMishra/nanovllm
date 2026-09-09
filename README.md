# nanovllm
nano-vLLM: a from-scratch reimplementation of vLLM's inference engine in pure NumPy — no PyTorch, no CUDA, no downloads. Demonstrates PagedAttention (block-based KV cache), continuous batching, and preemption/recompute on a tiny GPT, with a test proving the paged engine is bit-exact against a naive reference.


A tiny, from-scratch reimplementation of [vLLM](https://github.com/vllm-project/vllm)'s
**serving engine** in pure NumPy — no PyTorch, no CUDA, no model download.

It exists to show *how vLLM works*, not to be fast. It implements the three
ideas that make vLLM vLLM, and nothing else:

| Idea | What it does | Where |
|------|--------------|-------|
| **PagedAttention** | KV cache is split into fixed-size **blocks**; each sequence owns a list of block ids (a "block table"), like OS page tables. No contiguous allocation, near-zero fragmentation. | [`block_manager.py`](nanovllm/block_manager.py), attention in [`model.py`](nanovllm/model.py) |
| **Continuous batching** | Every engine step re-picks the batch: finished requests leave, queued requests join. A 6-token request never waits behind a 500-token one. | [`scheduler.py`](nanovllm/scheduler.py) |
| **Preemption / recompute** | When KV memory runs out, evict the newest running sequence, free its blocks, and re-prefill it later. | [`scheduler.py`](nanovllm/scheduler.py) |

The transformer itself ([`model.py`](nanovllm/model.py)) is a small GPT-2-style
decoder with **random, untrained weights**, so generated *text* is gibberish.
That's fine — the point is the engine, and correctness is proven by comparing
against a reference decoder (below).

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install numpy
.venv/bin/python example.py            # watch continuous batching, step by step
.venv/bin/python test_correctness.py   # prove the engine is bit-exact
```

## Why you can trust it: `test_correctness.py`

The reference generator is deliberately dumb: at every step it re-runs a
**dense, no-cache** forward pass over the whole sequence. If the paged KV cache,
the per-sequence attention gather, the continuous batching, *and* the
recompute-after-preemption path are all correct, greedy decoding must produce
**token-for-token identical** output. It does — even with the KV budget cranked
down so tight that the run triggers 40 preemptions:

```
prompt(len= 3)  match=True
prompt(len=12)  match=True
prompt(len= 1)  match=True
prompt(len= 4)  match=True
prompt(len= 8)  match=True

preemptions during run: 40
ALL MATCH ✓
```

## What `example.py` shows

Five requests of different lengths into a 128-slot KV budget with a max batch of
4. Watch request **A** finish and **E** immediately take its place, and watch KV
blocks get freed and reused:

```
step | running batch (seq:len)                  | KV blocks used
----------------------------------------------------------------
   1 | A:3 B:26 C:2 D:12                        |          8/16
   6 | B:31 C:7 D:17                            |          7/16  <- finished: A
   7 | B:32 C:8 D:18 E:27                       |         12/16   (E joins)
  ...
  36 |                                          |          0/16  <- finished: E
```

## Using it as a library

```python
from nanovllm import LLMEngine, ModelConfig, EngineConfig, SamplingParams

engine = LLMEngine(
    ModelConfig(n_layer=4, n_head=4, n_embd=128, vocab_size=256),
    EngineConfig(block_size=16, num_blocks=64, max_num_seqs=8),
)

prompts = [list("hello".encode()), list("world".encode())]  # byte-level ids
outputs = engine.generate(prompts, SamplingParams(max_tokens=16, temperature=0.8, top_p=0.9))
for o in outputs:
    print(o["seq_id"], o["output_ids"])
```

`LLMEngine.step()` runs exactly one forward pass over the current dynamic batch
and returns the sequences that finished on that step — the same `add_request` /
`step` split real vLLM exposes.

## How a step works

```
scheduler.schedule()      # admit waiting seqs, ensure a KV block for each decode,
                          #   preempt the newest seq if memory is full
   -> _build_inputs()     # flatten every query token across the batch into one
                          #   [num_tokens] array + its (block, slot) write location
      -> model.forward()  # ONE matmul for the linear layers over all tokens;
                          #   attention gathers each seq's KV from its blocks
         -> sample()      # greedy / temperature / top-k / top-p per request
            -> commit     # append token, mark KV cached, retire if finished
```

## Deliberately left out (real vLLM has these)

- Fused CUDA PagedAttention kernel — here attention is a per-sequence NumPy loop.
- Prefix caching / copy-on-write block sharing (blocks are single-owner).
- Chunked prefill, CPU/GPU KV swapping, speculative decoding, tensor parallelism.
- A real tokenizer and trained weights.

## Files

```
nanovllm/
  config.py         ModelConfig, EngineConfig
  sequence.py       per-request state (tokens, block table, status)
  block_manager.py  KVCache tensor + paged block allocator
  model.py          GPT with paged-KV attention (the "model runner")
  scheduler.py      continuous batching + preemption
  sampling.py       SamplingParams + greedy/temp/top-k/top-p sampler
  engine.py         LLMEngine: add_request / step / generate
example.py          step-by-step continuous-batching demo
test_correctness.py bit-exact check vs a dense no-cache reference
