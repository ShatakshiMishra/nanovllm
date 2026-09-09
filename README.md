# nanovllm
nano-vLLM: a from-scratch reimplementation of vLLM's inference engine in pure NumPy — no PyTorch, no CUDA, no downloads. Demonstrates PagedAttention (block-based KV cache), continuous batching, and preemption/recompute on a tiny GPT, with a test proving the paged engine is bit-exact against a naive reference.
