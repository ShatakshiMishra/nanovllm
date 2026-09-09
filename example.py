"""Watch continuous batching happen, step by step.

We submit five requests of very different lengths into a small KV budget and
print the batch on every engine step. You'll see sequences join, finish and
leave independently -- and, when memory is tight, get preempted and recomputed.
"""

from nanovllm import EngineConfig, LLMEngine, ModelConfig, SamplingParams
from nanovllm.sequence import SeqStatus


def encode(s: str):
    return list(s.encode("utf-8"))


def main():
    mc = ModelConfig(seed=7)
    ec = EngineConfig(block_size=8, num_blocks=16, max_num_seqs=4, max_model_len=256)
    engine = LLMEngine(mc, ec)

    requests = [
        ("A", "hi",                         6),
        ("B", "the quick brown fox jumps",  24),
        ("C", "x",                          10),
        ("D", "hello world",                14),
        ("E", "lorem ipsum dolor sit amet", 20),
    ]
    labels = {}
    for name, text, n in requests:
        sid = engine.add_request(encode(text), SamplingParams(max_tokens=n, temperature=0.0))
        labels[sid] = name

    print(f"KV budget: {ec.num_blocks} blocks x {ec.block_size} tokens "
          f"= {ec.num_blocks * ec.block_size} token-slots\n")

    header = f"{'step':>4} | {'running batch (seq:len)':<40} | {'KV blocks used':>14}"
    print(header)
    print("-" * len(header))

    step = 0
    while engine.scheduler.has_work():
        # schedule() runs inside step(); peek at the batch it chose by looking
        # at the scheduler's running set right after the step.
        finished = engine.step()
        step += 1
        running = engine.scheduler.running
        cells = [f"{labels[s.seq_id]}:{len(s)}" for s in running]
        used = ec.num_blocks - engine.bm.num_free
        note = ""
        if finished:
            note = "  <- finished: " + ", ".join(labels[s.seq_id] for s in finished)
        print(f"{step:>4} | {' '.join(cells):<40} | {used:>10}/{ec.num_blocks}{note}")

    print(f"\ntotal steps: {step}   preemptions: {engine.scheduler.preemptions}")
    print("\nAll requests completed. (Output text is random -- weights are untrained.)")


if __name__ == "__main__":
    main()
