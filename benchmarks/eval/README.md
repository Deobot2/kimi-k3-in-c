# Evaluation harness

Measures what a change to the weights costs, against a baseline of the released ones.

```bash
# 1. baseline: the released bf16 weights. Do this ONCE and keep the directory.
python3 benchmarks/eval/run_eval.py \
    --model /path/k3model --trunk /path/trunk-bf16 \
    --label bf16 --out benchmarks/eval/results/bf16

# 2. a candidate, scored against it
python3 benchmarks/eval/run_eval.py \
    --model /path/k3model --trunk /path/trunk-mxfp4 \
    --label mxfp4 --out benchmarks/eval/results/mxfp4 \
    --baseline benchmarks/eval/results/bf16

# 3. compare finished runs at any time, without re-running anything
python3 benchmarks/eval/run_eval.py --report benchmarks/eval/results/*
```

Add `--quick` first. It truncates each document to 128 ids and generates 16 tokens, which
is enough to see that the wiring works and not enough to conclude anything.

## Why this exists rather than just `--ppl`

An MXFP4 trunk scored **110.05** against the bf16 trunk's **97.40** — a 13% rise, which
reads like a modest tax. The same weights produced generations that collapsed into an
eight-token loop.

Both numbers were right, and the gap between them is structural rather than bad luck.
Perplexity is **teacher-forced**: at every position the model is conditioned on the *real*
text, never on its own output. It is never given the chance to wander, so it cannot report
that it would have. Degeneration is a property of the self-conditioned trajectory and is
invisible to any teacher-forced metric, at any corpus size, by construction.

So the harness runs two legs.

### Leg A — teacher forced

| metric | what it answers |
|---|---|
| perplexity | how surprised the model is by held-out text |
| loss ratio | `ppl(candidate) / ppl(baseline)` — the headline, and the weakest line here |
| **top-1 agreement** | how often the candidate predicts what **the baseline** predicted |
| top-1 correct | how often each model predicts the real next token |
| NLL delta p50/p95/p99 | whether the loss is spread evenly or carried by a few positions |
| mean entropy | whether the distribution is flattening or collapsing |

**Top-1 agreement is the instrument that matters.** A perplexity ratio compares the
candidate against the *text*; agreement compares it against *the model it has to stand in
for*, which is the actual question. And unlike a ratio of two means, it cannot average a
catastrophic minority away.

How much that matters is measurable. Two *unrelated* random-weight models, sharing nothing
but their shape, score:

```
  loss ratio            0.9999x        <- indistinguishable
  top-1 agreement       0.66%          <- chance level is 1/256 = 0.39%
```

The ratio says the models are the same. They have no weights in common. Any candidate that
comes back with a ratio near 1.0 and agreement below ~90% is not a slightly degraded
version of the baseline; it is a different model that happens to be equally surprised.

### Leg B — free running

| metric | what it answers |
|---|---|
| repeat rate | fraction of positions whose preceding 4-gram already occurred |
| **loop period** | the shortest cycle the tail is stuck in, `0` for none |
| distinct-3 | trigram diversity of the continuation |

**The control is what makes this leg mean anything.** Greedy decoding of a base model with
no chat template repeats on its own — that is ordinary behaviour, not damage. So the
baseline is measured on the same prompts with the same settings and what is reported is the
*excess* over it. Without that subtraction this leg would condemn the released weights too.

Two of the six prompts are traps: a numbered list and a markdown table are structures a
healthy model continues *by repeating a pattern*. They separate "repeats because the text
repeats" from "repeats because the model has collapsed".

`loop_period` can only see periods up to `--gen / 3`, because three cycles are what
separate a loop from a model that quoted itself once. At the default `--gen 64` that is a
period of 21. The driver warns when `--gen` is small enough for this to bite.

## The corpus

Eight documents in `corpus/`, spanning the registers where quantization damage shows up
differently: narrative prose, technical exposition, C, Python, dense factual recall,
dialogue, step-by-step arithmetic, and structured/repetitive data. Original text, committed,
no download and no network.

Per-document perplexity is reported as well as the total, because a register-specific
failure — code intact, factual recall destroyed — is exactly the shape that quantization
produces and exactly the shape a single mean hides.

`--max-ids` truncates each document (default 1024). Attention in the MLA layers is quadratic
in document length, so doubling a document more than doubles its cost, while the perplexity
estimate only cares about total *positions* and not how they are grouped. Many short
documents is the cheaper way to buy the same statistical power.

## Cost

Leg A is one teacher-forced sweep per document, all in **one process** — `--ppl-file`
amortizes the model load across the suite, and with a resident trunk it amortizes the trunk
read too. With a streamed trunk each document costs one trunk pass, which is unavoidable.

Leg B costs one full decode step per generated token, so `--gen` dominates the wall clock.
Start small.

## Verifying the harness without a checkpoint

```bash
python3 benchmarks/eval/run_eval.py --self-test     # metrics only, no model
make test                                           # runs the same thing
```

For end-to-end wiring on a machine with no checkpoint:

```bash
python3 tools/make_random_checkpoint.py build/rndck          # numpy only, no torch
python3 benchmarks/eval/run_eval.py --model build/rndck \
    --label a --out /tmp/ev/a --synthetic-vocab 256 --cache-gb 0.05
```

Those weights are **noise**. Perplexity comes back at approximately the vocabulary size and
entropy at exactly `log(vocab)`, which is what confirms the arithmetic is right and is the
only thing such a run can confirm. Running the same checkpoint twice must give exactly
100.00% agreement and a ratio of exactly 1.0000; that self-comparison is the sharpest
available check that the per-position records are aligned.
