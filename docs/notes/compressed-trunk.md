# Trunk size: two attempts, two endings

The trunk is 108.81 GB of bf16 and the engine reads it once per token. Two independent
attempts were made to shrink it. They reached opposite conclusions, and both are recorded
here because the reasons are the useful part.

| | attempt | outcome | origin |
|---|---|---|---|
| **[Part 1](#part-1--lossless-compression-investigated-shelved)** | Lossless Huffman compression | **Shelved.** Correct, but decodes too slowly to help the machines it was built for. | upstream v1.0.0 |
| **[Part 2](#part-2--mxfp4-quantisation-shipped-behind-a-flag)** | MXFP4 quantisation | **Shipped behind a flag.** Works, but costs accuracy and voids every bit-identity gate. | **this fork** |

> **Sent here by the quantised-trunk banner, by `--ppl`, or from `include/k3/k3.h`,
> `src/io/k3_trunk.h` or `tools/mxfp4_trunk.py`?**
> You want **Part 2**. The procedure is there. Part 1 is a different, shelved idea that
> this file happens to be named after.

---

# Part 1 — Lossless compression: investigated, shelved

> **Verdict: shelved. Never shipped.**
> The codec is correct and round-trips byte-exactly at 1.45x. What killed it was decode
> *speed*, not compression ratio.
> *Origin: upstream, v1.0.0. Untouched by this fork.*

## The idea

The trunk is bf16. A bf16 value's high byte (sign + exponent) carries about 2.8 bits of
real information on this checkpoint: roughly a dozen byte values cover 99.9% of all
weights. So entropy-code just that byte plane and store the mantissa byte raw, since it
measures as noise.

That shrinks the packed trunk to about 67% of its size. Because the decoded bytes are the
checkpoint's own bytes, it is lossless and the engine's output is unchanged. At a streamed
budget, fewer trunk bytes is fewer seconds per token.

## What was built, and what it proved

A canonical length-limited Huffman codec: a Python encoder and a decode-only C header.

- Round-trips **byte-exactly**, Python-encoded and C-decoded.
- **1.45x** compression (ratio 0.689) on a 4 MB sample of real trunk bytes.
- The format, the Kraft check, and the canonical code assignment all work.

The idea was sound and the implementation was correct. That was never the problem.

## The measurement that killed it

Decode speed. A hand-rolled scalar Huffman decoder reached only **~0.31 GB/s per core**
after two optimisation passes.

| decoder | measured | cores needed to feed a 3 GB/s laptop SSD |
|---|---|---|
| this scalar Huffman decoder | 0.31 GB/s per core | ~10 |
| FSE/Huff0, on the same exponent-plane bytes | 1724 MB/s | 2 |

Ten cores is not a laptop. So compression built this way **helps many-core machines, which
have the RAM to not need it, and starves the laptops that do.** That is backwards from the
goal.

A fast entropy decoder does exist — FSE/Huff0, above — but even at that speed it needs two
dedicated cores to match a laptop SSD, and vendoring it is about two thousand lines against
a project whose identity is a 176 KB binary with no dependencies. The payoff did not justify
the weight, for the machines that matter.

## What would unlock it

A decoder in the **1 to 1.5 GB/s per core** range that stays small. The standard routes are
four interleaved bitstreams per stripe for instruction-level parallelism, or a double-symbol
table. Either would let the encoder stay exactly as it is.

Until a decoder that fast is in hand, the streamed-trunk speed lever is pinning (RAM-first
`--preset auto`) and the resident int8 draft, not compression.

## The prototype code

Kept next to this note, deliberately not wired into the build:

| file | what it is |
|---|---|
| `huf_encode.py.shelved` | the pack-time encoder |
| `k3_huf.h.shelved` | the decode-only C header |

The format round-trips today, so these are a correct starting point for anyone who revisits
it.

---

# Part 2 — MXFP4 quantisation: shipped, behind a flag

> **Verdict: shipped, opt-in, and it is a fork of the contract rather than a setting.**
> Trunk goes from 108.81 GB to ~29 GB, which makes it fully resident on a 64 GB desktop.
> The price is +13.0% perplexity on a first measurement that is itself too small to quote,
> and the loss of every bit-identity gate in the repository.
> *Origin: **this fork**, 2026-08-12, commits `2a9f9b5`, `60f6184`, `961419b`.*

## Why a quantisation section lives in a compression note

Part 1 is about **lossless** compression, and its conclusion stands: entropy coding helps
the machines that do not need it. This part is the other axis — **quantising** the trunk —
which the project had declined outright, and which now exists in this fork as an opt-in
container.

The two share a goal (a smaller trunk) and nothing else. They are filed together because
this is where the trunk-size reasoning lives.

## What it is

`tools/mxfp4_trunk.py` rewrites a packed trunk with every matmul weight in OCP MX FP4 — the
same format the routed experts already ship in, read by the same kernel.

```
python3 tools/mxfp4_trunk.py /path/to/trunk /path/to/trunk-mx4 --sample-error
k3 <shards> --trunk /path/to/trunk-mx4 --ids ... --incremental
```

What passes through **untouched**: norms, biases, `A_log`, `dt_bias`, the conv kernels, and
the AttnRes projections. They are read elementwise as fp32 and are well under 1% of the
bytes. The router gate passes through too, because `k3_router` walks it with its own inline
fp32 matmul.

**108.81 GB becomes roughly 29 GB.** That is the whole point, and it is not a bandwidth
argument: at 29 GB the trunk is fully resident on a 64 GB desktop, so the per-token trunk
read does not get faster — **it stops happening.** On the reference machine that read is
62.40 s of a 135.8 s token.

## What it costs, stated plainly

The engine's entire validation apparatus is built on producing the same bytes as a
reference: the op fixtures, the full-model oracle, the claim that output is identical at
8 GB and at 224 GB.

**None of that survives contact with a quantised trunk**, because the weights are no longer
the checkpoint's. The engine prints a banner saying so when it opens one, and
`k3_trunk.quantised` records it, so a captured log cannot be mistaken for a normal run.

The prior evidence says to expect this to hurt:

| evidence | figure |
|---|---|
| post-hoc int4 on 31 real attention tensors (`docs/data/trunk-quantisation.txt`) | 17.4% mean relative weight error |
| the same tensors at int8 | 0.96% |
| K3 technical report, 4.1.4 | keeps exactly these tensors in higher precision, on purpose |

MXFP4 is not plain int4 — it carries a shared exponent per 32 elements, which is why the
experts tolerate it — but it is 4-bit on a trunk that was never trained for it.
`--sample-error` reports the reconstruction error it actually achieved on your checkpoint.

## The gate that replaces bit-identity

```
k3 <shards> --trunk <bf16_trunk>  --ids <held-out ids> --ppl
k3 <shards> --trunk <mx4_trunk>   --ids <held-out ids> --ppl
```

`--ppl` runs one teacher-forced sweep and reports perplexity: the mean negative
log-likelihood the model assigns to the ids the sequence actually continues with. It is a
single number, but it *is* a number, which is more than "it still emits fluent text" ever
was.

Use ids the model has not been tuned on, and use the same ids for both runs. **The
difference between the two perplexities is the entire measurement.**

`--tf-check` is the companion figure: how often the quantised model's greedy token agrees
with the sequence, which is the acceptance rate a draft design stands on.

> **`--ppl` alone is not sufficient, and this repository found that out the hard way.**
> Commit `627d94f` measured an MXFP4 trunk scoring the modest-looking +13% below *while its
> generations collapsed into an eight-token loop*. Perplexity is teacher-forced — at every
> position the model is conditioned on the real text, never on its own output — so
> degeneration lives in the self-conditioned trajectory and is invisible to it by
> construction, at any corpus size. Run `benchmarks/eval/` as well, which adds a
> self-conditioned leg for exactly this reason.

## The first measurement

One 21-token prompt, 20 scored positions, identical ids, same 22 GB trunk budget:

| | bf16 trunk | MXFP4 trunk |
|---|---|---|
| perplexity | 97.40 | 110.05 |
| mean NLL | 4.5788 nats | 4.7009 nats |
| trunk on disk | 108.81 GB | 29.81 GB (3.65x) |
| layers pinned at 22 GB | 14/93 | 56/93 |

**+13.0% perplexity, +0.122 nats.** Directionally what the int4 weight-error measurement
above predicts, and enough to say the default should stay lossless.

> **Twenty positions is not a measurement.** Per-token NLL varies by whole nats, so the
> error on this mean is comparable to the gap it reports. `--ppl` now says so when the
> sequence is short. Before quoting a figure, use a few thousand ids of held-out text.

## Why the timings in that run are not a speedup

The sweep times in that run — 144.8 s bf16 against 127.3 s quantised — are **not** a speedup
measurement either.

`--ppl` is a single teacher-forced sweep, so the trunk is read ONCE while expert I/O is paid
for every position, and the experts are identical between the two runs (they were already
MXFP4). The 17.5 s difference is the trunk alone: 89.6 GB streamed against 8.7 GB, which at
~4.6 GB/s effective is ~17.5 s.

Decode pays the trunk **per token**, which is the case quantisation exists for. So measure
speed with `--gen`, not with `--ppl`.

## When to use it

**If you need the released model's exact behaviour, do not.** Stream the bf16 trunk; that is
what the default does, and why.

If you have a 64 GB desktop, want the model to run at desktop speed rather than at disk
speed, and can accept a model that is *derived from* K3 rather than *being* K3 — measure the
perplexity gap on your own workload and decide with the number in front of you. That is the
fork this exists for.
