# Compressed trunk: investigated, shelved, with the measurement that decided it

## The idea

The trunk is bf16. A bf16 value's high byte (sign + exponent) carries about 2.8 bits of
real information on this checkpoint: roughly a dozen byte values cover 99.9% of all
weights. Entropy-coding just that byte plane, and storing the mantissa byte raw (it
measures as noise), shrinks the packed trunk to about 67% of its size. Because the decoded
bytes are the checkpoint's own bytes, it is lossless and the engine's output is unchanged.
At a streamed budget, fewer trunk bytes is fewer seconds per token.

## What was built and proven

A canonical length-limited Huffman codec, encoder in Python (`huf_encode.py.shelved`) and a
decode-only C header (`k3_huf.h.shelved`). On real trunk bytes it round-trips byte-exactly,
Python-encoded and C-decoded, at a measured **1.45x** (ratio 0.689) on a 4 MB sample. The
format, the Kraft check, and the canonical code assignment all work.

## The measurement that shelved it

Decode speed. A hand-rolled scalar Huffman decoder reached only **~0.31 GB/s per core** after
two optimisation passes. To keep a compressed trunk fed on a 3 GB/s laptop SSD at that rate
needs roughly ten cores decoding, which laptops do not have. So compression built this way
**helps many-core machines, which have the RAM to not need it, and starves the laptops that
do**. That is backwards from the goal.

A fast entropy decoder exists: FSE/Huff0 measured 1724 MB/s on the same exponent-plane bytes.
But even at that speed it needs two dedicated cores to match a laptop SSD, and vendoring it is
about two thousand lines against a project whose identity is a 176 KB binary with no
dependencies. The payoff did not justify the weight, for the machines that matter.

## If revisited

The unlock is a decoder in the 1 to 1.5 GB/s/core range that stays small: four interleaved
bitstreams per stripe for instruction-level parallelism, or a double-symbol table, are the
standard routes and would let the encoder stay as-is. The prototypes here are a correct
starting point; the format round-trips today. Until a decoder that fast is in hand, the
streamed-trunk speed lever is pinning (RAM-first `--preset auto`) and the resident int8 draft,
not compression.

---

# Quantised trunk: shipped, behind a flag, with a different gate

## Why this note continues

Everything above is about a LOSSLESS compressed trunk, and its conclusion still stands:
entropy coding helps the machines that do not need it. This section is about the other
axis, which the project had declined outright — quantising the trunk — and which now
exists as an opt-in fork of the container.

## What it is

`tools/mxfp4_trunk.py` rewrites a packed trunk with every matmul weight in OCP MX FP4,
the same format the routed experts already ship in and the same kernel reads. Norms,
biases, `A_log`, `dt_bias`, the conv kernels and the AttnRes projections pass through
untouched: they are read elementwise as fp32 and are well under 1% of the bytes. The
router gate passes through too, because `k3_router` walks it with its own inline fp32
matmul.

    python3 tools/mxfp4_trunk.py /path/to/trunk /path/to/trunk-mx4 --sample-error
    k3 <shards> --trunk /path/to/trunk-mx4 --ids ... --incremental

108.81 GB becomes roughly 29 GB. That is the whole point, and it is not a bandwidth
argument: at 29 GB the trunk is fully resident on a 64 GB desktop, so the per-token trunk
read does not get faster, it stops happening. On the reference machine that read is
62.40 s of a 135.8 s token.

## What it costs, stated plainly

The engine's entire validation apparatus is built on producing the same bytes as a
reference — the op fixtures, the full-model oracle, the claim that output is identical at
8 GB and at 224 GB. **None of that survives contact with a quantised trunk**, because the
weights are no longer the checkpoint's. The engine prints a banner saying so when it
opens one, and `k3_trunk.quantised` records it, so a captured log cannot be mistaken for
a normal run.

The prior evidence says to expect this to hurt. This repository measured post-hoc int4 on
31 real attention tensors at 17.4% mean relative weight reconstruction error against
0.96% for int8 (`docs/data/trunk-quantisation.txt`), and the K3 technical report (4.1.4)
keeps exactly these tensors in higher precision on purpose. MXFP4 is not plain int4 — it
carries a shared exponent per 32 elements, which is why the experts tolerate it — but it
is 4-bit on a trunk that was never trained for it, and `--sample-error` will report the
reconstruction error it actually achieved on your checkpoint.

## The gate that replaces bit-identity

    k3 <shards> --trunk <bf16_trunk>  --ids <held-out ids> --ppl
    k3 <shards> --trunk <mx4_trunk>   --ids <held-out ids> --ppl

`--ppl` runs one teacher-forced sweep and reports perplexity: the mean negative
log-likelihood the model assigns to the ids the sequence actually continues with. It is a
single number, but it is a number, which is more than "it still emits fluent text" ever
was. Use ids the model has not been tuned on and use the same ids for both runs; the
difference between the two perplexities is the entire measurement.

`--tf-check` is the companion figure: it reports how often the quantised model's greedy
token agrees with the sequence, which is the acceptance rate a draft design stands on.

## When to use it

If you need the released model's exact behaviour, do not. Stream the bf16 trunk; that is
what the default does and why.

If you have a 64 GB desktop, want the model to run at desktop speed rather than at disk
speed, and can accept a model that is *derived from* K3 rather than *being* K3 — measure
the perplexity gap on your own workload and decide with the number in front of you. That
is the fork this exists for.
