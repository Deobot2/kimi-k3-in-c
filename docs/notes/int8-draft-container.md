# Int8 draft container: built, correct, and not worth it on a laptop

> **Verdict: a large-memory technique with narrow, unproven economics — not the
> consumer-laptop speedup it was built to be.**
> Everything was built and everything is correct. Two separate attempts to make it *fast*
> were measured, and both failed for the same underlying reason: the routed experts, not
> the trunk, are what a draft step actually costs.
> *Origin: upstream, v1.0.0. Every measurement and conclusion below is upstream's and is
> unchanged. See [what this fork changed](#what-this-fork-changed) at the end.*

**Read this in order.** It is a chronology: hypothesis, build, measurement, failure, second
hypothesis, second measurement, second failure, conclusion.

---

## Where it fits

Hybrid decode (`--draft-trunk`) proposes tokens with a cheap draft model and verifies them
with the exact one. The correctness half was proven first:

| | measured |
|---|---|
| teacher-forced agreement of the int8 derivation | 94.2% (against a 96.2% ceiling) |
| first flight on the real checkpoint | 66.7% of drafts accepted, byte-identical output |

The emitted tokens are exactly the exact model's greedy output. That property is
**structural**: the draft only ever proposes, and the exact bf16 model is the sole authority
on what is emitted. No amount of draft inaccuracy can change the output — only the speed.

What was missing was the speed half. The draft trunk was stored in bf16 (109 GB): a fidelity
*simulator*, not a size-reduced container. On a 64 GB machine it cannot be resident and
streams about half of itself per draft step, which measured *slower* than the exact run at
that budget.

The draft only pays off when it is **resident**, and a true int8 container is 56.7 GB, which
fits the 64–110 GB band. So the container was built.

## The build

Three parts.

**1. Format.** `tools/pack_trunk.py --int8`: for every 2D bf16 trunk tensor, per-row
symmetric absmax int8 (or per-group, group 128, if per-row quality is short). Scales are
stored **inline** — one fp32 per row, prepended to that row's int8 bytes — so a weight matrix
stays a single tagged pointer and the kernel derives the scale from `W`. This avoids
threading a parallel scale array through `K3MlaW` / `K3KdaW` / `K3MoeW` / `K3LayerW`. The
manifest carries `dtype: "I8R"`.

**2. Kernel.** `k3_matmul_q8(y, x, W, in, out)`, where each row is `[f32 scale][int8[in]]`:
widen int8 to int32, multiply by the fp32 activation, accumulate, scale once at the end.

It does **not** need to be bit-identical to anything — the draft is only a proposal source.
So it can use the fastest AVX2 form (maddubs-style or cvt+fma) without the four-accumulator
determinism contract.

**3. Dispatch.** Add `K3_WI8` to the wdt enum, one branch in `k3_mmw`, `k3_wsz` returns the
per-row stride, and `k3_bind_layer_mem` recognises the `I8R` dtype from the trunk manifest.
Only the **draft** weights ever carry this tag; the exact model stays bf16, so no oracle gate
and no exactness claim is touched.

## The gate it had to pass

Pack the int8 container, confirm the draft's teacher-forced agreement is still ~94% (the
inline per-row scheme should match the qdq measurement), run the hybrid resident under a
64–68 GB cgroup cap, and require its s/token to **beat the exact run at the same budget** —
19.8 s/token on the proof NVMe.

Output identity against the exact greedy run is the correctness gate, and is already
structural.

---

## Attempt 1 — the resident container: built, correct, still slower

Everything was built and everything worked:

| component | status |
|---|---|
| `k3_matmul_q8` | unit-tested at rel-L2 0.37% vs fp32 |
| `tools/int8_trunk.py` | produces a **54.47 GB** container, half the bf16 trunk |
| `K3_DT_I8R` dtype + bind wiring | matmul weights tagged `K3_WI8`; elementwise tensors (like the AttnRes projection) dequantised into the widen buffer |

And it ran as designed:

| | measured |
|---|---|
| resident | yes — 66 GB RSS on a 64 GB cap |
| output | byte-identical |
| teacher-forced agreement | 90.9% on a 22-token sample (same band as the qdq simulator) |
| acceptance, one prompt | 100% of drafts, mean run 4.0 |
| **s/token** | **53, against 19.8 for exact decode at the same 64 GB** |

### Why it lost

The reason is a real one, and it is the important finding of this whole note:

> **Making the TRUNK resident does not make a draft step cheap, because the draft still
> streams the routed EXPERTS.**

The experts are MXFP4 and 1.45 TB. They can never be resident, so every drafted token still
reads ~25.8 GB of experts — the same as a real decode step. **A draft that costs as much as
the thing it drafts for cannot amortise anything.**

Splitting a 64 GB machine 56/2.5 between draft and exact made it worse: it starved the exact
model's trunk budget, so its verify sweeps streamed the full bf16 trunk.

---

## The hypothesis: make the draft cheap on the expert side

The fix had to be a **cheap draft on the expert side**, so a proposal costs far less than a
real step. Two routes, both identity-safe because the draft only proposes:

| route | idea | expert bytes per draft token |
|---|---|---|
| **cache-only routing** | the draft routes only among experts already resident in the cache; the exact verify still uses true routing | zero new |
| **top-k reduction** | the draft uses top-4 of 896 instead of top-16 | 4x fewer |

With either, a draft token costs a fraction of a real one, the resident int8 trunk carries
the draft's dense compute, and the batched bf16 verify amortises across accepted tokens.
That was the configuration in which the 90.9% agreement and 100% acceptance measured above
would turn into a real speedup.

Cache-only routing was built. Top-k reduction was not.

## Attempt 2 — cache-only routing: built, and it collapsed

`K3ExpertSrc::resident` plus a `cache_only` MoE mode make the draft route only among experts
already in the cache and renormalise over them, so a drafted token reads zero new expert
bytes. Output stays bit-exact (the exact model verifies), and the weightless gates are
untouched.

Measured with a small (4 GB) expert cache:

| | full-routing draft | cache-only draft |
|---|---|---|
| acceptance | 66–100% | **12.5%** |
| new expert bytes per draft token | ~25.8 GB | zero |

The cache held under 1% of the experts, so the draft routed among far too few and proposed
badly. **The draft became cheap but useless.**

---

## The two dead ends

| | accurate? | cheap? |
|---|---|---|
| full-routing draft | yes | no — a draft step costs as much as a real one |
| cache-only draft | only when the cache holds most routed experts | yes |

Both are resolved only in the same place: **large RAM.** The hybrid pays when the 55 GB int8
draft trunk is resident **and** the expert cache is large enough (tens of GB) that cache-only
routing stays accurate **and** the exact model still has a trunk pin.

That is a 100 GB-plus machine, not a laptop. And at 128 GB, plain exact decode is already
5.59 s/token — so the hybrid's marginal gain there is uncertain and unproven. It is not the
laptop lever it was hoped to be.

## Status

**Built, correct, bit-exact, and a reusable foundation:** the container, the q8 kernel, the
`I8R` format, the bind wiring, and cache-only routing.

**Never built:** top-k reduction — the one route from the hypothesis above that was not
measured. It is a small change to the draft's routing call, not another format.

**The honest conclusion:** the quantized-self-draft hybrid is a large-memory technique with
narrow, unproven economics, not a consumer-laptop speedup. The laptop wins are elsewhere —
the fused kernels, RAM-first pinning, chunk-union prefill, and conversation resume, all
measured.

---

## What this fork changed

**Nothing in the findings.** No measurement above was re-run in this fork, and no conclusion
was revised. The draft container, the q8 kernel, the `I8R` format, the bind wiring, and
cache-only routing are all upstream work as released in v1.0.0.

What this fork did touch was the code *around* the draft path, keeping it working as other
subsystems were replaced:

| fork commit | what it did to this path |
|---|---|
| `9106e67` — S3-FIFO cache | taught the new speculative prefetch to respect `cache_only`, so the cache-only draft mode still reads zero new expert bytes |
| `cb489a1` — latent KV cache | **added a restriction:** `--kv-window` cannot be combined with `--spec` or `--draft-trunk`. A rejected draft's rows wrap onto positions the replay still needs, so the replay would read a rejected draft's latent as if it were accepted |
| `2a9f9b5`, `cf154bb` — MXFP4 trunk | extended the weight-type enum with `K3_WMX4` alongside `K3_WI8`, and gave `K3_WI8` a `k3_row_bytes` stride path. The `I8R` inline-scale layout described above was the model MXFP4 packing followed |

The presentation of this note was also revised in this fork. The content is upstream's; the
ordering is not. It previously ended with a section proposing cache-only routing as the next
step, below the section reporting that cache-only routing had been built and had failed. That
section is now where it belongs chronologically, as the hypothesis that attempt 2 tested.
