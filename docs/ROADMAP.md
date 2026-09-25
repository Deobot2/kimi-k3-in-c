# Roadmap

Ordered by value, with the reasoning stated so the order can be argued with.

## 1. Chunked prefill, the highest-value missing piece

Prefill currently runs as a single forward pass over the whole prompt, and attention in
the 24 MLA layers is quadratic in sequence length. The practical effect: the context
ceiling is 32k tokens, but a 21k-token prompt does not complete in reasonable time.

Raising the ceiling was necessary and is done. `--mla-latent` has now removed the
*memory* half of the problem — 131k positions cost 7.25 GB instead of 310 GB — which
makes the remaining half purely about prefill cost. This is the part that makes it
usable.

## 2. Re-run the published campaign under the replicating harness

The measured noise floor is 33%, and almost every published figure is a single sample.
The harness itself is done, `benchmarks/memory-ladder.sh` and `benchmarks/split-sweep.sh`
already default to 3 repeats and report mean, sd and spread. What remains is re-running
the 12 ladder rungs and the 12 splits under it and replacing the single-sample tables in
docs/data/ with replicated ones.

**This is now the most urgent item on the list, not the second.** Five changes landed
that each claim a speed or memory effect — largest-first pinning, a deeper ring, io_uring
reads, S3-FIFO, speculative expert prefetch — and every one was argued from mechanism and
gated for correctness rather than measured for effect.

Two have since been measured on the released checkpoint, and **both arguments were
wrong**. S3-FIFO's flat 10% small queue LOST to the LRU it replaced at every size below
32 GB, including the laptop and desktop presets; it now sizes the small queue against one
token's working set. Speculative prefetch lost outright — 25.8 GB read per token to avoid
30% of it, on a run 96.9% I/O bound — and is off by default. That is 2 for 2 against
reasoning from mechanism, on a codebase whose own instrumentation was sitting there the
whole time, and the remaining three should be read in that light.

The deeper ring makes it 3 for 3. It shipped re-reading layers it had already fetched —
491 GB from a 29.81 GB trunk — and the first fix left 13.7% of the excess behind. Two
further explanations were then argued from the pinned set's shape, both without evidence,
because the fixtures could not reach the shape in question. What settled it was
enumerating the space instead: `t_sweep` in `tests/unit/test_trunk.c` walks a 93-layer
trunk at every budget and ring depth and failed 40 of 100 pinned shapes. **The pattern
across all three is the same — the mechanism argument was directionally right and
quantitatively unchecked**, and the fix each time came from measuring rather than
reasoning harder.
Each is A/B-able on a single binary by design (`K3_PIN_PREFIX`, `--trunk-ring`,
`K3_NOURING`, `K3_CACHE_POLICY=lru`, `K3_SPEC`), which is exactly what the harness
needs and exactly what has not yet been run.

## 3. Thread scaling

`OMP_NUM_THREADS` has never been swept on this engine. The workload is I/O bound at low
memory budgets, so the useful thread count is probably well below the core count, and
on memory-bound workloads throughput often *declines* past a point. Unknown here.

Two background threads now exist as well — the trunk reader and the expert-cache
speculator — so the sweep should cover their interaction with the OpenMP pool rather
than assuming the pool has the machine to itself.

## 4. SIMD in the remaining kernels — mostly done

The bf16 trunk matmul, the MXFP4 expert matmul, `k3_kda_step` (the KDA recurrence),
and now the `K3_WBF16` branch of `k3_matmul_tr` have hand-written AVX2 paths in
`src/core/k3_ops.c`. Both of the newer two vectorise across OUTPUT COLUMNS rather
than along the reduction — the opposite axis from the trunk/expert matmuls — because
each is a strided sweep where four or eight columns are independent scalar sums, so
that many SIMD lanes need no accumulator reshuffle at all: lane j is column j's serial
sum in the scalar order. Both checked bit-identical against the scalar path: the KDA
recurrence via `bench_kernels`' FNV1a hash and a standalone AVX2-vs-scalar comparison
across `dv` 1..20; `k3_matmul_tr` via a standalone comparison across seventeen `in`
values (1..512) and eight `rows` values, generating real finite bf16 weights rather
than raw random bytes (`bench_kernels`' own random-byte fill occasionally lands on a
bf16 NaN/Inf pattern, where FMA and separately-rounded multiply-add are legitimately
allowed to differ — that is a property of the test data, not of the kernel, and it is
why `bench_kernels`' FNV1a hash for `k3_matmul_tr` does not match between the two
builds while the standalone check, on finite weights, is exact).

Measured on `bench_kernels`' real dimensions, one token's worth: KDA's 69 layers
70.25 ms → 56.26 ms, `k3_matmul_tr`'s 96 heads × 24 MLA layers 52.30 ms → 15.85 ms.
Worth stating plainly: the recurrence is under 1% and `k3_matmul_tr` under 0.2% of the
measured 10 s/token compute budget (and `k3_matmul_tr` only runs at all under
`--mla-latent`), so these are real wins, not the trunk- or expert-matmul scale of
change.

What is left scalar: the `K3_WMX4` and `K3_WI8` branches of `k3_matmul_tr`, for a
quantised trunk under `--mla-latent`. Lower priority again — narrower still, and the
nibble-unpack-plus-per-group-scale shape (see the MXFP4 branch's own comment) makes an
equally rigorous AVX2 path more work for less.

## 5. Sampling

Greedy only today. Adding temperature and top-p is small, but note the trade-off: greedy
decoding is what makes output identical across memory budgets, which is a property the
test-suite depends on. Sampling must be opt-in and off by default.

## 6. Chat template

K3 ships an XTML chat format. Without it the engine produces base-model continuations,
which is why asking a question gets the question completed rather than answered.

## 7. Vision

K3 is natively multimodal. The vision tower is 27 layers and ~0.4B parameters, small,
and its weights are ~0.9 GB, a fraction of a percent of the checkpoint. Self-contained
enough to be tractable.

## 8. Serving

No HTTP API. Deliberately last: it is product surface, and everything above changes what
would be served.

## Explicitly not planned

**A precision dial for the trunk, as a default.** The trunk streams losslessly and always
will: post-hoc int4 measures ~17% mean relative weight error on K3 attention tensors
against ~1% for int8, streaming costs time which more memory buys back, and rounding
costs accuracy which nothing buys back.

What *does* exist is `tools/mxfp4_trunk.py`, which writes a quantised trunk into its own
container behind its own flag, because 29 GB is fully resident on a 64 GB desktop and
that removes the per-token trunk read rather than speeding it up. It is a fork of the
contract, not a setting: the engine announces it in a banner, every bit-identity gate in
the suite is void against it, and `--ppl` is what replaces them. See
[notes/compressed-trunk.md](notes/compressed-trunk.md). Anyone who needs the released
model's exact behaviour should stream bf16, which is what the default does.

**GPU support.** Out of scope for this project.
