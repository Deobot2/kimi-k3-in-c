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

## 4. SIMD in the KDA recurrence

**The KDA half of this is done, and it took no code.** The bf16 trunk matmul and the
MXFP4 expert matmul have hand-written AVX2 paths (`src/core/k3_ops.c`) because their
reduction is a genuine cross-element sum that needs a specific tree to stay bit-identical
across thread counts. `k3_kda_step`'s four loops are not that: `i` (dk) is the only
sequential axis, `j` (dv) carries no reduction at all, so vectorising over `j` changes no
arithmetic and GCC already does it — measured, a hand-written AVX2 path using the same
`mul_ps`/`add_ps` lane ops timed within noise of the plain scalar loop already in the tree
(~8.8 vs ~8.6 μs per head-step at `-O3 -march=native`, both ~4x faster than the same
binary built with `-fno-tree-vectorize`), so it was reverted rather than kept for no
measured gain. See the Research notes entry in `CHANGELOG.md`.

**`k3_matmul_tr` is also done, but the fix was not vectorisation.** It walked its output
column-outer, reading one element from every row per column — `in` elements apart, a
fresh cache line every access. The comment argued this was fine because the whole operand
fits in L2; measured, that bounded capacity misses but not the cost of touching a new
line for two bytes of it each time. Rewritten rows-outer (each row read once,
contiguously, into a small tiled output buffer, same summation order, bit-identical): 4.4
-5.0x faster, measured at the real call-site shape (in=512, rows=128, bf16).

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

## 9. The trunk reader goes idle at every token boundary, not just the first

Found while reviewing the streaming path; not yet fixed, because fixing it correctly
touches the one subsystem this project has been wrong about from mechanism three times
running (S3-FIFO, speculative prefetch, the ring re-read bugs), and there is no checkpoint
or real disk in this environment to measure the actual wall-clock effect. Recorded here
rather than shipped, per the project's own rule that this class of change needs a number,
not an argument.

**The mechanism.** `k3_trunk_prefetch(tr, L)` (`src/io/k3_trunk.c`) scans
`for (n = L; n < tr->n_layers && io->nq < tr->nslot; n++)` — forward only, never wrapping
past the last layer. The only production call site,
`k3_trunk_prefetch(w->trunk, L + 1)` in the per-layer loop in `src/cli/k3_run.c`, therefore
does nothing on the last layer of every walk (`L + 1 == n_bound` makes the loop condition
false immediately). Since `forward()` runs this whole loop once per generated token, the
reader thread has nothing queued at the end of every token, not just the first — whatever
the first unpinned layer of the NEXT token's walk is gets read with zero overlap against
the previous token's tail (final norm, lm_head, sampling), which is exactly the class of
cost the ring exists to hide. `tests/unit/test_trunk.c`'s own `walk()` harness calls
`k3_trunk_prefetch(tr, L + 1)` the same way, so it has never exercised a wraparound either
— this gap would not show up in any currently passing test.

**Why it is a safer bet than speculative expert prefetch, which was rejected.** That
guess is uncertain about WHICH experts the next token's router will pick, and was measured
losing (43.8 GB/token read to net 25.8 GB useful). This is not a guess: the walk order is
fixed 0..92 on every token by construction (already relied on elsewhere in this file's own
comments), so IF there is a next token, it is certain to need layer 0 (or whichever layer
follows the pinned run at the start) first. The only uncertainty is whether generation
continues at all — wrong only on the very last token of a run, a one-time cost bounded by
the ring depth, not a per-token gamble.

**The complication that stopped a quick fix.** The obvious patch — wrap the scan modulo
`tr->n_layers` — changes `test_trunk.c`'s strict per-layer bound
(`reads_of[L] > (pinned ? 1 : passes)` fails the test) at the tail of the final pass: the
wrapped scan queues a read for pass `passes + 1` of up to `nslot - 1` layers near the start
of the walk, which the test's fixed-`passes` loop never consumes but the reader thread
still completes, so `reads_of[L]` for those layers exceeds `passes`. In production this is
harmless (the read is simply wasted I/O on the last token of the run, once per whole
generation rather than once per token); in the test it is a real invariant violation that
needs a deliberate decision, not a silent loosening — e.g. bounding the tolerance to
`passes + 1` for at most `nslot - 1` layers rather than weakening the check for every
layer. Whoever picks this up should settle that test change first, then measure the actual
per-token latency saved against a real checkpoint before it ships, the same way largest-first
pinning and the ring depth were.

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
