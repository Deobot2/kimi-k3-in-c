# Tuning

## The short version

1. Run `./scripts/k3-doctor.sh` and use the preset it names.
2. If you tune by hand: **fill the trunk before you feed the expert cache.**
3. Use `--incremental` unless you are validating against full recompute.
4. Add `--mla-latent` unless you need bit-identity with the expanded KV cache. It makes
   context cost 55.3 KB per position instead of 2.37 MB.

Everything below is why.

## One decision matters more than the rest

The engine splits your memory budget between two caches:

- `--trunk-gb`, the ring buffer and pinned layers for the **dense trunk**
- `--cache-gb`, the arena for **routed experts**

These are not interchangeable, and the asymmetry is large.

Per token the engine re-reads the **entire 108.81 GB trunk**, all 93 layers, in a fixed
order, every single token. It reads only **~25.8 GB of routed experts**, because just 16
of 896 are selected per layer.

So a gigabyte given to the trunk removes about **1.17 GB/token of guaranteed traffic**
(one pinned layer, never read again). A gigabyte given to the expert cache removes,
below roughly 36 GB of arena, **nothing measurable**.

Measured at a fixed 128 GB budget, endpoints of a six-point sweep:

| trunk | cache | s/token |
|---:|---:|---:|
| 12.3 | 110.7 | 28.38 |
| **110.0** | **13.0** | **16.80** |

**1.69× from allocation alone.** The faster configuration has a *smaller* expert cache
and 0.0% expert retention.

These are single samples against a 33% noise floor, so read the *direction*, not the
exact factor. The direction is supported by twelve points across two independent budgets
(Spearman ρ = −0.886 at 128 GB, −0.714 at 32 GB) and by the mechanism above; the
magnitude is not replicated. All twelve rows and the caveats are in
[PERFORMANCE.md](PERFORMANCE.md#allocation-beats-capacity), raw data in
[data/trunk-cache-split.tsv](data/trunk-cache-split.tsv), and
`benchmarks/split-sweep.sh` re-runs the experiment with repetitions.

## Why the expert cache is so weak

This is the model's design, not a shortcoming of the implementation.

K3's router is trained with **Quantile Balancing**, which deliberately flattens expert
usage so that no expert is favoured. Flat usage is precisely what defeats a
least-recently-used cache: with no hot subset, a few gigabytes retain nothing worth
keeping. Measurements bear this out, retention stays at exactly 0.0% from 28 slots all
the way to 1,344, and the bytes read per token do not move by a single decimal.

It does eventually engage. Somewhere around 36 GB of arena, retention jumps to ~30% and
bytes/token fall from 25.83 to 18.11. But by then you could have pinned most of the trunk
for the same memory and gone faster.

### What changed, and what did not

The paragraph above says *LRU* is defeated by flat usage, and that is the more precise
claim. Flat is not uniform, and the difference is what a frequency-aware policy lives on.
The offline simulator put 25.5 points between LRU and Belady's optimum at 64 GB (36.24%
against 61.74%) — the largest single gap in this project's measurements, and an argument
that the lever is the policy rather than the size.

The engine therefore runs **S3-FIFO**, not LRU: a small FIFO takes every admission so
one-hit wonders leave without polluting the cache, a main FIFO holds what proved itself,
and a ghost queue of recently evicted keys lets a prompt return skip probation. There is
also a **speculative prefetch**: about 90% of expert requests in a real trace are repeats,
so on entering layer L the cache starts reads for what layer L wanted for the previous
token, before the router has run.

Neither changes the advice above. The trunk is still re-read in full every token and the
experts still are not, so trunk-first still wins by a wide margin. What they change is
what a gigabyte of expert cache is worth once you have given the trunk everything.

Both are A/B-able on one binary, which is the only honest way to attribute a difference:

    K3_CACHE_POLICY=lru   # the old policy
    K3_SPEC=1             # speculative prefetch ON (off by default; it lost)
    K3_NOPREFETCH=1       # no batch prefetch either

### The one number that decides whether cache capacity is worth anything

One token touches `topk` experts in each of the 92 MoE layers: **1,472 experts, 25.8 GB**.
Below that the cache cannot hold a single token's working set, so nothing survives to be
reused and **capacity does not move the hit rate at all** — measured flat from 4 GB to
24 GB on the recorded trace:

| cache | 4 GB | 8 GB | 16 GB | 24 GB | 32 GB | 64 GB | 128 GB |
|---|---|---|---|---|---|---|---|
| hit rate | 36.2% | 36.2% | 36.2% | 36.2% | **42.7%** | **50.0%** | **73.7%** |

So an expert cache is worth either the floor (~0.5 GB) or more than 25.8 GB, and nothing
in between. A 10 GB cache buys exactly what a 0.5 GB cache buys, and those 9.5 GB belong
in the trunk. `--preset auto` now allocates on exactly that rule.

It is also why the S3-FIFO small queue is not the paper's flat 10%: an admission filter
only pays once the main queue is worth protecting, and below one working set it just
evicts objects before their second touch. The engine sizes the small queue from
`cfg->topk` and the MoE layer count for that reason; `tools/sim_cache.py` carries the
table that decided it.

## Presets

```console
$ ./bin/k3 --list-presets
```

| preset | trunk | cache | total | expect |
|---|---:|---:|---:|---|
| `laptop` | 3 | 1 | ~10 GB | ~32 s/token |
| `desktop` | 16 | 10 | ~32 GB | ~31 s/token |
| `workstation` | 60 | 30 | ~96 GB | ~24 s/token |
| `server` | 110 | 13 | ~128 GB | ~17 s/token |
| `max` | 110 | 109 | ~224 GB | ~19 s/token |

`server` is the best value: the trunk is fully pinned at 110 GB and everything beyond
that is spent on a cache that contributes little. Note `max` is not faster than `server`
in these measurements, the extra 96 GB buys nothing outside the noise floor.

Flags after `--preset` override it, so `--preset server --cache-gb 40` works.

## Other options

**`--incremental`** carries a KV cache and the recurrent state between tokens instead of
recomputing the prefix. Verified to produce identical tokens to full recompute. Use it
unless you are specifically testing that equivalence.

Note it allocates ~2.37 MB of KV cache **per position**, so long contexts cost
memory: 16k positions is ~39 GB. The engine computes the requirement up front and refuses
with both numbers rather than being OOM-killed an hour in.

**`--trunk DIR`** enables trunk streaming and is what makes memory a dial. Without it the
trunk is fully resident and the floor is ~115 GB. Pack it once with
`tools/pack_trunk.py`.

**`--layers N`** binds only the first N layers. Useful for testing the machinery on a
partial download; the output is not the full model and the engine says so.

## Storage matters more than you expect

The engine moves ~135 GB per token at low budgets. Storage bandwidth is usually the
ceiling, not the CPU.

- Put the checkpoint and the packed trunk on the **fastest local NVMe** available.
- Network or spinning storage will dominate everything else. A 6× difference in device
  bandwidth is a 6× difference in throughput on this workload, which is larger than any
  tuning decision in this document.
- `./scripts/k3-doctor.sh` measures your device so you know which regime you are in.

## Threads

`OMP_NUM_THREADS` defaults to your core count. The workload is I/O bound at low memory
budgets, so more threads help less than you would expect once the trunk is streaming.

This has not been swept systematically on this engine see [ROADMAP.md](ROADMAP.md).

## Before you conclude a change helped

The measured run-to-run spread on an identical configuration is **33%**. Differences
smaller than that are not effects. Run each arm at least three times.
[BENCHMARKING.md](BENCHMARKING.md) has the procedure.

## Context, and the two dials that bound it

The MLA KV cache is the only thing in this engine that grows with the conversation.
Everything else — trunk, experts, the KDA recurrent matrices — is fixed.

| positions | expanded (default) | `--mla-latent` |
|---:|---:|---:|
| 4,096 | 9.7 GB | 226 MB |
| 16,384 | 38.8 GB | 905 MB |
| 32,768 | 77.7 GB | 1.81 GB |
| 131,072 | 310.6 GB | 7.25 GB |
| 1,048,576 | 2,485 GB | 58.0 GB |

`--mla-latent` caches the 576 floats `kv_a` emitted instead of the per-head keys and
values `kv_b` would expand them into, and uses **weight absorption** so the expansion
never has to happen: `W_UK` moves onto the query and `W_UV` past the attention sum. See
the long note in `include/k3/k3.h`.

It costs about 3.4× the attention arithmetic, which this engine can afford — it spends
40–77% of a token waiting on a disk. What it also costs is **bit-identity**: reassociating
two matmuls changes the last few ulps, so a latent run is not byte-comparable with an
expanded one. It is gated instead by `mla_latent` in `test_ops` (against the same torch
reference, at a stated multiple of the fixture tolerance) and by GATE 4 of the oracle,
which requires the same decoded ids.

`--kv-window N` caps the cache at N positions plus `--kv-sinks S` permanent early ones.
**This is local attention and it changes the output.** It is off by default, the engine
says so on stdout when it is on, and it exists for the case where the alternative is not
running at all.

There is no exact version of that trade. Reconstructing an evicted MLA latent needs the
layer's input at that position, which depends on every preceding attention, so the only
exact recompute is a full forward replay of the prefix — which is precisely what the
engine's default non-`--incremental` mode already does. If you want to spend compute to
avoid KV memory *exactly*, drop `--incremental`; that is the knob, and it already existed.

## Reads, and how deep they go

`--trunk-ring N` (default 2) sets how many layers can be in flight. One slot is the layer
being computed on; the rest are reads running ahead. Two overlaps one read with one
layer's compute, which is enough only while the two take about the same time — and they
do not, so a third slot lets the reader stay a layer further ahead. It costs one more
slot of RAM (2.37 GB at the floor) and the budget still wins if it does not fit.

Reads themselves use **io_uring** where the platform has it, splitting each layer into
8 MB requests with up to 16 outstanding, because an NVMe device does not reach its rated
bandwidth at queue depth one — which is what a `pread` loop is. `K3_NOURING=1` forces
`pread`; `K3_SQPOLL=1` asks the kernel to poll submissions from its own thread, which
costs a spinning core and is off by default.

Pinning is **largest-first**. Which layers you pin is byte-neutral for the traffic it
avoids, but not for the ring: the uniform slot has to hold the largest layer still
streaming, and layers here range to 2.34 GB against a 1.17 GB mean. Taking the fattest
out of the ring first shrinks the slot, which frees budget, which pins more.
`K3_PIN_PREFIX=1` restores the old prefix order for comparison.
