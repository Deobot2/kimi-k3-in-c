# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Seven changes, all aimed at the same fact: this engine is I/O bound by a factor of four,
so almost every remaining win comes from spending idle CPU to avoid a disk read, or from
not needing the bytes at all.

### Added

- **`--mla-latent`**: caches the 576-float MLA latent instead of expanded per-head keys
  and values — **55.3 KB per position instead of 2.37 MB**, so 131k context costs 7.25 GB
  rather than 310 GB. Weight absorption (`W_UK` onto the query, `W_UV` past the attention
  sum) means the expansion never happens rather than happening repeatedly. It costs about
  3.4× the attention arithmetic and it is **not bit-identical** to the expanded path;
  gated by `mla_latent`/`mla_lat_inc`/`mla_lat_window` in `test_ops` and by GATE 4 of the
  oracle, which requires the same decoded ids.
- **`--kv-window N` / `--kv-sinks S`**: bounds the latent cache to a rolling window plus
  attention sinks. This is local attention and **changes the output**; off by default, and
  the engine says so on stdout when it is on. GATE 5 pins the one case where it must
  change nothing.
- **`--trunk-ring N`**: streaming ring depth (default 2). A third slot lets the reader run
  a layer further ahead, which matters because a trunk read is 62.40 s of a 135.8 s token
  and two slots leave the reader idle for part of every layer.
- **io_uring trunk reads** (`src/io/k3_uring.c`): each layer run is split into 8 MB
  requests with up to 16 outstanding, because an NVMe device does not reach its rated
  bandwidth at the queue depth of one a `pread` loop provides. Raw syscalls, no liburing
  dependency, and it falls back to `pread` whenever unavailable or disabled with
  `K3_NOURING=1`. `K3_SQPOLL=1` is available and off by default.
- **`--ppl`**: perplexity over an id sequence in one teacher-forced sweep. This is the
  gate that replaces bit-identity for a quantised trunk, which cannot be byte-compared
  against anything.
- **`tools/mxfp4_trunk.py`**: writes a quantised trunk, ~29 GB against 108.81 GB, which is
  fully resident on a 64 GB desktop — so the per-token trunk read stops happening rather
  than getting faster. **It is a fork of the contract, not a setting**: the weights are no
  longer the checkpoint's, every bit-identity gate is void against it, and the engine
  prints a banner saying so. See `docs/notes/compressed-trunk.md`.
- **`tests/unit/test_trunk.c`**: the streaming trunk was unreachable without a 108.81 GB
  checkpoint and every failure in it is silent. It now runs against a synthetic trunk
  whose layers carry byte patterns derived from their own index, at three ring depths and
  on both read paths — and asserts that no layer is read more than the walk owes, per
  layer as well as in total. `t_sweep` re-runs that assertion across the whole space of
  pinned shapes at the real 93-layer count, because a hand-built fixture proves one
  arrangement is handled and the arrangement that escaped was not one anybody guessed.
- **`tools/awq_trunk.py`**: 4-bit activation-aware quantisation of the trunk. Plain MXFP4
  rounds every weight identically regardless of what it multiplies, which is the wrong
  objective — what reaches the next layer is `W x`. AWQ uses the exact identity
  `W x == (W diag(s)) (diag(s)^-1 x)` to spend the 4-bit grid where the activations are
  large. **The inverse scale is folded into the preceding RMSNorm weight**, which is fp32
  in the trunk, so the fold is exact and the output is *ordinary MXFP4* that the existing
  kernel reads unchanged — no format change, no kernel change. Tensors whose input has no
  norm in front of it (`o_proj`, `f_b_proj`, the shared and dense `down_proj`) are
  quantised plain; the tool reports which. α is searched per group with α=0 always in the
  grid, so it cannot do worse than `mxfp4_trunk.py`.
- **`k3 --calib-dump PATH`**: per-input-channel activation statistics over an eval suite,
  which is what AWQ needs and only this engine can produce at full scale. Five slots, one
  per activation that both feeds quantised weights and has a norm to fold into.
- **`tools/verify_awq_fold.py`** and `make awq-check`: build a trunk with the scaling and
  the fold applied but *no* quantisation, and require it to reproduce the source model.
  Plus a **coverage audit inside `awq_trunk.py`** that refuses to run when any narrow
  tensor matching a rescaled activation's width is neither a declared consumer nor a
  declared non-consumer. Both exist because they catch different bugs: omitting the router
  rescale moves log-probabilities by 1.9e-04 nats and the numerical check catches it,
  while omitting `g_proj` moves them by 2.9e-06 against a 1.4e-06 rounding floor and only
  the structural check catches it. Both omissions are equally wrong.
- **`tools/make_random_checkpoint.py`**: a loadable checkpoint of noise, numpy only, so
  the quantisers and the harness can be exercised without torch or the 110 GB download.

### Changed

- **Expert cache replacement is S3-FIFO, not LRU.** The project's own simulator put 25.5
  points between LRU and Belady at 64 GB, and its own conclusion was that the lever is the
  policy rather than the size. Small FIFO, main FIFO, ghost queue; uniform object size
  makes it the friendliest possible case. `K3_CACHE_POLICY=lru` restores the old policy on
  the same binary, and `tools/sim_cache.py` reports both.
- **Speculative expert prefetch, implemented and then turned OFF after measuring it.**
  On entering layer L the cache can start reads for what that layer wanted for the
  previous token, before the router has run. Measured on the released checkpoint it read
  1,472 experts per token and 30.3% of the guessed set was requested: 43.8 GB per token
  against a 25.8 GB maximum, on a run 96.9% I/O bound. The "90% of requests are repeats"
  premise describes reuse within a full-recompute forward pass, not consecutive-token
  overlap. `K3_SPEC=1` enables it for machines with spare read bandwidth.
- **Trunk layers are pinned largest-first, not as a prefix.** Which layers you pin is
  byte-neutral for the traffic avoided but not for the ring, whose uniform slot must hold
  the largest layer still streaming — prefix pinning took the 2.34 GB dense layer first
  and then kept paying for a slot sized to it. `K3_PIN_PREFIX=1` restores the old order.
- `test_cache` now runs its whole battery under both replacement policies, and adds a
  section that drives the speculation thread the way `k3_moe` does.
- Saved state records the KV layout and window geometry and refuses a mismatch: the two
  caches hold different tensors of the same float count, so restoring one as the other
  would be fluent and wrong. State version 1 → 2.
- **AVX2 for the KDA recurrence** (`k3_kda_step`), the last hot kernel on the non-I/O path
  without a vector path (docs/ROADMAP.md item 4). It vectorises more simply than the
  matmul kernels: every reduction here is already an outer loop over the contraction
  index accumulating into an array indexed by the vectorised inner one, so there is only
  ever one accumulator per output element rather than a reduction tree to reproduce.
  Every vector loop still does a separate mul (or sub) and add, never an FMA, matching
  `-ffp-contract=off` exactly the way the bf16 matmul kernel already does. Verified
  bit-identical against a pure-scalar build across dk/dv spanning multiples and
  non-multiples of 8; a synthetic microbenchmark at the real kda_head_dim=128 measured
  ~20% faster, best-of-7 to cut scheduler noise.

### Fixed

- The trunk reader's `hits`/`misses` are classified once per bind rather than incremented
  by whichever thread performed the read, so they sum to the bind count instead of double
  counting every prefetched layer.
- Two hangs found by the new trunk test, both intermittent at roughly one run in three:
  the pinned-layer read on the main thread was driving the reader thread's io_uring (one
  ring, two submitters, each reaping the other's completions), and the submit path assumed
  `io_uring_enter` always consumes every queued entry, so a partial submission left
  entries in the ring forever while the next call waited on a completion that was never
  coming.
- The S3-FIFO victim search could cycle the small queue forever when its head was the
  expert the caller was still using, while the main queue sat full of evictable slots.
  That is a failed admission and a dropped expert, not a slow path.
- **The trunk ring re-read layers it had already prefetched.** `claim_slot_locked`
  protected slots being read into and the slot in use, but not one holding a layer the
  prefetcher had fetched and the walk had not yet reached — so a further-ahead prefetch
  evicted the very next layer needed, which was then read again. Worst right after a
  pinned layer, which holds no ring slot and so leaves every slot looking claimable.
  Measured on the released checkpoint as **491 GB read from a 29.81 GB trunk over ten
  passes**; reproduced in `test_trunk` at ring depth 2 as 6.23 MB against a 4.72 MB
  bound, and that bound is now asserted on every ring depth.
- **…and then re-read a smaller share of them anyway, because the protection window
  measured the wrong thing.** It protected layers `walk+1 .. walk+nslot` by INDEX, which
  is only the right window when nothing is pinned: `k3_trunk_prefetch` skips pinned layers
  while scanning rather than stopping at them, so with pinned layers ahead it queues
  something many indices out that is only a few slots out, and the index window called it
  stale. It now counts the layers that would actually take a slot, which is identical
  whenever nothing is pinned and strictly wider otherwise. A run on the released
  checkpoint sat 13.7% above the bound after the first fix; a new 93-layer sweep across
  budgets and ring depths failed **40 of 100 pinned shapes**, worst 35 layers re-read at
  56 pinned with a four-slot ring, and passes all 100 with this.
- `K3Trunk` now counts reads **per layer**, and `k3_trunk_report` prints the layers that
  went over what the walk owed. The aggregate byte total could say a run went over and
  never which layers; two explanations for the 13.7% were argued from the pinned set's
  shape before this existed, and both were wrong.
- `k3_moe`'s cache-only draft path called `K3ExpertSrc::resident` unconditionally, even
  though `k3.h` documents it as optional ("may be NULL; callers must cope"), the same as
  `getmany` and `speculate` right above it in the struct. The one existing source
  (`K3Cache`) always provides it, so behaviour is unchanged; this makes the call site
  honour the contract it already promises rather than relying on there only ever being
  one implementation.
- `k3_st_open`'s directory scan left two allocations (the shard-path array's `realloc`,
  and the per-path `malloc`) unchecked, unlike every other allocation failure in the
  file: a failed `realloc` both leaked every path already collected and handed the next
  write a null pointer instead of the `fprintf`-and-refuse this file uses everywhere
  else.
- Removed an `if (0) { ... }` block in `k3_cache_init` left over from the arena
  allocation being rewritten for hugepages — the real error path already returns above
  it, so the block was unreachable.
- **CLI**: `--layers N --preset auto` sized its recurrent-state memory forecast from the
  model's full layer count rather than the truncated one actually being bound, which
  could pick a needlessly small trunk/cache split or refuse a run that would fit.
  `--gen 0` divided by zero generated tokens and printed/wrote `nan`, which is not a
  legal JSON token, into `--out`; both now guard the division. Three output-file writes
  (the `--ppl` and `--tf-check` summaries, `--dump-cache-trace`'s histogram) swallowed
  `fopen`/write failures silently instead of reporting them like every other failure
  path in this file. Removed three `--help`/`--version`/`--list-presets` branches in the
  per-option loop that could never run, because the scan at the top of `main()` already
  covers all of `argv` and returns before that loop is reached.

## [1.0.0] - 2026-08-07

Verified end to end on the full released checkpoint, and made substantially faster, with
byte-identical output preserved at every step. The first-run experience, which was broken
on a clean clone, now works.

### Added

- **`--preset auto`**: sizes the trunk and expert-cache budgets from the machine's own free
  RAM, trunk-first, so a user need not pick a preset by hand. A gigabyte given to the trunk
  is worth far more than a gigabyte of expert cache, and auto pins accordingly, capping the
  pin below the RAM ceiling after a heavy-pin regression was measured.
- **Chunk-union prefill**: a batched-prefill MoE that fetches each unique routed expert once
  per chunk instead of once per token, measured to read about half the expert bytes on a
  prompt, with the generated token bit-identical to the per-token path.
- **Conversation resume** (`--save-state` / `--load-state`): carries the recurrent state and
  KV cache to disk so a second turn resumes instead of re-reading the whole prompt, measured
  3.9x faster on turn two with identical output. Refuses to restore state from a different
  architecture.
- **`--spec N`**: speculative decode by n-gram drafting with batched greedy verification;
  output is exactly the serial greedy decode by construction.
- `--tf-check`, teacher-forced agreement over an id sequence in one sweep, for measuring
  draft quality; `tools/qdq_trunk.py` and `tools/int8_trunk.py` for deriving quantized
  trunks.

### Changed

- **Fused matmul kernels** (fp32, bf16, MXFP4): sixteen partitioned accumulators with
  explicitly fused products, taking the trunk matmul to its memory floor (about eight times
  less per-token compute) while keeping the scalar and AVX2 paths bitwise identical.
- **KDA recurrence parallelised over heads**, bit-identical to the serial form.
- `scripts/k3-doctor.sh` per-preset speed expectations refreshed to the v1.0.0 numbers, with
  the streaming presets noted as disk-bound and the resident tier as compute-bound.

### Fixed

- All shell scripts are committed executable; the first documented command no longer fails
  with Permission denied on a clean clone.
- `scripts/download-model.sh` uses the current `hf` CLI and pins an immutable revision with
  checksum verification; it no longer attempts a pip install that cannot succeed on the
  target OS, and refuses to start without free space for the checkpoint.
- `scripts/k3-doctor.sh` no longer fails a machine that can build and test the engine; the
  memory floor is a warning about running the checkpoint, not a hard stop.
- The config-refusal fixtures the docs describe now exist and are gated in `make test`,
  ctest and CI; the tokenizer leg reports NOT RUN rather than passing silently; CI runs
  `make test` rather than a hand-picked subset.
- A silent-corruption path in the MLA KV overflow and one in the single-slot trunk reader,
  both of which could emit a plausible wrong token, now abort or are prevented.
- The MXFP4 packer alignment and the tiny-checkpoint scale rule.

### Research notes, not shipped as features

- Lossless trunk compression and a quantized-self-draft hybrid were both built and measured,
  and both turned out to help only narrow regimes. The findings and prototypes are kept in
  [`docs/notes/`](docs/notes/).

## [0.1.0] - 2026-07-31

First public release.

### Added

- Full 93-layer Kimi K3 inference: 69 KDA + 24 Gated MLA layers, 896 routed experts with
  top-16 selection, SiTU-GLU, Attention Residuals, native MXFP4 expert weights.
- **Trunk streaming**, which turns the memory budget into a dial rather than a floor. The
  model runs in 8 GB and in 224 GB and produces byte-identical output at every budget
  measured in between.
- MXFP4 matmul that consumes packed nibbles directly, never materialising a dequantised
  expert.
- BPE tokenizer in C, reading the released `tiktoken.model` directly, text in, text out
  with no external step.
- Config reader that loads the checkpoint's own `config.json` and **refuses** a config it
  cannot fully understand rather than defaulting missing fields.
- Incremental decode with a KV cache and carried recurrent state, verified to produce the
  same tokens as full recompute.
- Named memory presets (`--preset laptop|desktop|workstation|server|max`) derived from
  the measured memory ladder.
- `scripts/k3-doctor.sh`, reports whether a machine can run the model, which preset
  fits, and how fast its storage is.
- `scripts/download-model.sh`, fetches the checkpoint and verifies it byte-exactly
  against the published total, because a partial download produces wrong output silently.
- Test suite that runs entirely without model weights: op fixtures, expert cache,
  safetensors reader, config reader, and end-to-end oracle gates (teacher forcing,
  greedy decode, and incremental decode).
- CI: build matrix across GCC and Clang, warnings-as-errors, ASan and UBSan, Python and
  shell lint. Tokenizer parity is built and reported but CANNOT gate on a clean
  checkout, because it needs the vocabulary that ships with the model weights; run
  `make tok` locally against a downloaded checkpoint.

### Known limitations

- No chunked prefill, so long prompts are impractical despite a 32k context ceiling.
- Greedy decoding only; no chat template; no serving layer; no vision; CPU only.

See [docs/ROADMAP.md](docs/ROADMAP.md).

[Unreleased]: https://github.com/FareedKhan-dev/kimi-k3-in-c/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/FareedKhan-dev/kimi-k3-in-c/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/FareedKhan-dev/kimi-k3-in-c/releases/tag/v0.1.0
