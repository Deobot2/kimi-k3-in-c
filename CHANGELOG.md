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
  on both read paths.

### Changed

- **Expert cache replacement is S3-FIFO, not LRU.** The project's own simulator put 25.5
  points between LRU and Belady at 64 GB, and its own conclusion was that the lever is the
  policy rather than the size. Small FIFO, main FIFO, ghost queue; uniform object size
  makes it the friendliest possible case. `K3_CACHE_POLICY=lru` restores the old policy on
  the same binary, and `tools/sim_cache.py` reports both.
- **Speculative expert prefetch.** About 90% of expert requests in a real trace are
  repeats, so on entering layer L the cache starts reads for what that layer wanted for
  the previous token, on a background thread, before the router has run. `K3_NOSPEC=1`
  disables it; it also refuses to start when the cache cannot hold two tokens' working
  set.
- **Trunk layers are pinned largest-first, not as a prefix.** Which layers you pin is
  byte-neutral for the traffic avoided but not for the ring, whose uniform slot must hold
  the largest layer still streaming — prefix pinning took the 2.34 GB dense layer first
  and then kept paying for a slot sized to it. `K3_PIN_PREFIX=1` restores the old order.
- `test_cache` now runs its whole battery under both replacement policies, and adds a
  section that drives the speculation thread the way `k3_moe` does.
- Saved state records the KV layout and window geometry and refuses a mismatch: the two
  caches hold different tensors of the same float count, so restoring one as the other
  would be fluent and wrong. State version 1 → 2.

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
