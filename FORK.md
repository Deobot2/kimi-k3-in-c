# This fork

Orientation for anyone landing here — including future me.

## What this is

A fork of [FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c),
taken at **v1.0.0** (`ff11dce`, 2026-08-07).

**Almost all of this repository is upstream's work** — the engine, the tokenizer, the test
suite, the documentation, and the findings in both research notes. This fork adds
**17 commits** dated 2026-08-12 to 2026-08-14 (44 files, +7539 / −665). Those commits are
AI-assisted; `git log` shows the authorship.

Nothing here contains model weights. See `NOTICE` for third-party components and licensing.

## The short version

Every change in this fork aims at one fact: **the engine is I/O bound by a factor of four.**
So each one either spends idle CPU to avoid a disk read, or avoids needing the bytes at all.

The more useful property is the second one: **this fork measured its own work and walked back
the parts that did not hold.** Four of the seventeen commits exist to retract or qualify one
of the other thirteen. That is the part worth reading.

## What changed, by area

| area | commits | what it does |
|---|---|---|
| **MLA latent cache** | `cf154bb` | caches the 576-float latent instead of expanded per-head k/v — 55.3 KB per position instead of 2.37 MB, so 131k context costs 7.25 GB rather than 310 GB. Not bit-identical to the expanded path; gated by GATE 4 |
| **Trunk streaming** | `9cd7520`, `df6fd7e`, `7eacd01` | largest-first pinning, a deeper ring, io_uring reads (8 MB requests, 16 outstanding, raw syscalls, no liburing), and two fixes stopping the reader re-reading what the prefetcher already fetched |
| **Expert cache** | `9106e67`, `6415d7a`, `fcb7ae4` | S3-FIFO replacing LRU, then both cache and `--preset auto` sized against one token's measured working set (1,472 slots, 25.8 GB) rather than an assumed figure |
| **Trunk quantisation** | `2a9f9b5`, `961419b`, `bb4f5d7`, `ee44444`, `b35c333` | an MXFP4 container behind a flag (108.81 GB → ~29 GB), `--ppl` as the gate that replaces bit-identity, then AWQ activation-aware 4-bit with the fold verified |
| **Eval harness** | `627d94f` | a two-leg baseline harness over a committed 8-document corpus, because perplexity alone missed a real failure (below) |
| **Fixes** | `cb489a1`, `eb11c88`, `e91969b` | composing the latent cache with a quantised trunk, a correctly packed MXFP4 trunk being refused as mixed-format, and two test report lines that said the wrong thing |
| **Docs** | `60f6184` | records the seven changes and labels which claims are **measured** and which are **argued from mechanism** |

## What this fork retracted or qualified

| commit | what it walked back | why |
|---|---|---|
| `fcb7ae4` | **turned the speculative prefetch OFF** — a feature added five commits earlier in `9106e67` | measured, it read a full token's worth of experts (43.8 GB/token, against a 25.8 GB maximum if nothing were cached) to avoid re-reading 30.3% of them. 18 GB per token of waste on a run with no spare bandwidth |
| `961419b` | added a warning to its own headline number | the +13.0% perplexity figure came from 20 scored positions. Per-token NLL varies by whole nats, so the error on that mean is comparable to the gap it reports |
| `627d94f` | established that `--ppl` alone is not a sufficient gate | the MXFP4 trunk scored a modest-looking +13% perplexity **while its generations collapsed into an eight-token loop**. Perplexity is teacher-forced, so it is blind to degeneration by construction, at any corpus size |
| `b35c333` | "stop overstating what the fold check proves" | the AWQ fold check was described as proving more than it did. Same commit fixed a crash where fixtures writing norms as fp32 had hidden that the real checkpoint stores them bf16 |

## Where the failures are written down

| file | what it records |
|---|---|
| [`docs/notes/compressed-trunk.md`](docs/notes/compressed-trunk.md) | **Part 1:** lossless Huffman trunk compression — upstream, shelved on decode speed. **Part 2:** MXFP4 quantisation — this fork, shipped behind a flag |
| [`docs/notes/int8-draft-container.md`](docs/notes/int8-draft-container.md) | the int8 speculative-decoding draft — upstream, built, correct, and abandoned. This fork changed no finding in it |
| [`CHANGELOG.md`](CHANGELOG.md) `[Unreleased]` | the seven changes, each labelled measured or argued |
| [`docs/data/`](docs/data/) | the raw measurement output behind every number in the docs |
| [`benchmarks/eval/`](benchmarks/eval/) | the baseline harness and its committed corpus |

## Picking it back up

```sh
make                # build (auto-detects platform); make portable for generic AVX2
make test           # under a minute, no model weights needed
make help           # all targets
```

The full test suite runs **weightless** — the checkpoint is 1.56 TB, so correctness is
verifiable without it. CI builds on Ubuntu (gcc + clang) and macOS, and asserts the suite
is not silently empty.

## Known loose ends

- **GitHub Actions has never run on this fork.** Actions are disabled on forks by default,
  which is why the stale test-count assertion below went unnoticed until 2026-09-17. The
  suite is green locally on this commit — `make ARCH="-mavx2 -mfma" test` reports
  **27 passed, 0 failed, 0 skipped**, plus the scale test, the oracle verdict and the ruff
  and compileall legs — but no GitHub runner has ever confirmed it.
- The README's CI badge and the CHANGELOG's compare links point at **upstream**, not at this
  fork, so the badge shows upstream's build status rather than this fork's.
- `docs/images/patrick_pray.png` is no longer referenced by anything.
- `[Unreleased]` in the CHANGELOG has never been cut as a release.
