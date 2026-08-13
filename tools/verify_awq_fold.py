#!/usr/bin/env python3
"""verify_awq_fold.py - check that AWQ's scale fold is exact.

    python3 tools/verify_awq_fold.py <shard_dir> <bf16_trunk> [--work DIR] [--alpha A]

WHAT IS BEING CHECKED, AND WHY IT NEEDS ITS OWN TOOL

    awq_trunk.py rests on an identity: W x == (W diag(s)) (diag(s)^-1 x). The right-hand
    factor is applied by dividing the RMSNorm weight in front of the activation, so the
    identity holds only if EVERY tensor reading that activation gets the matching column
    scale. Miss one and it silently receives an input divided by s -- the run completes,
    the weights all look fine, and the model computes something else.

    No weight-space metric can see that. Reconstruction error is measured per tensor and
    each tensor is individually correct; it is the COMBINATION that is wrong.

    So: build a trunk with the scaling and the fold applied but NO quantisation, and
    compare it against the source. With no rounding in the way, the identity is exact and
    the only difference left is bf16 re-rounding of the scaled weights. Anything larger is
    a missing consumer.

    The check runs on the per-position log-probabilities from --ppl-dump rather than on
    perplexity, because perplexity is a mean over the vocabulary and washes this out. When
    this was written, omitting just the router rescale moved perplexity by 0.002 and left
    top-1 agreement at 100.00% -- while the maximum per-position log-probability delta went
    from 1.4e-06 to 8.6e-05, a factor of sixty. The delta is the instrument; the others are
    not sensitive enough to rely on.

    alpha is FORCED rather than searched. The search legitimately picks alpha=0 when the
    calibration activations are isotropic, and alpha=0 means s=1, which makes this check
    pass trivially while testing nothing.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = np.dtype([("target", "<i4"), ("top1", "<i4"),
                ("target_lp", "<f4"), ("top1_lp", "<f4"), ("entropy", "<f4")])

# bf16 keeps 8 significand bits, so re-rounding a scaled weight perturbs it by up to
# 2^-9 relative. Accumulated over a deep stack that lands well below 1e-4 nats on the
# fixtures measured; a missing consumer lands above it. The gap measured was 60x, so the
# threshold does not need to be tight to separate them.
TOL_NATS = 1e-4


def run(cmd, log):
    with open(log, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.stderr.write(f"failed: {' '.join(cmd)}\ntail of {log}:\n")
        with open(log) as f:
            sys.stderr.write("".join(f.readlines()[-20:]))
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards")
    ap.add_argument("trunk", help="the BF16 packed trunk")
    ap.add_argument("--work", default="build/awqcheck")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--suite", help="a --ppl-file suite; one is synthesised if omitted")
    ap.add_argument("--vocab", type=int, default=256, help="only used to synthesise a suite")
    ap.add_argument("--trunk-gb", default="1")
    ap.add_argument("--cache-gb", default="0.05")
    a = ap.parse_args()

    os.makedirs(a.work, exist_ok=True)
    k3 = os.path.join(REPO, "bin", "k3")
    suite = a.suite
    if not suite:
        suite = os.path.join(a.work, "suite.tsv")
        rng = np.random.default_rng(7)
        with open(suite, "w") as f:
            f.write("# synthesised by verify_awq_fold.py; ids are arbitrary\n")
            for d in range(4):
                ids = rng.integers(0, a.vocab, size=64)
                f.write(f"doc{d}\t" + ",".join(str(int(v)) for v in ids) + "\n")

    base = ["--trunk-gb", a.trunk_gb, "--cache-gb", a.cache_gb, "--ppl-file", suite]
    cal = os.path.join(a.work, "calib.bin")
    d0 = os.path.join(a.work, "src.bin")
    d1 = os.path.join(a.work, "fold.bin")
    folded = os.path.join(a.work, "trunk-fold")

    print(f"1/3 calibrating against {a.trunk}")
    run([k3, a.shards, "--trunk", a.trunk] + base + ["--calib-dump", cal,
        "--ppl-dump", d0, "--out", os.path.join(a.work, "src.json")],
        os.path.join(a.work, "src.log"))

    print(f"2/3 folding at alpha={a.alpha}, no quantisation")
    run([sys.executable, os.path.join(REPO, "tools", "awq_trunk.py"), a.trunk, folded,
         "--calib", cal, "--fold-only", "--force-alpha", str(a.alpha)],
        os.path.join(a.work, "fold.log"))

    print("3/3 re-running against the folded trunk")
    run([k3, a.shards, "--trunk", folded] + base + ["--ppl-dump", d1,
        "--out", os.path.join(a.work, "fold.json")],
        os.path.join(a.work, "fold.log2"))

    x = np.fromfile(d0, dtype=REC)
    y = np.fromfile(d1, dtype=REC)
    if len(x) != len(y):
        print(f"FAIL: {len(x)} vs {len(y)} scored positions")
        return 1

    delta = np.abs(x["target_lp"].astype(np.float64) - y["target_lp"].astype(np.float64))
    agree = float((x["top1"] == y["top1"]).mean())
    ok = delta.max() < TOL_NATS

    print()
    print(f"  positions            {len(x)}")
    print(f"  top-1 agreement      {100 * agree:.2f}%")
    print(f"  max |delta logprob|  {delta.max():.3e} nats  (tolerance {TOL_NATS:.0e})")
    print(f"  mean |delta|         {delta.mean():.3e} nats")
    print()
    if ok:
        print("FOLD VERIFIED: the scaled trunk reproduces the source to bf16 rounding,")
        print("so every consumer of every rescaled activation is in awq_trunk's GROUPS.")
    else:
        print("FOLD BROKEN: the scaled trunk does NOT reproduce the source.")
        print("Some tensor reads a rescaled activation without a matching column scale.")
        print("Check GROUPS in tools/awq_trunk.py against the call sites in k3_ops.c.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
