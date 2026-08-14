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

    TWO STAGES, because only one of them is rigorous and it is important to say which.

    Stage 1, alpha = 0. The scale is exactly 1, so the fold is the identity: norms are
    divided by 1, weight columns multiplied by 1, and a bf16 value that round-trips
    through f32 comes back bit-identical. The output must therefore match the source
    EXACTLY -- zero, not "small". This validates the whole rewrite path (tensor offsets,
    alignment, dtype promotion of the norms, the gate rescale, run padding) with no
    tolerance to argue about.

    Stage 2, alpha > 0. Now s varies per channel and the fold is still exact in fp32 --
    but the scaled weights are written back as bf16, because the container's narrow
    tensors are bf16 and a layer of fp32 matmul weights would overflow the widen area.
    So W*s is re-rounded, and that sets a NOISE FLOOR this check cannot see beneath.
    Measured on the random fixture the floor is 7.4e-04 nats, while omitting the router
    rescale entirely measured 1.9e-04. THE FLOOR IS LARGER THAN THE BUG. Stage 2 therefore
    catches gross errors only, and its tolerance is deliberately loose.

    THE REAL COMPLETENESS GATE IS THE COVERAGE AUDIT inside awq_trunk.py, which is
    structural and does not depend on any of this: a narrow tensor whose input width
    matches a rescaled activation must be declared a consumer or declared not to be one,
    and being unmentioned is refused. That is what catches an omission whose numerical
    signature is smaller than bf16 rounding -- g_proj, for one, at 2.9e-06.
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
# Stage 2 only, and loose on purpose: see the note above. bf16 re-rounding of the scaled
# weights measured 7.4e-04 nats on the random fixture, so anything tighter fails honest
# runs, and anything at all only catches order-of-magnitude breakage. Stage 1 is the
# check with no tolerance.
TOL_NATS = 5e-3


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

    print(f"1/4 calibrating against {a.trunk}")
    run([k3, a.shards, "--trunk", a.trunk] + base + ["--calib-dump", cal,
        "--ppl-dump", d0, "--out", os.path.join(a.work, "src.json")],
        os.path.join(a.work, "src.log"))
    x = np.fromfile(d0, dtype=REC)

    def fold_and_run(alpha, tag):
        out = os.path.join(a.work, f"trunk-{tag}")
        dump = os.path.join(a.work, f"{tag}.bin")
        run([sys.executable, os.path.join(REPO, "tools", "awq_trunk.py"), a.trunk, out,
             "--calib", cal, "--fold-only", "--force-alpha", str(alpha)],
            os.path.join(a.work, f"{tag}.log"))
        run([k3, a.shards, "--trunk", out] + base + ["--ppl-dump", dump,
            "--out", os.path.join(a.work, f"{tag}.json")],
            os.path.join(a.work, f"{tag}.run.log"))
        y = np.fromfile(dump, dtype=REC)
        if len(x) != len(y):
            raise SystemExit(f"{len(x)} vs {len(y)} scored positions")
        return (np.abs(x["target_lp"].astype(np.float64) - y["target_lp"].astype(np.float64)),
                float((x["top1"] == y["top1"]).mean()))

    print("2/4 stage 1: folding at alpha=0, which must be the identity")
    d_id, _ = fold_and_run(0.0, "identity")
    print(f"3/4 stage 2: folding at alpha={a.alpha}")
    d_sc, agree_sc = fold_and_run(a.alpha, "scaled")
    print("4/4 comparing")

    exact_ok = d_id.max() == 0.0
    loose_ok = d_sc.max() < TOL_NATS

    print()
    print(f"  positions                  {len(x)}")
    print(f"  stage 1  alpha=0           max delta {d_id.max():.3e} nats  (must be exactly 0)")
    print(f"  stage 2  alpha={a.alpha:<4}        max delta {d_sc.max():.3e} nats  "
          f"(tolerance {TOL_NATS:.0e}), agreement {100 * agree_sc:.2f}%")
    print()
    if exact_ok and loose_ok:
        print("FOLD VERIFIED.")
        print("  Stage 1 is exact: the rewrite path -- offsets, alignment, the norm")
        print("  division, the gate rescale -- introduces nothing at all.")
        print("  Stage 2 is within a loose bound. It does NOT prove the GROUPS table is")
        print("  complete; bf16 re-rounding of the scaled weights is larger than a missing")
        print("  consumer's signature. awq_trunk's coverage audit is what proves that.")
    elif not exact_ok:
        print("STAGE 1 FAILED, and it has no tolerance to blame.")
        print("  At alpha=0 every scale is 1, so the rewrite must reproduce the source")
        print("  byte for byte. Something in the rewrite itself is wrong: a tensor offset,")
        print("  the run padding, a dtype conversion, or the norm/gate handling.")
    else:
        print("STAGE 2 FAILED: the scaled fold is far enough out to be a real error,")
        print("  not rounding. Check GROUPS in tools/awq_trunk.py against k3_ops.c.")
    return 0 if (exact_ok and loose_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
