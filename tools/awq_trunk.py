#!/usr/bin/env python3
"""awq_trunk.py: rewrite a packed trunk in MXFP4, activation-aware.

    k3 <shards> --trunk <bf16_trunk> --ppl-file suite.tsv --calib-dump calib.bin
    python3 tools/awq_trunk.py <bf16_trunk> <out_trunk> --calib calib.bin

WHAT THIS DOES THAT tools/mxfp4_trunk.py DOES NOT

    Plain MXFP4 rounds every weight to the same grid regardless of what it multiplies.
    That is the wrong objective: what reaches the next layer is W x, so a column of W
    that meets a large input contributes proportionally more output error for the same
    rounding step. Measured on this model, plain 4-bit on the trunk raised perplexity
    from 97.40 to 110.05 and collapsed generation into an eight-token loop.

    Activation-aware quantisation (Lin et al., AWQ) exploits an exact identity:

        W x  ==  (W diag(s)) (diag(s)^-1 x)

    for any positive s. Scaling up the columns that meet large activations makes their
    rounding error smaller relative to their own magnitude, and the inverse scale on the
    input puts the arithmetic back. Nothing is approximated by the identity itself; the
    only approximation is still the 4-bit rounding, now spent where it costs least.

WHERE THE INVERSE SCALE GOES, WHICH IS THE WHOLE DESIGN QUESTION

    diag(s)^-1 has to be applied to the input somewhere, and this engine's kernels take
    no per-channel input scale. Adding one would change the format, the kernel and every
    bit-identity claim about the MXFP4 path.

    It does not need one. Every activation this tool scales has an RMSNorm immediately in
    front of it, and RMSNorm ends in an elementwise multiply by a weight vector:

        x = (h / rms(h)) * g        so    x / s = (h / rms(h)) * (g / s)

    Dividing the norm weight by s is therefore EXACTLY equivalent to dividing the input
    by s, and it costs nothing at run time. The released checkpoint stores those norm
    weights as BF16 and the binder widens them, so this tool writes the divided ones back
    as F32 to keep the fold exact rather than re-rounding it -- about 16 MB across the
    model. THE OUTPUT IS OTHERWISE ORDINARY MXFP4: the engine reads it with the kernel it
    already has, and this tool and mxfp4_trunk.py produce interchangeable containers.

    That constraint is also what fixes the grouping. Every tensor reading the same norm
    output must share one s -- there is only one norm weight to divide. The engine emits
    exactly those groups (K3_CAL_* in k3.h), which is why calibration is collected per
    norm rather than per tensor.

WHAT IS NOT COVERED, STATED PLAINLY

    A matmul whose input has no norm in front of it has nowhere to fold the inverse, and
    is quantised PLAIN, exactly as mxfp4_trunk.py would: o_proj in both attention
    families, f_b_proj, the shared expert down_proj, and the dense down_proj. On the
    released shape that is about 20% of KDA weight bytes and 24% of MoE weight bytes.

    Handling them needs a different mechanism (folding into the producing tensor's output
    rows, which is only sound when that tensor is not itself quantised). Not done here.
    The per-group report below says exactly which tensors got a scale and which did not.

THE ROUTER IS RESCALED TOO
    block_sparse_moe.gate reads the same post-attention norm output as down_proj and the
    shared experts. It is never quantised, but it still sees the rescaled input, so its
    columns are multiplied by s. Omitting that would change which experts get routed, and
    no weight-error metric would show it. Unlike the norms it keeps its source dtype: it
    is 1.18 GB as bf16 across the model, and promoting it would add 4% to the output trunk
    to remove a 0.2% rounding error on a matrix feeding a top-k over 896 experts.

CHOOSING s
    s_j = mean|x_j| ** alpha, normalised to unit geometric mean so the fold does not
    drift the norm weight's magnitude. alpha is searched over a grid per group,
    minimising the activation-weighted reconstruction error

        sum_ij  (Wq_ij/s_j - W_ij)^2 * E[x_j^2]

    which is the diagonal approximation to the layer's output MSE: exact for the
    per-column error, and assuming rounding errors are uncorrelated across channels.
    alpha = 0 is in the grid, so a group where scaling does not help keeps s = 1 and this
    tool degrades to plain MXFP4 rather than to something worse.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

from _bf16 import bf16_to_f32, f32_to_bf16

GROUP = 32
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

DENY_SUFFIX = (".block_sparse_moe.gate.weight",)

# Slot indices, mirroring the K3_CAL_* enum in include/k3/k3.h.
CAL_IN, CAL_POST, CAL_QA, CAL_KVA, CAL_LAT = range(5)
SLOT_NAME = ["in", "post", "qa", "kva", "lat"]

# For each calibration slot: the norm weight the inverse scale folds into, the quantised
# tensors whose columns get scaled, and any fp32 tensors that read the same activation
# and must be rescaled exactly. Suffixes are matched against the tensor name.
#
# Correctness depends on this table being COMPLETE: a consumer of the same activation
# that is left unscaled sees an input divided by s and silently computes something else.
# It was built by reading the call sites in src/core/k3_ops.c, and the engine's own
# observation points are placed on the same four norms.
GROUPS = {
    CAL_IN: {
        "norm": ".input_layernorm.weight",
        "quant": (".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                  ".self_attn.v_proj.weight", ".self_attn.g_proj.weight",
                  ".self_attn.b_proj.weight", ".self_attn.f_a_proj.weight",
                  ".self_attn.q_a_proj.weight", ".self_attn.kv_a_proj_with_mqa.weight"),
        "exact": (),
    },
    CAL_POST: {
        "norm": ".post_attention_layernorm.weight",
        "quant": (".block_sparse_moe.routed_expert_down_proj.weight",
                  ".block_sparse_moe.shared_experts.gate_proj.weight",
                  ".block_sparse_moe.shared_experts.up_proj.weight",
                  ".mlp.gate_proj.weight", ".mlp.up_proj.weight"),
        "exact": (".block_sparse_moe.gate.weight",),
    },
    CAL_QA:  {"norm": ".self_attn.q_a_layernorm.weight",
              "quant": (".self_attn.q_b_proj.weight",), "exact": ()},
    CAL_KVA: {"norm": ".self_attn.kv_a_layernorm.weight",
              "quant": (".self_attn.kv_b_proj.weight",), "exact": ()},
    CAL_LAT: {"norm": ".block_sparse_moe.routed_expert_norm.weight",
              "quant": (".block_sparse_moe.routed_expert_up_proj.weight",), "exact": ()},
}

# Tensors whose column count can COINCIDE with a slot's activation width without their
# being a consumer of it. Everything narrow that matches a width and is in neither this
# list nor GROUPS is treated as an unaudited consumer and refused -- see audit_coverage.
NOT_CONSUMERS = (
    ".self_attn.o_proj.weight",                             # attention output, no norm
    ".self_attn.f_b_proj.weight",                           # reads f_a's output
    ".self_attn.q_b_proj.weight", ".self_attn.kv_b_proj.weight",   # their own slots
    ".block_sparse_moe.shared_experts.down_proj.weight",    # reads the SiTU-GLU output
    ".block_sparse_moe.routed_expert_up_proj.weight",       # its own slot
    ".mlp.down_proj.weight",                                # reads the dense GLU output
)


def audit_coverage(lay, stats_for_layer):
    """Every narrow tensor whose input width matches a rescaled activation must be either
    a declared consumer or a declared non-consumer. Nothing may be merely unmentioned.

    THIS IS THE CHECK THAT MATTERS, and it exists because the numerical one is not
    sensitive enough to be relied on. Omitting the router rescale moves the folded model's
    log-probabilities by 1.9e-04 nats, which a comparison against the source catches
    easily. Omitting g_proj moves them by 2.9e-06 against a bf16 re-rounding floor of
    1.4e-06 -- a factor of two, which is not a gate. Both are equally wrong; only one is
    visible numerically, and which one is which depends on the model's own dynamics rather
    than on anything about the bug.

    So completeness is checked structurally instead. Returns a list of complaints."""
    # The question is whether a tensor is accounted for AT ALL, not whether it belongs to
    # a particular slot -- GROUPS already answers the second. Asking per slot mis-fires
    # whenever two slots share a width, which `in` and `post` always do: both are the
    # hidden size, so every tensor reading either one matches both.
    declared = set(NOT_CONSUMERS)
    for spec in GROUPS.values():
        declared |= set(spec["quant"]) | set(spec["exact"])

    widths = {st[0].size for st in stats_for_layer if st is not None}
    bad = []
    for name, t in lay["tensors"].items():
        if t.get("dtype") not in ("BF16", "bf16"):
            continue
        shape = [int(x) for x in (t.get("shape") or [])]
        if len(shape) != 2 or shape[0] <= 1 or shape[1] not in widths:
            continue
        if not any(name.endswith(s) for s in declared):
            bad.append(f"{name.split('layers.')[-1]} reads {shape[1]} input channels, "
                       "which matches a rescaled activation, and is in neither list")
    return bad


# ------------------------------------------------------------------ formats
def load_wide(raw, shape, dtype):
    """A wide (elementwise) tensor, whatever dtype the container holds it in.

    THE RELEASED CHECKPOINT STORES THESE AS BF16. k3_bind widens them to fp32 at bind
    time, which is what "reads them elementwise as fp32" means -- it is a statement about
    the kernel, not about the bytes. Both fixture generators in tools/ write them as F32,
    so assuming F32 here worked on every fixture and crashed on the first real trunk."""
    if dtype in ("BF16", "bf16"):
        return bf16_to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(shape)
    return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()


def quantize_mx4(f32):
    """[rows][cols] float32 -> packed MXFP4 bytes. Identical rule to mxfp4_trunk.py:
    shared exponent floor(log2(absmax)) - 2, low nibble = even element."""
    rows, cols = f32.shape
    assert cols % GROUP == 0, f"cols {cols} is not a multiple of {GROUP}"
    g = f32.reshape(rows, cols // GROUP, GROUP)
    absmax = np.abs(g).max(axis=2)
    with np.errstate(divide="ignore"):
        exp = np.floor(np.log2(np.where(absmax > 0, absmax, 1.0))).astype(np.int32) - 2
    exp = np.where(absmax > 0, exp, 0)
    scale_byte = np.clip(exp + 127, 0, 254).astype(np.uint8)
    scale = np.power(2.0, (scale_byte.astype(np.int32) - 127)).astype(np.float32)
    v = g / scale[:, :, None]
    sign = (v < 0).astype(np.uint8)
    idx = np.abs(np.abs(v)[..., None] - MAG[None, None, None, :]).argmin(axis=-1).astype(np.uint8)
    code = (idx | (sign << 3)).reshape(rows, cols)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return np.concatenate([packed, scale_byte], axis=1)


def dequantize_mx4(blob, cols):
    rows = blob.shape[0]
    pn = cols // 2
    packed, scale_byte = blob[:, :pn], blob[:, pn:]
    code = np.empty((rows, cols), dtype=np.uint8)
    code[:, 0::2] = packed & 0x0F
    code[:, 1::2] = packed >> 4
    scale = np.power(2.0, (scale_byte.astype(np.int32) - 127)).astype(np.float32)
    return (E2M1[code].reshape(rows, cols // GROUP, GROUP) * scale[:, :, None]).reshape(rows, cols)


# ------------------------------------------------------------------ calibration
def load_calib(path):
    """-> stats[layer][slot] = (mean_abs[n], mean_sq[n]), or None where unobserved."""
    with open(path, "rb") as f:
        if f.read(8) != b"K3CALIB1":
            raise SystemExit(f"{path} is not a k3 calibration dump")
        n_layers, n_slots = struct.unpack("<ii", f.read(8))
        out = [[None] * n_slots for _ in range(n_layers)]
        for L in range(n_layers):
            for t in range(n_slots):
                (n,) = struct.unpack("<i", f.read(4))
                (cnt,) = struct.unpack("<q", f.read(8))
                if n == 0:
                    continue
                a = np.frombuffer(f.read(8 * n), dtype="<f8").copy()
                q = np.frombuffer(f.read(8 * n), dtype="<f8").copy()
                if cnt <= 0:
                    continue
                out[L][t] = (a / cnt, q / cnt)
    return out


def choose_scale(mats, mean_abs, mean_sq, grid):
    """Search alpha; return (s, alpha, loss_ratio_vs_plain).

    The loss is the activation-weighted squared reconstruction error summed over every
    tensor in the group, which is what makes the choice a joint one: these tensors share
    a norm and therefore cannot have separate scales."""
    base = np.maximum(mean_abs.astype(np.float64), 1e-12)
    best = (None, 0.0, np.inf)
    plain = None
    for alpha in grid:
        if alpha == 0.0:
            s = np.ones_like(base)
        else:
            s = base ** alpha
            # Unit geometric mean. Without it, alpha changes the overall magnitude of s
            # and therefore of the folded norm weight, which is harmless arithmetically
            # but makes the losses at different alpha incomparable.
            s = s / np.exp(np.log(s).mean())
            s = np.clip(s, 1e-4, 1e4)
        loss = 0.0
        for W in mats:
            Wq = dequantize_mx4(quantize_mx4((W * s[None, :]).astype(np.float32)), W.shape[1])
            d = Wq.astype(np.float64) / s[None, :] - W.astype(np.float64)
            loss += float((d * d * mean_sq[None, :]).sum())
        if alpha == 0.0:
            plain = loss
        if loss < best[2]:
            best = (s, alpha, loss)
    ratio = best[2] / plain if plain and plain > 0 else 1.0
    return best[0], best[1], ratio


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--calib", required=True, help="from k3 --calib-dump")
    ap.add_argument("--grid", default="0,0.25,0.5,0.75,1.0",
                    help="alpha values to search (0 means plain MXFP4)")
    ap.add_argument("--report", help="write the per-group choices as JSON")
    ap.add_argument("--force-alpha", type=float,
                    help="skip the search and use this alpha everywhere. For TESTING "
                         "the fold: the search picks alpha=0 whenever activations are "
                         "isotropic, which makes s=1 and any check of the fold vacuous")
    ap.add_argument("--fold-only", action="store_true",
                    help="apply the scaling and the norm fold but keep BF16 -- no "
                         "quantisation at all. The result must reproduce the source "
                         "model to bf16 rounding, which is the direct test that the "
                         "GROUPS table names every consumer of every rescaled "
                         "activation. A missing consumer is silent in every other check")
    a = ap.parse_args()

    grid = [float(x) for x in a.grid.split(",")]
    if 0.0 not in grid:
        # Keeping 0 in the grid is what guarantees this tool never does worse than
        # mxfp4_trunk.py on any group: the fallback is always available to the search.
        grid = [0.0] + grid
        print("note: alpha=0 added to the grid so plain MXFP4 stays reachable")

    stats = load_calib(a.calib)
    man = json.load(open(os.path.join(a.src, "trunk.json")))
    align = int(man.get("align", 4096))
    if len(stats) != len(man["layers"]):
        raise SystemExit(f"calibration has {len(stats)} layers, trunk has "
                         f"{len(man['layers'])}; they are not the same model")
    os.makedirs(a.dst, exist_ok=True)

    src = open(os.path.join(a.src, "trunk.bin"), "rb")
    dst = open(os.path.join(a.dst, "trunk.bin"), "wb")
    out = {"n_layers": man.get("n_layers", len(man["layers"])), "align": align,
           "group": GROUP, "awq": True, "layers": []}
    if not a.fold_only:
        out["quant"] = "mxfp4"

    file_off = 0
    n_quant = n_pass = n_scaled = 0
    src_bytes = dst_bytes = 0
    report = []
    alpha_hist = {}

    for li, lay in enumerate(man["layers"]):
        src.seek(lay["file_off"])
        run = src.read(lay["nbytes"])
        src_bytes += lay["nbytes"]
        items = sorted(lay["tensors"].items(), key=lambda kv: kv[1]["off"])

        def load(name, _lay=lay, _run=run):
            t = _lay["tensors"][name]
            raw = _run[int(t["off"]):int(t["off"]) + int(t["nbytes"])]
            sh = [int(x) for x in t["shape"]]
            if t["dtype"] in ("BF16", "bf16"):
                return bf16_to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(sh)
            if t["dtype"] in ("F32", "f32"):
                return np.frombuffer(raw, dtype="<f4").reshape(sh).copy()
            return None

        # ---- audit BEFORE deciding anything ----
        complaints = audit_coverage(lay, stats[li])
        if complaints:
            print(f"\nREFUSING: layer {li} has tensors this tool cannot account for.\n"
                  "A narrow tensor whose input width matches a rescaled activation is\n"
                  "either a consumer (and must be scaled) or it is not (and must be\n"
                  "declared). Silently leaving one out produces a model that runs and is\n"
                  "wrong. Add it to GROUPS or to NOT_CONSUMERS in this file.\n",
                  file=sys.stderr)
            for c in complaints:
                print("  " + c, file=sys.stderr)
            return 2

        # ---- decide a scale per group, from this layer's own activations ----
        scale_of = {}          # tensor name -> s over its input dimension
        norm_div = {}          # norm tensor name -> s to divide by
        for slot, spec in GROUPS.items():
            st = stats[li][slot] if li < len(stats) else None
            if st is None:
                continue
            mean_abs, mean_sq = st
            norm = next((n for n in lay["tensors"] if n.endswith(spec["norm"])), None)
            if norm is None:
                continue
            quant_names = [n for n in lay["tensors"]
                           if any(n.endswith(sfx) for sfx in spec["quant"])]
            mats = []
            for n in quant_names:
                W = load(n)
                # The scale runs over the INPUT dimension, which is the column axis, so
                # a tensor whose columns do not match the observed activation width is
                # not a consumer of it and must be left alone.
                if W is not None and W.ndim == 2 and W.shape[1] == mean_abs.size:
                    mats.append((n, W))
            if not mats:
                continue
            if a.force_alpha is not None:
                s, alpha, ratio = choose_scale([m for _, m in mats], mean_abs, mean_sq,
                                               [a.force_alpha])
            else:
                s, alpha, ratio = choose_scale([m for _, m in mats], mean_abs, mean_sq, grid)
            alpha_hist[alpha] = alpha_hist.get(alpha, 0) + 1
            for n, _ in mats:
                scale_of[n] = s
            for sfx in spec["exact"]:
                for n in lay["tensors"]:
                    if n.endswith(sfx):
                        scale_of[n] = s
            norm_div[norm] = s
            n_scaled += len(mats)
            report.append({"layer": li, "slot": SLOT_NAME[slot], "alpha": alpha,
                           "err_ratio": round(ratio, 4),
                           "tensors": [n.rsplit(".", 2)[-2] for n, _ in mats]})

        # ---- rebuild the run ----
        pieces, tensors, at = [], {}, 0
        for name, t in items:
            off, nb = int(t["off"]), int(t["nbytes"])
            shape = [int(x) for x in (t.get("shape") or [])]
            dtype = t.get("dtype")
            raw = run[off:off + nb]
            at = (at + 7) & ~7

            do_q = (dtype in ("BF16", "bf16") and len(shape) == 2
                    and shape[0] > 1 and shape[1] % GROUP == 0
                    and not any(name.endswith(s) for s in DENY_SUFFIX)
                    and nb == shape[0] * shape[1] * 2)

            if name in norm_div:
                # The other half of the identity; the run is wrong without it.
                #
                # WRITTEN AS F32 even when the source was BF16. g/s re-rounded to bf16
                # carries ~0.2% relative error, and this tool's entire claim is that the
                # fold is EXACT while the quantisation is the only approximation. Keeping
                # that claim costs 4 bytes per hidden channel per norm -- about 16 MB
                # across the model, against a 30 GB trunk. The binder widens whatever
                # dtype it finds, so an F32 wide tensor needs no widening at all.
                g = load_wide(raw, shape, dtype)
                b = (g / norm_div[name].astype(np.float32)).astype(np.float32).tobytes()
                tensors[name] = {"off": at, "nbytes": len(b), "dtype": "F32", "shape": shape}
                n_pass += 1
            elif do_q:
                W = bf16_to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(shape)
                s = scale_of.get(name)
                if s is not None:
                    W = (W * s[None, :].astype(np.float32)).astype(np.float32)
                if a.fold_only:
                    b = f32_to_bf16(W).tobytes()
                    tensors[name] = {"off": at, "nbytes": len(b), "dtype": dtype,
                                     "shape": shape}
                    n_pass += 1
                else:
                    b = quantize_mx4(W).tobytes()
                    assert len(b) == shape[0] * (shape[1] // 2 + shape[1] // GROUP)
                    tensors[name] = {"off": at, "nbytes": len(b), "dtype": "MX4",
                                     "shape": shape}
                    n_quant += 1
            elif name in scale_of:
                # A wide consumer of a rescaled activation: the router gate.
                #
                # This one keeps its SOURCE dtype. It is 1.18 GB across the model as bf16,
                # so promoting it to F32 would add that much again to a ~30 GB trunk -- 4%,
                # to remove a 0.2% rounding error on a matrix whose output is fed to a
                # top-k over 896 experts. The norms above are 16 MB and get the exact
                # treatment; this one does not earn it.
                W = load_wide(raw, shape, dtype)
                Wv = (W * scale_of[name][None, :].astype(np.float32)).astype(np.float32)
                b = (f32_to_bf16(Wv) if dtype in ("BF16", "bf16") else Wv).tobytes()
                tensors[name] = {"off": at, "nbytes": len(b), "dtype": dtype, "shape": shape}
                n_pass += 1
            else:
                b = raw
                tensors[name] = {"off": at, "nbytes": nb, "dtype": dtype, "shape": shape}
                n_pass += 1
            pieces.append((at, b))
            at += len(b)

        run_len = (at + align - 1) & ~(align - 1)
        buf = bytearray(run_len)
        for o, b in pieces:
            buf[o:o + len(b)] = b
        dst.write(buf)
        dst_bytes += run_len
        out["layers"].append({"file_off": file_off, "nbytes": run_len, "tensors": tensors})
        file_off += run_len
        if (li + 1) % 10 == 0 or li + 1 == len(man["layers"]):
            print(f"  layer {li + 1}/{len(man['layers'])}, {dst_bytes / 1e9:.2f} GB written",
                  flush=True)

    src.close()
    dst.close()
    with open(os.path.join(a.dst, "trunk.json"), "w") as f:
        json.dump(out, f)
    if a.report:
        with open(a.report, "w") as f:
            json.dump(report, f, indent=2)

    if a.fold_only:
        print(f"\nFOLD ONLY: no quantisation. {n_scaled} tensors rescaled and their norms "
              f"divided,\n{n_pass - n_scaled} passed through unchanged. This trunk must "
              "reproduce the source model.")
    else:
        print(f"\nquantised {n_quant} tensors ({n_scaled} activation-aware, "
              f"{n_quant - n_scaled} plain), passed {n_pass} through")
    print(f"trunk {src_bytes / 1e9:.2f} GB -> {dst_bytes / 1e9:.2f} GB "
          f"({src_bytes / dst_bytes if dst_bytes else 0:.2f}x)")
    print("alpha chosen: " + ", ".join(f"{k}={v}" for k, v in sorted(alpha_hist.items())))
    if report:
        gain = np.mean([r["err_ratio"] for r in report])
        print(f"activation-weighted reconstruction error: {gain:.4f}x plain MXFP4 "
              f"(mean over {len(report)} groups; <1 is better)")
        print("  This is a WEIGHT-space number and it is not the measurement. It says the")
        print("  search worked, not that the model did. Only the harness says that.")
    print("\nTHIS TRUNK IS NOT THE RELEASED MODEL. Every bit-identity gate is void\n"
          "against it. Measure it:\n"
          f"    python3 benchmarks/eval/run_eval.py --model <shards> --trunk {a.dst} \\\n"
          "        --label awq4 --out results/awq4 --baseline results/bf16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
