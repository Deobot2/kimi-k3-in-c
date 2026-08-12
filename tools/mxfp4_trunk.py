#!/usr/bin/env python3
"""mxfp4_trunk.py: rewrite a packed trunk with its matmul weights in OCP MX FP4.

    python3 tools/mxfp4_trunk.py <src_trunk_dir> <dst_trunk_dir>

WHAT THIS IS FOR, AND WHAT IT COSTS

    The engine's trunk is 108.81 GB of bf16, and it is re-read IN FULL on every token
    when streamed -- 62.40 s of a 135.8 s token on the reference machine. At MXFP4 the
    same weights are about 29 GB, which fits entirely in RAM on a 64 GB desktop. The
    per-token trunk I/O then does not shrink, it DISAPPEARS, and that is a 5-10x
    difference rather than a marginal one.

    It is also the one axis this project deliberately declined, and the reason is
    written down: Kimi K3's technical report (4.1.4) keeps every non-expert component in
    higher precision on purpose, and this repository's own measurement on 31 real
    attention tensors put post-hoc int4 at 17.4% mean relative weight error against
    0.96% for int8 (docs/data/trunk-quantisation.txt). Streaming costs time, which more
    memory buys back; rounding costs accuracy, which nothing buys back.

    So this is a FORK OF THE CONTRACT, not an optimisation. A run against a quantised
    trunk is not the released model, cannot be byte-compared against anything, and
    invalidates every bit-identity gate the test suite is built on. What replaces those
    gates is `k3 --ppl`, which measures perplexity on a held-out id sequence: run it
    against the bf16 trunk and against this one, and the difference is the whole of what
    you are buying and paying. docs/notes/compressed-trunk.md carries the procedure.

WHAT IS QUANTISED, AND WHY THE ANSWER MUST BE "ALL OR NOTHING"

    Exactly the tensors the engine reads through a matmul: 2D, bf16, more than one row.
    Everything else -- norms, biases, A_log, dt_bias, the conv kernels, the AttnRes
    projections -- is read ELEMENTWISE as fp32 and passes through untouched. They are
    well under 1% of the bytes.

    The router gate is the one exception that has to be named rather than derived. It is
    2D, bf16 and 1.18 GB per layer's worth across the model, so every size-based rule
    would take it -- but k3_router walks it with its own inline fp32 matmul and the
    engine wants it widened, so quantising it produces a trunk the binder refuses.

    The refusal is deliberate and is the reason this tool does not offer a --skip flag.
    A layer's weight format is a tag on the STRUCT, not on each tensor, so a layer that
    mixes bf16 and MXFP4 matmul weights cannot be described: whichever tag is chosen,
    the other format is read as the wrong bytes and the model runs and is wrong.

THE FORMAT
    Per row: [packed nibbles, cols/2 bytes][E8M0 scales, cols/32 bytes], so a row is
    cols*17/32 bytes and the row is self-contained -- which is what lets the engine
    describe the matrix with a single tagged pointer, exactly as it does for the int8
    draft container. Nibble order follows the checkpoint's own convention: the LOW
    nibble of each byte is the EVEN element. Reversing it yields a matrix with the right
    values in the wrong places, which every statistic would pass.

    Group size is 32, matching the routed experts, so the same kernel reads both.
"""
import argparse
import json
import os
import sys

import numpy as np

GROUP = 32
# The sixteen E2M1 values, indexed by nibble; bit 3 is the sign. Must match K3_E2M1 in
# src/core/k3_ops.c exactly -- this table and that one are the same specification.
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
# Magnitudes in code order, for nearest-value rounding.
MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

# Read elementwise as fp32 by k3_router, despite being a large 2D bf16 matrix.
DENY_SUFFIX = (".block_sparse_moe.gate.weight",)


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def quantize_mx4(f32):
    """[rows][cols] float32 -> packed bytes, one self-contained row at a time.

    The shared exponent follows the OCP MX rule: floor(log2(absmax)) minus the element
    format's maximum exponent, which is 2 for E2M1. That puts the group's largest
    magnitude in [4, 8), so the 6.0 code covers it after rounding and nothing saturates
    to zero. A group that is entirely zero gets scale 127 (2^0) and all-zero nibbles.
    """
    rows, cols = f32.shape
    assert cols % GROUP == 0, "cols %d is not a multiple of the group %d" % (cols, GROUP)
    ngrp = cols // GROUP

    g = f32.reshape(rows, ngrp, GROUP)
    absmax = np.abs(g).max(axis=2)

    with np.errstate(divide="ignore"):
        exp = np.floor(np.log2(np.where(absmax > 0, absmax, 1.0))).astype(np.int32) - 2
    exp = np.where(absmax > 0, exp, 0)
    # E8M0 is a bare biased exponent; 255 is NaN by spec, so the usable range is 0..254.
    scale_byte = np.clip(exp + 127, 0, 254).astype(np.uint8)
    scale = np.power(2.0, (scale_byte.astype(np.int32) - 127)).astype(np.float32)

    v = g / scale[:, :, None]
    sign = (v < 0).astype(np.uint8)
    a = np.abs(v)
    # Nearest E2M1 magnitude. The grid is tiny and irregular, so compare against all
    # eight rather than trying to derive an index arithmetically.
    idx = np.abs(a[..., None] - MAG[None, None, None, :]).argmin(axis=-1).astype(np.uint8)
    code = (idx | (sign << 3)).reshape(rows, cols)

    # LOW nibble = EVEN element.
    lo = code[:, 0::2]
    hi = code[:, 1::2]
    packed = (lo | (hi << 4)).astype(np.uint8)

    return np.concatenate([packed, scale_byte.astype(np.uint8)], axis=1)


def rel_error(f32, blob, cols):
    """Mean relative reconstruction error, so the tool reports what it cost."""
    rows = f32.shape[0]
    pn = cols // 2
    packed = blob[:, :pn]
    scale_byte = blob[:, pn:]
    lo = (packed & 0x0F).astype(np.uint8)
    hi = (packed >> 4).astype(np.uint8)
    code = np.empty((rows, cols), dtype=np.uint8)
    code[:, 0::2] = lo
    code[:, 1::2] = hi
    scale = np.power(2.0, (scale_byte.astype(np.int32) - 127)).astype(np.float32)
    deq = E2M1[code].reshape(rows, cols // GROUP, GROUP) * scale[:, :, None]
    deq = deq.reshape(rows, cols)
    denom = np.abs(f32).mean()
    if denom == 0:
        return 0.0
    return float(np.abs(deq - f32).mean() / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--sample-error", action="store_true",
                    help="measure reconstruction error on every quantised tensor "
                         "(slower; the number is what --ppl then has to confirm)")
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.src, "trunk.json")))
    align = int(man.get("align", 4096))
    os.makedirs(a.dst, exist_ok=True)

    src = open(os.path.join(a.src, "trunk.bin"), "rb")
    dst = open(os.path.join(a.dst, "trunk.bin"), "wb")

    out = {"n_layers": man.get("n_layers", len(man["layers"])), "align": align,
           "quant": "mxfp4", "group": GROUP, "layers": []}
    file_off = 0
    n_quant = 0
    n_pass = 0
    err_sum = 0.0
    err_n = 0
    src_bytes = 0
    dst_bytes = 0

    for li, lay in enumerate(man["layers"]):
        src.seek(lay["file_off"])
        run = src.read(lay["nbytes"])
        src_bytes += lay["nbytes"]

        # Rebuild the run tensor by tensor, in the source's own order, so anything the
        # container relies on about ordering is preserved.
        items = sorted(lay["tensors"].items(), key=lambda kv: kv[1]["off"])
        pieces = []
        tensors = {}
        at = 0
        for name, t in items:
            off, nb = int(t["off"]), int(t["nbytes"])
            shape = t.get("shape") or []
            dtype = t.get("dtype")
            raw = run[off:off + nb]

            do_q = (dtype in ("BF16", "bf16") and len(shape) == 2
                    and int(shape[0]) > 1 and int(shape[1]) % GROUP == 0
                    and not any(name.endswith(s) for s in DENY_SUFFIX)
                    and nb == int(shape[0]) * int(shape[1]) * 2)

            at = (at + 7) & ~7          # 8-byte align every tensor, as pack_trunk does
            if do_q:
                rows, cols = int(shape[0]), int(shape[1])
                f32 = bf16_to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(rows, cols)
                blob = quantize_mx4(f32)
                if a.sample_error:
                    err_sum += rel_error(f32, blob, cols)
                    err_n += 1
                b = blob.tobytes()
                assert len(b) == rows * (cols // 2 + cols // GROUP)
                tensors[name] = {"off": at, "nbytes": len(b), "dtype": "MX4",
                                 "shape": shape}
                n_quant += 1
            else:
                b = raw
                tensors[name] = {"off": at, "nbytes": nb, "dtype": dtype, "shape": shape}
                n_pass += 1
            pieces.append((at, b))
            at += len(b)

        # Pad the run to the container alignment: O_DIRECT needs offset AND length
        # aligned, and the next run starts where this one ends.
        run_len = (at + align - 1) & ~(align - 1)
        buf = bytearray(run_len)
        for off, b in pieces:
            buf[off:off + len(b)] = b
        dst.write(buf)
        dst_bytes += run_len

        out["layers"].append({"file_off": file_off, "nbytes": run_len,
                              "tensors": tensors})
        file_off += run_len
        if (li + 1) % 10 == 0 or li + 1 == len(man["layers"]):
            print("  layer %d/%d, %.2f GB written"
                  % (li + 1, len(man["layers"]), dst_bytes / 1e9), flush=True)

    src.close()
    dst.close()
    with open(os.path.join(a.dst, "trunk.json"), "w") as f:
        json.dump(out, f)

    print("\nquantised %d tensors, passed %d through untouched" % (n_quant, n_pass))
    print("trunk %.2f GB -> %.2f GB (%.2fx)"
          % (src_bytes / 1e9, dst_bytes / 1e9,
             src_bytes / dst_bytes if dst_bytes else 0.0))
    if err_n:
        print("mean relative weight reconstruction error: %.4f over %d tensors"
              % (err_sum / err_n, err_n))
    print("\nTHIS TRUNK IS NOT THE RELEASED MODEL. Every bit-identity gate in the test\n"
          "suite is void against it. Measure it instead:\n"
          "    k3 <shards> --trunk <bf16_trunk> --ids ... --ppl\n"
          "    k3 <shards> --trunk %s --ids ... --ppl\n"
          "and compare the two perplexities." % a.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
