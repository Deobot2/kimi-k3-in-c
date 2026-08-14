#!/usr/bin/env python3
"""make_random_checkpoint.py - a loadable checkpoint of RANDOM weights, numpy only.

    python3 tools/make_random_checkpoint.py <out_dir> [--layers N] [--hidden N] [--seed N]

WHAT THIS IS AND, MUCH MORE IMPORTANTLY, WHAT IT IS NOT

    It is a checkpoint that `bin/k3` will load, bind, and run to completion. Every tensor
    the binder asks for is present, at the right name, shape and dtype, and the experts
    are real MXFP4 with real per-32 scales.

    It is NOT A MODEL. The weights are gaussian noise. Its perplexity is approximately
    the vocabulary size, its generations are uniform noise, and NO NUMBER MEASURED
    AGAINST IT MEANS ANYTHING ABOUT QUALITY. It exists so that machinery which is
    otherwise unreachable without a 110 GB download -- the evaluation harness, the
    streaming trunk, a quantiser's output path -- can be exercised end to end. That is a
    plumbing test, and plumbing tests are exactly the ones this repository has repeatedly
    found it was missing.

    tools/make_tiny_checkpoint.py is the one to use when the numbers must mean something:
    it builds a real (if tiny) trained-shaped model through k3_ref and ships reference
    logits to score against. It needs torch. This does not, which is the entire point --
    a smoke test that cannot run without a 2 GB dependency does not get run.

    The engine prints a banner when it loads one of these, so a log from a random
    checkpoint cannot be mistaken for a log from the real one.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

PRE = "language_model.model."
GROUP = 32


# ------------------------------------------------------------------ safetensors
def write_safetensors(path, tensors):
    header, blobs, off = {}, [], 0
    dtmap = {np.dtype("uint8"): "U8", np.dtype("uint16"): "BF16", np.dtype("float32"): "F32"}
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        raw = arr.tobytes()
        header[name] = {"dtype": dtmap[arr.dtype], "shape": list(arr.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)
    return off + len(hj) + 8


def bf16(a):
    """Round-to-nearest-even is what the checkpoint uses; truncation biases toward zero
    and would make this fixture's weights subtly different from a real one's for no
    reason. The engine widens bf16 by shifting left 16, so the stored value must be the
    correctly rounded one."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


# ------------------------------------------------------------------ MXFP4
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def mxfp4_quant(w):
    """OCP MX FP4: per group of 32, one E8M0 (power-of-two) scale and 32 E2M1 nibbles.

    The LOW nibble of each byte is the EVEN element, matching the released checkpoint.
    Reversing it produces a matrix with the right values in the wrong places, which every
    summary statistic passes and every output fails."""
    rows, cols = w.shape
    assert cols % GROUP == 0, f"{cols} is not a multiple of {GROUP}"
    g = w.reshape(rows, cols // GROUP, GROUP).astype(np.float32)

    amax = np.abs(g).max(axis=2)
    # Largest representable magnitude is 6.0, so the scale exponent is chosen to bring
    # amax just inside it. amax == 0 would take log2(0); those groups get exponent 0.
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.where(amax > 0, amax, 1.0) / 6.0)) + 1
    e = np.clip(e, -127, 127).astype(np.int32)
    e = np.where(amax > 0, e, 0)
    scale = np.exp2(e.astype(np.float32))

    q = g / scale[:, :, None]
    idx = np.abs(q)[..., None] - np.abs(E2M1[:8])[None, None, None, :]
    nib = np.abs(idx).argmin(axis=3).astype(np.uint8)
    nib = np.where(q < 0, nib | 0x8, nib).astype(np.uint8)
    # -0.0 is representable but pointless; fold it onto +0.0 so two encodings of zero do
    # not appear in a fixture whose whole job is to be compared byte for byte.
    nib = np.where(nib == 0x8, 0, nib)

    flat = nib.reshape(rows, cols)
    packed = (flat[:, 0::2] | (flat[:, 1::2] << 4)).astype(np.uint8)
    scales = (e + 127).astype(np.uint8)          # E8M0 bias 127
    return packed, scales


# ------------------------------------------------------------------ config
def make_config(hidden, layers, vocab, seed):
    """Shapes small enough to build in seconds, structurally faithful in every way the
    binder and the kernels care about: a dense layer 0, both attention families, MoE with
    shared experts, AttnRes blocks, and every narrow width a multiple of 32."""
    n_kda_heads = 4
    # A MULTIPLE OF 32, always. f_b_proj is [heads*dim][kda_head_dim], so this value is
    # its column count -- and a narrow tensor whose columns are not a multiple of the
    # MXFP4 group cannot be quantised, which leaves its layer mixing BF16 and MX4 and the
    # binder refuses the whole trunk. The released model uses 128. A fixture that cannot
    # be quantised cannot test a quantiser, and the failure appears at layer 0 of an
    # unrelated tool.
    kda_head_dim = max(GROUP, (hidden // n_kda_heads // 2 + GROUP - 1) // GROUP * GROUP)
    # MLA layers: every 4th, one-based in the config as the reader expects.
    full_attn = [i for i in range(4, layers + 1, 4)]
    if not full_attn:
        full_attn = [layers]
    return {
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "vocab_size": vocab,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "kda_num_heads": n_kda_heads,
        "kda_head_dim": kda_head_dim,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "num_attention_heads": 4,
        "q_lora_rank": 64,
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 24,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "mla_use_output_gate": True,
        "num_experts": 8,
        "num_experts_per_token": 2,
        "num_shared_experts": 2,
        "routed_expert_hidden_size": 64,
        "moe_intermediate_size": 64,
        "routed_scaling_factor": 1.0,
        "moe_renormalize": True,
        "latent_moe_use_norm": True,
        "first_k_dense_replace": 1,
        "intermediate_size": 96,
        "attn_res_block_size": 3,
        "situ_beta": 4.0,
        "situ_linear_beta": 25.0,
        "full_attn_layers": full_attn,
        "_random_weights": True,
        "_seed": seed,
    }


def build(cfg, rng):
    H = cfg["hidden_size"]
    NL = cfg["num_hidden_layers"]
    V = cfg["vocab_size"]
    P = cfg["kda_num_heads"] * cfg["kda_head_dim"]
    qh = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
    NH = cfg["num_attention_heads"]
    LAT = cfg["routed_expert_hidden_size"]
    MI = cfg["moe_intermediate_size"]
    SI = MI * cfg["num_shared_experts"]
    NE = cfg["num_experts"]
    mla = set(i - 1 for i in cfg["full_attn_layers"])     # config is ONE-based

    T = {}

    def n(name, *shape):
        """A narrow (matmul) weight: bf16, the format the binder tags a layer with.

        std 0.05 rather than 0.02: at 0.02 the signal through 13 layers decays into the
        rounding floor and downstream tensors stop mattering, which makes the fixture
        useless for exactly the checks it exists to support."""
        T[name] = bf16(rng.normal(0, 0.05, size=shape))

    def w(name, *shape):
        """A wide weight the engine points at IN PLACE: F32 on disk, no widening.

        The conv kernels, A_log, dt_bias, o_norm and the router bias are here. See wb()
        for why the split matters and is not a free choice."""
        T[name] = rng.normal(0, 0.02, size=shape).astype(np.float32)

    def wb(name, *shape):
        """A wide weight the engine WIDENS from BF16 into the slot's widen area.

        THE SPLIT BETWEEN w AND wb IS NOT COSMETIC. k3_bind_widen_bytes budgets the widen
        area for exactly one set of tensors -- the six per-layer norms, the two MLA
        layernorms, the routed-expert norm and the router gate -- and a trunk that stores
        anything ELSE as bf16 overflows that area and is refused at bind time. The
        released checkpoint stores precisely this set as bf16 and the rest as fp32, which
        is why it fits.

        Both fixture generators wrote every wide tensor as F32, so no fixture ever
        exercised the widen path at all. tools/awq_trunk.py assumed fp32 norms on that
        evidence and crashed on the first real trunk; making everything bf16 instead
        overflowed the widen area. The real checkpoint is neither, and this is it."""
        T[name] = bf16(rng.normal(0, 0.02, size=shape))

    n(PRE + "embed_tokens.weight", V, H)
    n("language_model.lm_head.weight", V, H)
    w(PRE + "norm.weight", H)
    w(PRE + "output_attn_res_norm.weight", H)
    w(PRE + "output_attn_res_proj.weight", H)

    for L in range(NL):
        p = PRE + f"layers.{L}."
        for leaf in ("input_layernorm", "post_attention_layernorm",
                     "self_attention_res_norm", "self_attention_res_proj",
                     "mlp_res_norm", "mlp_res_proj"):
            wb(p + leaf + ".weight", H)

        if L in mla:
            n(p + "self_attn.q_a_proj.weight", cfg["q_lora_rank"], H)
            wb(p + "self_attn.q_a_layernorm.weight", cfg["q_lora_rank"])
            n(p + "self_attn.q_b_proj.weight", NH * qh, cfg["q_lora_rank"])
            n(p + "self_attn.kv_a_proj_with_mqa.weight",
              cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"], H)
            wb(p + "self_attn.kv_a_layernorm.weight", cfg["kv_lora_rank"])
            n(p + "self_attn.kv_b_proj.weight",
              NH * (cfg["qk_nope_head_dim"] + cfg["v_head_dim"]), cfg["kv_lora_rank"])
            n(p + "self_attn.o_proj.weight", H, NH * cfg["v_head_dim"])
            if cfg["mla_use_output_gate"]:
                n(p + "self_attn.g_proj.weight", NH * cfg["v_head_dim"], H)
        else:
            for leaf in ("q_proj", "k_proj", "v_proj", "g_proj"):
                n(p + f"self_attn.{leaf}.weight", P, H)
            n(p + "self_attn.o_proj.weight", H, P)
            n(p + "self_attn.f_a_proj.weight", cfg["kda_head_dim"], H)
            n(p + "self_attn.f_b_proj.weight", P, cfg["kda_head_dim"])
            n(p + "self_attn.b_proj.weight", cfg["kda_num_heads"], H)
            # THE KDA RECURRENCE HAS TO ACTUALLY ENGAGE, and at std 0.02 it does not.
            # The decay, the delta-rule gate and the ShortConv compound, and the layer
            # output underflows to ~1e-6 -- scale_test reports exactly that on a
            # full-width random layer. A fixture in that state cannot test anything
            # upstream of the KDA output: zeroing q_proj or g_proj outright changed
            # nothing measurable, because they were being multiplied into zero.
            #
            # These three are the values tools/make_k3_oracle.py uses to make the caps
            # and the routing bias engage, and they exist for the same reason.
            T[p + "self_attn.A_log"] = np.linspace(-0.4, 1.2, cfg["kda_head_dim"]).astype(np.float32)
            T[p + "self_attn.dt_bias"] = rng.normal(0, 0.5, size=P).astype(np.float32)
            for leaf in ("q_conv1d", "k_conv1d", "v_conv1d"):
                T[p + f"self_attn.{leaf}.weight"] = rng.normal(
                    0, 0.4, size=(P, 1, cfg["short_conv_kernel_size"])).astype(np.float32)
            w(p + "self_attn.o_norm.weight", cfg["kda_head_dim"])

        if L < cfg["first_k_dense_replace"]:
            n(p + "mlp.gate_proj.weight", cfg["intermediate_size"], H)
            n(p + "mlp.up_proj.weight", cfg["intermediate_size"], H)
            n(p + "mlp.down_proj.weight", H, cfg["intermediate_size"])
        else:
            q = p + "block_sparse_moe."
            wb(q + "gate.weight", NE, H)
            w(q + "gate.e_score_correction_bias", NE)
            n(q + "routed_expert_down_proj.weight", LAT, H)
            n(q + "routed_expert_up_proj.weight", H, LAT)
            wb(q + "routed_expert_norm.weight", LAT)
            n(q + "shared_experts.gate_proj.weight", SI, H)
            n(q + "shared_experts.up_proj.weight", SI, H)
            n(q + "shared_experts.down_proj.weight", H, SI)
            for e in range(NE):
                for leaf, shape in (("w1", (MI, LAT)), ("w3", (MI, LAT)), ("w2", (LAT, MI))):
                    packed, scales = mxfp4_quant(rng.normal(0, 0.5, size=shape).astype(np.float32))
                    T[q + f"experts.{e}.{leaf}.weight_packed"] = packed
                    T[q + f"experts.{e}.{leaf}.weight_scale"] = scales
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--layers", type=int, default=13)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    if a.hidden % 64:
        print("--hidden must be a multiple of 64 so every narrow width is a "
              "multiple of the MXFP4 group size", file=sys.stderr)
        return 2

    os.makedirs(a.out_dir, exist_ok=True)
    cfg = make_config(a.hidden, a.layers, a.vocab, a.seed)
    rng = np.random.default_rng(a.seed)
    tensors = build(cfg, rng)

    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    nbytes = write_safetensors(os.path.join(a.out_dir, "model.safetensors"), tensors)

    print(f"wrote {a.out_dir}: {len(tensors)} tensors, {nbytes / 1e6:.1f} MB, "
          f"{a.layers} layers, hidden {a.hidden}, vocab {a.vocab}")
    print("THESE WEIGHTS ARE NOISE. The engine will run to completion against them and")
    print("every number it reports about QUALITY is meaningless. Use this to test that")
    print("machinery works, never to measure whether a model is good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
