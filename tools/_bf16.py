"""_bf16.py - bf16 <-> f32 conversion, shared by every trunk-quantising tool.

WHY THIS EXISTS
    awq_trunk.py, int8_trunk.py, mxfp4_trunk.py and qdq_trunk.py each carried their own
    copy of bf16_to_f32/f32_to_bf16. All four happened to agree, but nothing enforced
    that: a rounding fix applied to one copy and not the others would make the trunks
    those tools produce silently disagree on the least significant bit of every rounded
    weight, and the first symptom would be a --ppl regression on the tool nobody touched.
    One definition, imported everywhere, is the only way that class of drift is
    structurally impossible rather than merely unlikely.
"""
from __future__ import annotations

import numpy as np


def bf16_to_f32(u16):
    """bf16 bit pattern -> float32. Widening, exact: bf16 is float32's top 16 bits."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32):
    """float32 -> bf16 bit pattern, round-to-nearest-even on the dropped mantissa bits,
    matching torch's cast (the reference every quantised trunk is checked against).

    ascontiguousarray first: view() reinterprets bytes in place and requires both a
    contiguous buffer and the exact dtype it expects, which a caller's array is not
    guaranteed to already be (a strided slice, a ravel() the caller forgot, a stray
    float64). Silently viewing the wrong bytes is a worse failure than the extra copy
    ascontiguousarray costs when it is actually needed and skips when it is not.
    """
    u = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16)).astype(np.uint16)
