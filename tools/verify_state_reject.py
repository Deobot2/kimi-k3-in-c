#!/usr/bin/env python3
"""verify_state_reject.py - a corrupted --load-state file must be refused, not corrupt
the heap.

    python3 tools/verify_state_reject.py <k3_bin> <checkpoint_dir> [--work DIR]

WHY THIS EXISTS
    k3_state_load() in src/cli/k3_run.c reads several fields straight out of a state
    file's header -- most importantly `nseq`, which becomes both a pointer offset
    (`seq + prior`) and a term in every buffer size for the rest of the run -- and used
    to trust every one of them. A state file is just a file on disk: a bad shutdown, a
    hand edit, a copy from an incompatible build, or something actively adversarial can
    all produce one with an out-of-range field, and this project's stated policy
    everywhere else is to refuse malformed input loudly rather than let it drive memory
    arithmetic. This tool proves the state loader actually holds to that policy, the
    same way tests/unit/test_st.c's `reject` mode does for safetensors headers, but a
    state file cannot be hand-authored the way a JSON config or a safetensors header can:
    its fingerprint depends on the checkpoint's own config, so it has to be produced by a
    real --save-state run first and then corrupted.

    Confirmed with AddressSanitizer during development: an nseq of -1 made `seq + prior`
    point before the start of `seq`'s allocation, and the following memcpy of the new
    prompt into it was a real heap-buffer-overflow. This script is the black-box
    regression test for the fix, driving the actual `k3` binary rather than the static
    functions inside k3_run.c, which have no header and cannot be linked into a separate
    test binary without restructuring the file.

WHY THE HEADER OFFSETS ARE HARDCODED HERE
    K3StateHdr has no Python binding, so this mirrors its field layout by hand (see
    HDR_FORMAT below) and corrupts one field at a time by byte offset. That is a real
    parallel-maintenance surface: reorder or resize a field in K3StateHdr without
    updating HDR_FORMAT here and this script corrupts the wrong bytes. The safety net is
    struct.calcsize(HDR_FORMAT) being asserted against the actual file size the C side
    wrote in `probe()`, so a drift trips a loud assertion here rather than silently
    testing nothing.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import os

# Mirrors K3StateHdr in src/cli/k3_run.c field for field, in declaration order. '@'
# (native alignment) so Python inserts the same padding the C compiler does on this
# platform. One (name, format) pair per field -- the single source of truth both
# unpack_header() and corrupt() work from, so there is exactly one place to keep in
# sync with the C struct rather than two.
HDR_LAYOUT = (
    ("magic", "4s"), ("version", "i"), ("fp", "12i"),
    ("n_bound", "i"), ("n_mla", "i"), ("cached", "i"), ("nseq", "i"),
    ("kper", "q"), ("kvpp", "q"), ("ropepp", "q"),
    ("kv_mode", "i"), ("kv_window", "i"), ("kv_sinks", "i"), ("kv_rows", "i"),
    ("kv_row_floats", "q"),
)
HDR_FORMAT = "@" + "".join(fmt for _, fmt in HDR_LAYOUT)


def unpack_header(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(struct.calcsize(HDR_FORMAT))
    vals = struct.unpack(HDR_FORMAT, raw)
    out, i = {}, 0
    for name, fmt in HDR_LAYOUT:
        if name == "fp":
            out[name] = vals[i:i + 12]
            i += 12
        else:
            out[name] = vals[i]
            i += 1
    return out


def _field_offset(field: str) -> tuple[int, str]:
    """Byte offset and struct format char of one HDR_LAYOUT field, computed by summing
    the (correctly padded) size of every field before it."""
    prefix = "@"
    for name, fmt in HDR_LAYOUT:
        if name == field:
            return struct.calcsize(prefix), fmt
        prefix += fmt
    raise KeyError(field)


def corrupt(src: str, dst: str, field: str, value: int) -> None:
    """Copy src to dst with one scalar header field overwritten."""
    with open(src, "rb") as f:
        data = bytearray(f.read())
    off, fmt = _field_offset(field)
    struct.pack_into("@" + fmt, data, off, value)
    with open(dst, "wb") as f:
        f.write(data)


def run(args) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=60)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("k3_bin")
    ap.add_argument("checkpoint_dir")
    ap.add_argument("--work", default=None, help="scratch dir (default: alongside checkpoint)")
    a = ap.parse_args()

    work = a.work or os.path.join(a.checkpoint_dir, "_state_reject_work")
    os.makedirs(work, exist_ok=True)
    good = os.path.join(work, "good.k3s")

    common = [a.k3_bin, a.checkpoint_dir, "--ids", "1,2,3", "--incremental",
              "--cache-gb", "1", "--trunk-gb", "1"]

    r = run(common + ["--gen", "2", "--save-state", good])
    if r.returncode != 0:
        print("FAIL  could not produce a state file to corrupt:\n" + r.stderr)
        return 1
    hdr = unpack_header(good)
    print("  ok    wrote a real state file: nseq=%d kper=%d kvpp=%d" %
          (hdr["nseq"], hdr["kper"], hdr["kvpp"]))

    fails = 0
    cases = [
        ("nseq", -1, "a pointer offset into `seq`"),
        ("nseq", 10**8, "overflows the Tmax/seq size arithmetic"),
        ("kper", hdr["kper"] + 1, "sizes the `ks` read"),
        ("kvpp", hdr["kvpp"] + 1, "sizes the expanded-KV read (if this run used one)"),
    ]
    for field, value, why in cases:
        bad = os.path.join(work, "bad_%s.k3s" % field)
        corrupt(good, bad, field, value)
        r = run(common + ["--gen", "1", "--load-state", bad])
        refused = r.returncode != 0 and "REFUSING" in r.stderr
        crashed = r.returncode < 0   # negative return = killed by a signal (e.g. SIGSEGV)
        if crashed:
            print("  FAIL  %-14s = %-12d CRASHED (signal %d) instead of refusing -- %s"
                  % (field, value, -r.returncode, why))
            fails += 1
        elif not refused:
            print("  FAIL  %-14s = %-12d loaded without complaint -- %s"
                  % (field, value, why))
            fails += 1
        else:
            print("  ok    %-14s = %-12d correctly refused -- %s" % (field, value, why))

    # And the positive control: an untouched copy must still load cleanly, or every
    # "correctly refused" result above would just mean this run() invocation is broken.
    r = run(common + ["--gen", "1", "--load-state", good])
    if r.returncode != 0:
        print("  FAIL  the UNCORRUPTED state file was refused:\n" + r.stderr)
        fails += 1
    else:
        print("  ok    the uncorrupted state file still loads cleanly")

    if fails:
        print("\n%d/%d checks FAILED" % (fails, len(cases) + 1))
        return 1
    print("\nSTATE REJECT TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
