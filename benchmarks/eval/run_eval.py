#!/usr/bin/env python3
"""run_eval.py - measure what a change to the weights costs, against a baseline.

    # 1. record what the released weights do (do this ONCE, keep the output)
    python3 benchmarks/eval/run_eval.py --model /path/k3model --tok /path/k3model \\
        --trunk /path/trunk-bf16 --label bf16 --out benchmarks/eval/results/bf16

    # 2. record what a candidate does, and score it against that baseline
    python3 benchmarks/eval/run_eval.py --model /path/k3model --tok /path/k3model \\
        --trunk /path/trunk-mxfp4 --label mxfp4 --out benchmarks/eval/results/mxfp4 \\
        --baseline benchmarks/eval/results/bf16

    # 3. compare any number of finished runs
    python3 benchmarks/eval/run_eval.py --report benchmarks/eval/results/*

WHY THIS EXISTS

    A quantised trunk cannot be byte-compared against anything, so every bit-identity
    gate in this repository is void against it and `--ppl` was left carrying the whole
    argument on its own. It is not enough on its own, and the way it is not enough has
    already cost a wrong conclusion here: an MXFP4 trunk scored 110.05 against the bf16
    trunk's 97.40, a 13% rise that reads like a modest tax, and the same weights produced
    generations that collapsed into an eight-token loop.

    Both numbers were right. Perplexity is TEACHER-FORCED: at every position the model is
    conditioned on the REAL text, never on its own output, so it never gets the chance to
    wander and cannot report that it would have. Degeneration is a property of the
    self-conditioned trajectory, and it is invisible to any teacher-forced metric by
    construction.

    So this runs two legs and reports them side by side.

    LEG A, teacher forced, over a fixed corpus:
      perplexity            the conventional number, per document and overall
      loss ratio            ppl(candidate) / ppl(baseline)
      top-1 agreement       how often the candidate predicts what the BASELINE predicted.
                            This is the sharp instrument. It compares the candidate
                            against the model it must stand in for rather than against
                            the text, and unlike a ratio of two means it does not average
                            a catastrophic minority away.
      NLL delta spread      per-position, so "uniformly slightly worse" and "fine except
                            on 3% of tokens" stop looking identical.

    LEG B, free running, from short prompts:
      repeat rate           fraction of generated tokens that continue an already-seen
                            n-gram: the direct measure of the observed failure
      loop period           the shortest cycle the tail is stuck in, or none
      distinct-n            vocabulary diversity of the continuation

    THE CONTROL THAT MAKES LEG B MEAN ANYTHING. Greedy decoding of a base model with no
    chat template repeats on its own; that is ordinary behaviour, not damage. So the
    baseline is measured on the SAME prompts with the SAME settings, and what is reported
    is the candidate's degeneration MINUS the baseline's. Without that subtraction this
    leg would condemn the released weights too.

COST

    Leg A is one teacher-forced sweep per document, all in one process. Compute scales
    with total positions and attention is quadratic in document length, which is why the
    corpus is many short documents rather than a few long ones. Leg B costs one full
    decode step per generated token, so --gen dominates; keep it small at first.

    Start with --quick to see the shape of the thing before committing to a full run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# Must match K3PplRec in src/cli/k3_run.c.
REC = np.dtype([("target", "<i4"), ("top1", "<i4"),
                ("target_lp", "<f4"), ("top1_lp", "<f4"), ("entropy", "<f4")])

# Short, open-ended, and deliberately varied in how strongly they invite repetition.
# The last two are the traps: a list and a table are structures a healthy model continues
# by repeating a pattern, so they separate "repeats because the text repeats" from
# "repeats because the model has collapsed".
GEN_PROMPTS = [
    ("open-prose",   "The difference between the two approaches only became obvious"),
    ("explanatory",  "There are three reasons a cache with more memory can be slower:"),
    ("code",         "static int ring_write(Ring *r, const unsigned char *src, size_t n)\n{"),
    ("factual",      "The speed of light in vacuum is"),
    ("list-trap",    "Checklist before release:\n1. Run the full test suite\n2."),
    ("table-trap",   "| layer | bytes | kind |\n|---|---|---|\n| 0 | 2513924096 | dense |\n|"),
]


# ------------------------------------------------------------------ tokenizing
def tokenize(tok_dir: str, text: str) -> list[int]:
    """Tokenize through tools/tok.py, which is the only tokenizer in this repo that reads
    the released tiktoken.model. Shelling out keeps one implementation rather than two."""
    r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "tok.py"), "encode", text],
                       capture_output=True, text=True,
                       env={**os.environ, "K3_HF_DIR": tok_dir})
    if r.returncode != 0:
        raise SystemExit(f"tok.py failed: {r.stderr.strip()}")
    return [int(x) for x in r.stdout.strip().split(",") if x.strip()]


def synthetic_ids(name: str, vocab: int, n: int) -> list[int]:
    """Deterministic pseudo-text for wiring checks against a checkpoint with no
    tokenizer. Seeded from the document name so every run scores identical ids, which is
    the one property the comparison actually depends on."""
    # crc32, NOT hash(). Python salts the hash of a string per process unless
    # PYTHONHASHSEED is set, so hash() here produced a DIFFERENT suite on every
    # invocation and the baseline comparison rejected its own model as a different one.
    # Caught by the identical-model self-check, which is what that check is for.
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return [int(v) for v in rng.integers(0, vocab, size=n)]


def build_suite(tok_dir: str, out_path: str, max_ids: int, synth_vocab: int = 0) -> tuple[int, int]:
    """Tokenize the corpus once into the suite file the engine reads.

    Written next to the results so a run carries the exact ids it was scored on. Two runs
    compared against different tokenizations would produce a confident and meaningless
    ratio, and nothing in the numbers would show it."""
    files = sorted(glob.glob(os.path.join(CORPUS, "*.txt")))
    if not files:
        raise SystemExit(f"no corpus documents in {CORPUS}")
    lines, total = [], 0
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        if synth_vocab:
            ids = synthetic_ids(name, synth_vocab, max_ids or 128)
        else:
            with open(p, encoding="utf-8") as f:
                ids = tokenize(tok_dir, f.read())
            if max_ids and len(ids) > max_ids:
                ids = ids[:max_ids]
        if len(ids) < 2:
            continue
        lines.append(f"{name}\t{','.join(str(i) for i in ids)}")
        total += len(ids) - 1
    with open(out_path, "w") as f:
        f.write("# k3 evaluation suite. Ids are tokenizer-specific: a run scored on ids\n")
        f.write("# from a different vocabulary is meaningless and looks fine.\n")
        f.write(f"# tokenizer: {'SYNTHETIC (wiring check only)' if synth_vocab else tok_dir}\n")
        f.write("\n".join(lines) + "\n")
    return len(lines), total


# ------------------------------------------------------------------ degeneration
def repeat_rate(ids: list[int], n: int = 4) -> float:
    """Fraction of positions whose preceding n-gram has already occurred.

    This is the blunt, robust measure: it fires on any repetition, exact or drifting,
    which is what makes it hard to game and easy to interpret."""
    if len(ids) <= n:
        return 0.0
    seen, hits = set(), 0
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        if g in seen:
            hits += 1
        seen.add(g)
    return hits / (len(ids) - n + 1)


def loop_period(ids: list[int], min_reps: int = 3) -> int:
    """Shortest p such that the tail of the sequence is p-periodic for min_reps cycles.

    0 means no loop found. This is the metric that names the observed failure directly:
    an eight-token loop returns 8. Reported alongside repeat_rate because they disagree
    usefully -- a model quoting a phrase twice raises the rate and has no period, while a
    collapsed model has both.

    IT CAN ONLY SEE PERIODS UP TO len(ids) // min_reps. Three cycles is what separates a
    loop from a model that happened to repeat itself once, and the cost of that threshold
    is that a p-token loop needs 3p generated tokens to be visible at all. At the default
    --gen 64 that is a period of 21; an eight-token loop is comfortably inside it, and a
    forty-token one would report 0 and look healthy. run_eval warns when --gen is small
    enough for this to bite."""
    if not ids:
        return 0
    for p in range(1, len(ids) // min_reps + 1):
        tail = ids[-p * min_reps:]
        if all(tail[i] == tail[i % p] for i in range(len(tail))):
            return p
    return 0


def distinct_n(ids: list[int], n: int = 3) -> float:
    if len(ids) < n:
        return 1.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


def degeneration(ids: list[int]) -> dict:
    return {
        "tokens": len(ids),
        "repeat_rate_4": round(repeat_rate(ids, 4), 4),
        "distinct_3": round(distinct_n(ids, 3), 4),
        "loop_period": loop_period(ids),
    }


# ------------------------------------------------------------------ engine
def k3(args: list[str], log: str) -> None:
    cmd = [os.path.join(REPO, "bin", "k3")] + args
    with open(log, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.stderr.write(f"\nk3 failed (exit {r.returncode}); tail of {log}:\n")
        with open(log) as f:
            sys.stderr.write("".join(f.readlines()[-25:]))
        raise SystemExit(1)


def model_args(a) -> list[str]:
    out = [a.model]
    if a.trunk:
        out += ["--trunk", a.trunk]
    if a.trunk_gb:
        out += ["--trunk-gb", str(a.trunk_gb)]
    if a.cache_gb:
        out += ["--cache-gb", str(a.cache_gb)]
    if a.preset:
        out += ["--preset", a.preset]
    return out


def run_one(a, outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    suite = os.path.join(outdir, "suite.tsv")
    ndoc, npos = build_suite(a.tok, suite, a.max_ids, a.synthetic_vocab)
    print(f"  corpus: {ndoc} documents, {npos} scored positions")

    result = {"label": a.label, "trunk": a.trunk or "(resident)",
              "model": a.model, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---- leg A ----
    print("  leg A: teacher-forced perplexity ...", flush=True)
    t0 = time.time()
    k3(model_args(a) + ["--ppl-file", suite,
                        "--ppl-dump", os.path.join(outdir, "ppl.bin"),
                        "--out", os.path.join(outdir, "ppl.json")],
       os.path.join(outdir, "ppl.log"))
    with open(os.path.join(outdir, "ppl.json")) as f:
        result["ppl"] = json.load(f)
    result["ppl"]["seconds"] = round(time.time() - t0, 1)
    print(f"    perplexity {result['ppl']['perplexity']:.4f} "
          f"over {result['ppl']['ppl_positions']} positions "
          f"({result['ppl']['seconds']:.0f} s)")

    # ---- leg B ----
    if a.gen > 0 and not a.synthetic_vocab:
        print(f"  leg B: free generation, {a.gen} tokens x {len(GEN_PROMPTS)} prompts ...",
              flush=True)
        gens = []
        for name, text in GEN_PROMPTS:
            gp = os.path.join(outdir, f"gen-{name}.json")
            pf = os.path.join(outdir, f"prompt-{name}.txt")
            with open(pf, "w") as f:
                f.write(text)
            t1 = time.time()
            k3(model_args(a) + ["--prompt-file", pf, "--tok", a.tok,
                                "--gen", str(a.gen), "--incremental", "--out", gp],
               os.path.join(outdir, f"gen-{name}.log"))
            with open(gp) as f:
                g = json.load(f)
            d = degeneration(g["generated_ids"])
            d["prompt"] = name
            d["seconds"] = round(time.time() - t1, 1)
            d["ids"] = g["generated_ids"]
            gens.append(d)
            loop = f"LOOP p={d['loop_period']}" if d["loop_period"] else "no loop"
            print(f"    {name:<14} repeat {d['repeat_rate_4']:.2f}  "
                  f"distinct-3 {d['distinct_3']:.2f}  {loop}")
        result["gen"] = gens

    with open(os.path.join(outdir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


# ------------------------------------------------------------------ comparison
def compare(base: dict, cand: dict, base_dir: str, cand_dir: str) -> dict:
    """Score a candidate against a baseline. Everything here is a DIFFERENCE; nothing in
    this function is meaningful about a single run on its own."""
    out = {}
    bp, cp = base["ppl"]["perplexity"], cand["ppl"]["perplexity"]
    out["ppl_baseline"] = bp
    out["ppl_candidate"] = cp
    out["loss_ratio"] = cp / bp if bp else float("nan")

    bb = os.path.join(base_dir, "ppl.bin")
    cb = os.path.join(cand_dir, "ppl.bin")
    if os.path.exists(bb) and os.path.exists(cb):
        b = np.fromfile(bb, dtype=REC)
        c = np.fromfile(cb, dtype=REC)
        if len(b) != len(c):
            out["agreement_error"] = (f"{len(b)} baseline records vs {len(c)} candidate; "
                                      "the two runs scored different suites")
        elif not np.array_equal(b["target"], c["target"]):
            out["agreement_error"] = ("the two runs scored different ids at the same "
                                      "positions; the suites are not the same text")
        else:
            # The headline: how often the candidate would have emitted what the baseline
            # emitted, at identical context. Teacher forcing is what makes this a fair
            # comparison -- both models see the same history at every position, so a
            # single early divergence cannot cascade the way it does in free running.
            out["top1_agreement"] = float((b["top1"] == c["top1"]).mean())
            out["top1_agreement_n"] = len(b)
            # Where each model would have been RIGHT, which is the accuracy the
            # perplexity is a soft version of.
            out["baseline_top1_correct"] = float((b["top1"] == b["target"]).mean())
            out["candidate_top1_correct"] = float((c["top1"] == c["target"]).mean())

            d = (b["target_lp"] - c["target_lp"]).astype(np.float64)   # >0 = worse
            out["nll_delta_mean"] = float(d.mean())
            out["nll_delta_p50"] = float(np.percentile(d, 50))
            out["nll_delta_p95"] = float(np.percentile(d, 95))
            out["nll_delta_p99"] = float(np.percentile(d, 99))
            # The shape question: is the loss spread evenly, or carried by a few
            # positions? A ratio of means cannot tell these apart and they are different
            # failures. >1 nat is a position where the candidate went from confident to
            # lost.
            out["frac_positions_worse_1nat"] = float((d > 1.0).mean())
            out["frac_positions_better"] = float((d < 0).mean())
            out["entropy_baseline"] = float(b["entropy"].mean())
            out["entropy_candidate"] = float(c["entropy"].mean())

    if "gen" in base and "gen" in cand:
        bg = {g["prompt"]: g for g in base["gen"]}
        rows = []
        for g in cand["gen"]:
            b0 = bg.get(g["prompt"])
            if not b0:
                continue
            rows.append({
                "prompt": g["prompt"],
                "repeat_baseline": b0["repeat_rate_4"],
                "repeat_candidate": g["repeat_rate_4"],
                "repeat_excess": round(g["repeat_rate_4"] - b0["repeat_rate_4"], 4),
                "loop_baseline": b0["loop_period"],
                "loop_candidate": g["loop_period"],
                "distinct_baseline": b0["distinct_3"],
                "distinct_candidate": g["distinct_3"],
            })
        out["gen"] = rows
        if rows:
            out["repeat_excess_mean"] = round(
                sum(r["repeat_excess"] for r in rows) / len(rows), 4)
            # New loops are the unambiguous signal: the baseline did not cycle here and
            # the candidate does.
            out["new_loops"] = sum(1 for r in rows
                                   if r["loop_candidate"] and not r["loop_baseline"])
    return out


def render(cmp_: dict, base_label: str, cand_label: str) -> str:
    L = []
    L.append(f"\n{cand_label}  vs  {base_label}")
    L.append("=" * 62)
    L.append("\nLEG A - teacher forced")
    L.append(f"  perplexity            {cmp_['ppl_baseline']:.4f} -> {cmp_['ppl_candidate']:.4f}")
    L.append(f"  loss ratio            {cmp_['loss_ratio']:.4f}x")
    if "top1_agreement" in cmp_:
        L.append(f"  top-1 agreement       {100 * cmp_['top1_agreement']:.2f}%  "
                 f"over {cmp_['top1_agreement_n']} positions")
        L.append(f"  top-1 correct         {100 * cmp_['baseline_top1_correct']:.2f}% -> "
                 f"{100 * cmp_['candidate_top1_correct']:.2f}%")
        L.append(f"  NLL delta  mean/p95   {cmp_['nll_delta_mean']:+.4f} / "
                 f"{cmp_['nll_delta_p95']:+.4f} nats")
        L.append(f"  positions >1 nat worse {100 * cmp_['frac_positions_worse_1nat']:.2f}%")
        L.append(f"  mean entropy          {cmp_['entropy_baseline']:.3f} -> "
                 f"{cmp_['entropy_candidate']:.3f} nats")
    elif "agreement_error" in cmp_:
        L.append(f"  top-1 agreement       UNAVAILABLE: {cmp_['agreement_error']}")

    if cmp_.get("gen"):
        L.append("\nLEG B - free running (excess over baseline; the baseline repeats too)")
        L.append(f"  {'prompt':<14} {'repeat b->c':>16} {'excess':>8} {'loop b->c':>12}")
        for r in cmp_["gen"]:
            lb = r["loop_baseline"] or "-"
            lc = r["loop_candidate"] or "-"
            L.append(f"  {r['prompt']:<14} "
                     f"{r['repeat_baseline']:>7.2f}->{r['repeat_candidate']:<7.2f} "
                     f"{r['repeat_excess']:>+8.2f} {lb!s:>5}->{lc!s:<6}")
        L.append(f"  mean repeat excess    {cmp_['repeat_excess_mean']:+.4f}")
        L.append(f"  NEW loops             {cmp_['new_loops']} of {len(cmp_['gen'])} prompts")

    L.append("\nREADING THIS")
    L.append("  Loss ratio is the headline and the least informative line here. Top-1")
    L.append("  agreement says how often the candidate would have made the baseline's")
    L.append("  own choice; a ratio near 1.0 with agreement below ~90% means the two")
    L.append("  models are equally surprised by the text and no longer the same model.")
    L.append("  NEW loops is the one that maps onto output being visibly broken, and no")
    L.append("  leg-A number predicts it.")
    return "\n".join(L)


# ------------------------------------------------------------------ self-test
def self_test() -> int:
    """Check the degeneration metrics against sequences whose answers are known.

    These run without a model, which matters: they are the only part of this harness that
    can be verified at all on a machine without the checkpoint, and loop_period is the
    metric that has to catch the failure this whole file exists for. A metric that is
    wrong in the direction of "no loop found" would clear a broken model silently."""
    fails = []

    def eq(what, got, want):
        if got != want:
            fails.append(f"{what}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {what:<44} {got!r}")

    # The observed failure: eight tokens, cycling.
    eq("loop_period, exact 8-cycle", loop_period([1, 2, 3, 4, 5, 6, 7, 8] * 5), 8)
    eq("loop_period, single token repeated", loop_period([42] * 30), 1)
    eq("loop_period, clean text", loop_period([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]), 0)
    # A loop that STARTS partway through is the realistic case: the model is fine for a
    # while and then collapses. Only the tail is periodic.
    eq("loop_period, collapses at the tail",
       loop_period([11, 22, 33, 44, 55, 66] + [7, 8, 9] * 6), 3)
    # Two repeats is a model quoting itself; three is a model stuck. The threshold is
    # what keeps ordinary repetition out of the loop count.
    eq("loop_period, only two cycles is not a loop", loop_period([1, 2, 3, 1, 2, 3]), 0)

    eq("repeat_rate, all distinct", round(repeat_rate(list(range(40)), 4), 4), 0.0)
    # 29 four-grams, of which exactly 4 are distinct in a period-4 cycle.
    eq("repeat_rate, fully periodic", round(repeat_rate([1, 2, 3, 4] * 8, 4), 4), 0.8621)
    eq("distinct_3, all distinct", round(distinct_n(list(range(40)), 3), 4), 1.0)
    eq("distinct_3, single token", round(distinct_n([5] * 40, 3), 4), round(1 / 38, 4))

    # The record layout must match the C struct, or every comparison silently misaligns.
    eq("K3PplRec is 20 bytes", REC.itemsize, 20)

    print("\n" + ("SELF TEST FAILED" if fails else "SELF TEST PASSED"))
    for f in fails:
        print("  " + f)
    return 1 if fails else 0


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="checkpoint directory")
    ap.add_argument("--tok", help="directory with tiktoken.model (defaults to --model)")
    ap.add_argument("--trunk", help="packed trunk directory; omit for a resident trunk")
    ap.add_argument("--label", default="run", help="name for this run in reports")
    ap.add_argument("--out", help="directory to write this run's results into")
    ap.add_argument("--baseline", help="a previous --out directory to score against")
    ap.add_argument("--report", nargs="*", help="compare finished run directories")
    ap.add_argument("--self-test", action="store_true",
                    help="check the degeneration metrics against known sequences; needs no model")
    ap.add_argument("--gen", type=int, default=64, help="tokens per generation prompt (0 disables leg B)")
    ap.add_argument("--max-ids", type=int, default=1024, help="truncate each document to N ids")
    ap.add_argument("--quick", action="store_true",
                    help="128 ids per document and 16 generated tokens, to check the wiring")
    ap.add_argument("--synthetic-vocab", type=int, default=0,
                    help="WIRING CHECK ONLY: build the suite from pseudo-random ids in "
                         "[0,N) instead of tokenizing the corpus, so the harness can be "
                         "exercised against a checkpoint that has no tokenizer. Every "
                         "number it produces is meaningless as a measure of quality")
    ap.add_argument("--trunk-gb", type=float)
    ap.add_argument("--cache-gb", type=float)
    ap.add_argument("--preset")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    if a.report:
        dirs = [d for pat in a.report for d in sorted(glob.glob(pat))]
        runs = []
        for d in dirs:
            p = os.path.join(d, "result.json")
            if os.path.exists(p):
                with open(p) as f:
                    runs.append((d, json.load(f)))
        if len(runs) < 2:
            raise SystemExit("--report needs at least two finished run directories")
        base_dir, base = runs[0]
        print(f"baseline: {base['label']}  ({base_dir})")
        for d, r in runs[1:]:
            print(render(compare(base, r, base_dir, d), base["label"], r["label"]))
        return 0

    if not a.model or not a.out:
        raise SystemExit("--model and --out are required (or use --report)")
    if a.quick:
        a.max_ids, a.gen = 128, 16
    if 0 < a.gen < 24:
        print(f"NOTE: --gen {a.gen} can only expose loops of period {a.gen // 3} or less,\n"
              "      because three cycles are what distinguish a loop from a repeat. The\n"
              "      repeat rate still works at any length; loop_period does not.\n")
    a.tok = a.tok or a.model
    if a.synthetic_vocab:
        print("SYNTHETIC IDS: this is a wiring check. The corpus is not being read, the\n"
              "generation leg is skipped, and no number below says anything about model\n"
              "quality.\n")

    print(f"run: {a.label}")
    res = run_one(a, a.out)

    if a.baseline:
        with open(os.path.join(a.baseline, "result.json")) as f:
            base = json.load(f)
        cmp_ = compare(base, res, a.baseline, a.out)
        with open(os.path.join(a.out, "vs-baseline.json"), "w") as f:
            json.dump(cmp_, f, indent=2)
        print(render(cmp_, base["label"], res["label"]))
    else:
        print("\nNo --baseline given, so this run is a BASELINE and nothing here is a")
        print("comparison. Perplexity alone does not say whether a model is good; it says")
        print("what this model scores on this corpus. Re-run with --baseline pointing at")
        print(f"{a.out} to score a candidate against it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
