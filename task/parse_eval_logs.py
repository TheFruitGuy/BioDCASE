#!/usr/bin/env python3
"""
parse_eval_logs.py — pull per-class P/R out of eval logs and compute avg-PR F1.

Reads each runs/logs/eval_*.log file produced by run_all_evals.sh,
finds the last ENSEMBLE RESULT block, parses the three per-class lines,
then computes:

    P̄ = (P_bmabz + P_d + P_bp) / 3
    R̄ = (R_bmabz + R_d + R_bp) / 3
    F1 = 2·P̄·R̄ / (P̄ + R̄)

Prints two tables: full per-class breakdown, then a slide-ready summary.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

# (log filename, display label, slide reference)
LOGS = [
    ("eval_baseline_5seed.log",         "5-seed baseline",       "slide 4"),
    ("eval_pgi_5seed.log",              "PGI 5-seed",            "slide 5"),
    ("eval_mt_std_5seed_seg30.log",     "Std MSE MT @ 30s",      "slide 7"),
    ("eval_mt_amse_5seed_seg30.log",    "AMSE MT @ 30s",         "slide 7"),
    ("eval_mt_amse_5seed_seg60.log",    "AMSE MT @ 60s ★",       "slide 9"),
    ("eval_mt_confmask_seed1337.log",   "ConfMask MT (1 seed)",  "slide 7"),
    ("eval_hybrid_per_class_seg60.log", "Hybrid per-class ★",    "slide 10"),
]
LOGDIR = Path("runs/logs")

# A per-class line inside the ENSEMBLE block looks like:
#     BMABZ  t=0.25  TP=    5 FP=    3 FN=    7  P=0.625 R=0.417 F1=0.500
CLASS_RE = re.compile(
    r'^\s*(BMABZ|D|BP)\s+t=([\d.]+)\s+'
    r'TP=\s*(\d+)\s+FP=\s*(\d+)\s+FN=\s*(\d+)\s+'
    r'P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)',
    re.IGNORECASE,
)
HEADER_RE = re.compile(r'ENSEMBLE RESULT', re.IGNORECASE)


def parse_log(path: Path) -> dict | None:
    """Find last ENSEMBLE RESULT block and return {'BMABZ':{P,R,F1}, ...}."""
    if not path.exists():
        return None
    text = path.read_text(errors="replace")

    # Use the last ENSEMBLE RESULT block in case multiple runs are appended.
    last_header_end = -1
    for m in HEADER_RE.finditer(text):
        last_header_end = m.end()
    if last_header_end < 0:
        # Single-seed eval (cmd 6) has no ENSEMBLE block — fall back to last
        # per-class block in the file.
        last_header_end = 0

    tail = text[last_header_end:]
    classes: dict[str, dict[str, float]] = {}
    for line in tail.splitlines():
        m = CLASS_RE.search(line)
        if not m:
            continue
        cls = m.group(1).upper()
        if cls in classes:           # take the first occurrence after header
            continue
        classes[cls] = {
            "t":  float(m.group(2)),
            "tp": int(m.group(3)),
            "fp": int(m.group(4)),
            "fn": int(m.group(5)),
            "P":  float(m.group(6)),
            "R":  float(m.group(7)),
            "F1": float(m.group(8)),
        }
        if len(classes) == 3:
            break
    return classes if len(classes) == 3 else None


def avg_pr_f1(classes: dict) -> dict:
    ps = [classes[c]["P"] for c in ("BMABZ", "D", "BP")]
    rs = [classes[c]["R"] for c in ("BMABZ", "D", "BP")]
    Pbar = sum(ps) / 3
    Rbar = sum(rs) / 3
    f1 = 2 * Pbar * Rbar / (Pbar + Rbar) if (Pbar + Rbar) > 0 else 0.0
    # Also report mean-of-per-class-F1 for reference / cross-check
    macro_f1 = sum(classes[c]["F1"] for c in ("BMABZ", "D", "BP")) / 3
    return {"Pbar": Pbar, "Rbar": Rbar, "F1_avgPR": f1, "F1_meanOfF1": macro_f1}


def main():
    rows = []
    print()
    print("=" * 96)
    print(f"  {'Experiment':<24}  {'BMABZ P / R / F1':<22}  {'D P / R / F1':<22}  {'BP P / R / F1':<22}")
    print("=" * 96)
    for log_name, display, _ in LOGS:
        path = LOGDIR / log_name
        classes = parse_log(path)
        if classes is None:
            status = "MISSING FILE" if not path.exists() else "NO ENSEMBLE BLOCK PARSED"
            print(f"  {display:<24}  [{status}]   ({log_name})")
            rows.append((display, None))
            continue
        b, d, p = classes["BMABZ"], classes["D"], classes["BP"]
        print(f"  {display:<24}  "
              f"{b['P']:.3f} / {b['R']:.3f} / {b['F1']:.3f}     "
              f"{d['P']:.3f} / {d['R']:.3f} / {d['F1']:.3f}     "
              f"{p['P']:.3f} / {p['R']:.3f} / {p['F1']:.3f}")
        rows.append((display, classes))

    print()
    print("=" * 96)
    print("  SLIDE-READY SUMMARY (avg-PR F1 = 2·P̄·R̄ / (P̄+R̄), the BioDCASE official metric)")
    print("=" * 96)
    print(f"  {'Experiment':<24}  {'P̄':>8}  {'R̄':>8}  {'avg-PR F1':>12}  {'(mean-of-F1)':>14}  slide")
    print("  " + "-" * 92)
    for (display, classes), (_, _, slide) in zip(rows, LOGS):
        if classes is None:
            print(f"  {display:<24}  {'—':>8}  {'—':>8}  {'—':>12}  {'—':>14}  {slide}")
            continue
        m = avg_pr_f1(classes)
        print(f"  {display:<24}  {m['Pbar']:.4f}  {m['Rbar']:.4f}  "
              f"{m['F1_avgPR']:.4f}        {m['F1_meanOfF1']:.4f}        {slide}")

    print()
    print("  References (paper-reported, same metric):")
    print(f"  {'WhaleVAD paper baseline':<24}  {'—':>8}  {'—':>8}  {'0.4400':>12}")
    print(f"  {'WhaleVAD-BPN paper':<24}  {'—':>8}  {'—':>8}  {'0.4750':>12}")
    print()

    # Emit a markdown block the user can paste back into chat
    print("=" * 96)
    print("  PASTE-BACK BLOCK (copy this into chat to update the slides)")
    print("=" * 96)
    print()
    print("```")
    print(f"{'Experiment':<24}  {'P̄':>7}  {'R̄':>7}  {'avg-PR F1':>10}")
    for (display, classes), _ in zip(rows, LOGS):
        if classes is None:
            print(f"{display:<24}  {'—':>7}  {'—':>7}  {'—':>10}")
            continue
        m = avg_pr_f1(classes)
        print(f"{display:<24}  {m['Pbar']:.4f}   {m['Rbar']:.4f}   {m['F1_avgPR']:.4f}")
    print("```")
    print()


if __name__ == "__main__":
    main()
