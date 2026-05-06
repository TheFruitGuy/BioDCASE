"""
diagnose_d_subtypes.py  (v2 — fixed for actual schema)
=======================================================

Decomposes the 3-class "d" label into its underlying fine-grained labels
(found in the `annotation` column) and summarises distribution and
frequency-band statistics across train and validation sites.

Crucially, this script also looks at `low_frequency` / `high_frequency`
columns. Even if all D-class rows happen to share the same `annotation`
string, distinct call types often live in distinct frequency bands —
e.g. fin-whale 20-Hz pulses (~15-30 Hz, ~1s) vs blue-whale D-calls
(~30-100 Hz, ~1-4s). If the frequency distribution is bimodal, that's
direct evidence of acoustic subtypes hidden inside a single label.

Run from BioDCASE/task/:

    python diagnose_d_subtypes.py 2>&1 | tee d_diagnostics_stats.txt

Output: prints to stdout. No PNGs.
"""

from __future__ import annotations

import sys
import textwrap
import pandas as pd

import config as cfg
from dataset import load_annotations


# Candidate column names for the "fine-grained label", in priority order.
# 'annotation' is what your schema uses; the others are kept for
# compatibility with possible future relabelings.
FINE_LABEL_CANDIDATES = ["annotation", "label", "label_7class",
                         "label_orig", "label_full"]


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def find_fine_col(df: pd.DataFrame) -> str | None:
    for c in FINE_LABEL_CANDIDATES:
        if c in df.columns and c != "label_3class":
            return c
    return None


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `duration_s` column. Schema has `start_datetime` / `end_datetime`."""
    if "duration" in df.columns:
        # Val split has it precomputed
        df = df.copy()
        df["duration_s"] = pd.to_numeric(df["duration"], errors="coerce")
        return df
    if "start_datetime" in df.columns and "end_datetime" in df.columns:
        df = df.copy()
        s = pd.to_datetime(df["start_datetime"], utc=True, errors="coerce")
        e = pd.to_datetime(df["end_datetime"], utc=True, errors="coerce")
        df["duration_s"] = (e - s).dt.total_seconds()
        return df
    df = df.copy()
    df["duration_s"] = float("nan")
    return df


def describe_split(name: str, datasets: list[str]) -> tuple[pd.DataFrame, str | None]:
    """Print breakdown for a split, return (d-only frame, fine_col_name)."""
    section(f"{name.upper()}: subtype decomposition of 3-class 'd'")

    ann = load_annotations(datasets)
    print(f"\nAnnotation columns available: {list(ann.columns)}")
    print(f"Total annotations: {len(ann)}")

    if "label_3class" not in ann.columns:
        print("\nERROR: no 'label_3class' column. Aborting.")
        sys.exit(1)

    fine_col = find_fine_col(ann)
    if fine_col is None:
        print(f"\nNo fine-label column found. Tried: {FINE_LABEL_CANDIDATES}")
        return pd.DataFrame(), None
    print(f"Using '{fine_col}' as the fine-grained label column.\n")

    d_only = ann[ann["label_3class"] == "d"].copy()
    d_only = add_duration(d_only)
    print(f"Rows with label_3class == 'd': {len(d_only)}")
    if d_only.empty:
        return d_only, fine_col

    # Q1: which fine labels collapse into 'd'?
    print(f"\nFine-grained labels mapping to 'd':")
    counts = d_only[fine_col].value_counts(dropna=False)
    for label, n in counts.items():
        pct = 100.0 * n / len(d_only)
        print(f"  {str(label):24s}  n={n:6d}  ({pct:5.1f}%)")

    if len(counts) == 1:
        only_label = str(counts.index[0])
        print(f"\nOnly ONE fine label ('{only_label}') maps to 'd'. The label "
              f"column doesn't distinguish D subtypes, but the FREQUENCY band "
              f"analysis below may still reveal acoustic subtypes.")

    # Q2: per-site × per-subtype counts
    print(f"\nPer-site × per-subtype counts:")
    pivot = d_only.groupby(["dataset", fine_col]).size().unstack(fill_value=0)
    print(pivot.to_string())

    # Q3: duration per subtype
    print(f"\nDuration (s) per subtype:")
    for label in counts.index:
        sub = d_only[d_only[fine_col] == label]["duration_s"].dropna()
        if sub.empty:
            print(f"  {str(label):24s}  (no duration data)")
            continue
        print(f"  {str(label):24s}  "
              f"min={sub.min():.2f}  "
              f"p25={sub.quantile(0.25):.2f}  "
              f"median={sub.median():.2f}  "
              f"p75={sub.quantile(0.75):.2f}  "
              f"max={sub.max():.2f}  "
              f"mean={sub.mean():.2f}  "
              f"n={len(sub)}")

    # Q4: frequency band per subtype — the killer feature
    print(f"\nFrequency band (Hz) per subtype:")
    if "low_frequency" in d_only.columns and "high_frequency" in d_only.columns:
        lo = pd.to_numeric(d_only["low_frequency"], errors="coerce")
        hi = pd.to_numeric(d_only["high_frequency"], errors="coerce")
        d_only["lo_Hz"] = lo
        d_only["hi_Hz"] = hi
        d_only["bw_Hz"] = hi - lo

        for label in counts.index:
            sub = d_only[d_only[fine_col] == label]
            valid = sub.dropna(subset=["lo_Hz", "hi_Hz"])
            if valid.empty:
                print(f"  {str(label):24s}  (no freq-band data)")
                continue
            print(f"  {str(label)}  (n={len(valid)})")
            print(f"    low_Hz   "
                  f"min={valid['lo_Hz'].min():6.1f}  "
                  f"p25={valid['lo_Hz'].quantile(0.25):6.1f}  "
                  f"med={valid['lo_Hz'].median():6.1f}  "
                  f"p75={valid['lo_Hz'].quantile(0.75):6.1f}  "
                  f"max={valid['lo_Hz'].max():6.1f}")
            print(f"    high_Hz  "
                  f"min={valid['hi_Hz'].min():6.1f}  "
                  f"p25={valid['hi_Hz'].quantile(0.25):6.1f}  "
                  f"med={valid['hi_Hz'].median():6.1f}  "
                  f"p75={valid['hi_Hz'].quantile(0.75):6.1f}  "
                  f"max={valid['hi_Hz'].max():6.1f}")
            print(f"    bw_Hz    "
                  f"min={valid['bw_Hz'].min():6.1f}  "
                  f"p25={valid['bw_Hz'].quantile(0.25):6.1f}  "
                  f"med={valid['bw_Hz'].median():6.1f}  "
                  f"p75={valid['bw_Hz'].quantile(0.75):6.1f}  "
                  f"max={valid['bw_Hz'].max():6.1f}")
    else:
        print("  (no low_frequency/high_frequency columns)")

    # Q5: bimodality check on low_frequency — direct test for "two D types"
    if "low_frequency" in d_only.columns:
        print(f"\nlow_frequency distribution (D-class only) — coarse histogram:")
        lo_vals = pd.to_numeric(d_only["low_frequency"], errors="coerce").dropna()
        if not lo_vals.empty:
            # Bin in 5-Hz steps from 0 to 100, then 100+
            edges = list(range(0, 105, 5)) + [float("inf")]
            labels = [f"{edges[i]:>3.0f}-{edges[i+1]:>3.0f} Hz"
                      for i in range(len(edges) - 1)]
            cuts = pd.cut(lo_vals, bins=edges, labels=labels,
                          include_lowest=True, right=False)
            counts = cuts.value_counts().sort_index()
            mx = counts.max()
            for bin_label, n in counts.items():
                bar = "#" * int(40 * n / mx) if mx > 0 else ""
                print(f"  {bin_label:>14s}  {n:6d}  {bar}")
            print(f"\n  Look for bimodality (two peaks). A bimodal distribution "
                  f"is direct evidence of two acoustic subtypes.")

    return d_only, fine_col


def cross_split(d_train: pd.DataFrame, d_val: pd.DataFrame, fine_col: str) -> None:
    section("TRAIN vs VAL: fine-label coverage")
    if d_train.empty or d_val.empty:
        print("One side empty, skipping.")
        return

    train_counts = d_train[fine_col].value_counts()
    val_counts = d_val[fine_col].value_counts()
    all_labels = sorted(set(train_counts.index) | set(val_counts.index),
                        key=lambda x: -(int(train_counts.get(x, 0))
                                        + int(val_counts.get(x, 0))))

    print(f"\n  {'subtype':24s}  {'train':>8}  {'val':>8}  {'val/train':>10}")
    print(f"  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*10}")
    for label in all_labels:
        n_t = int(train_counts.get(label, 0))
        n_v = int(val_counts.get(label, 0))
        ratio = "n/a" if n_t == 0 else f"{n_v/n_t:.3f}"
        print(f"  {str(label):24s}  {n_t:>8d}  {n_v:>8d}  {ratio:>10}")

    train_only = set(train_counts.index) - set(val_counts.index)
    val_only = set(val_counts.index) - set(train_counts.index)
    if train_only:
        print(f"\n  WARNING: in TRAIN but not VAL: {train_only}")
    if val_only:
        print(f"  WARNING: in VAL but not TRAIN: {val_only} "
              f"(distribution shift, model literally cannot learn these)")
    if not train_only and not val_only:
        print(f"\n  Both sides cover the same labels.")


def cross_split_freqs(d_train: pd.DataFrame, d_val: pd.DataFrame) -> None:
    section("TRAIN vs VAL: frequency-band statistics (D-class)")
    for name, df in [("train", d_train), ("val", d_val)]:
        if df.empty or "low_frequency" not in df.columns:
            continue
        lo = pd.to_numeric(df["low_frequency"], errors="coerce").dropna()
        hi = pd.to_numeric(df["high_frequency"], errors="coerce").dropna()
        if lo.empty or hi.empty:
            continue
        print(f"\n  {name}:")
        print(f"    low_Hz   med={lo.median():6.1f}  "
              f"p10={lo.quantile(0.1):6.1f}  p90={lo.quantile(0.9):6.1f}")
        print(f"    high_Hz  med={hi.median():6.1f}  "
              f"p10={hi.quantile(0.1):6.1f}  p90={hi.quantile(0.9):6.1f}")


def main() -> None:
    print(textwrap.dedent(f"""
        D-class subtype diagnostic (v2)
        ================================
        Train datasets: {cfg.TRAIN_DATASETS}
        Val   datasets: {cfg.VAL_DATASETS}
    """).strip())

    d_train, fc_train = describe_split("train", cfg.TRAIN_DATASETS)
    d_val, fc_val = describe_split("val", cfg.VAL_DATASETS)

    if fc_train and fc_val and fc_train == fc_val and not d_train.empty and not d_val.empty:
        cross_split(d_train, d_val, fc_train)

    if not d_train.empty and not d_val.empty:
        cross_split_freqs(d_train, d_val)

    section("DECISION GUIDE")
    print(textwrap.dedent("""
        Read the histogram of low_frequency above.

          - SINGLE peak (e.g. one mode at 20-25 Hz):
              D-class is acoustically homogeneous. No specialisation
              possible. Pivot to recipe changes (oversampling, D-centred
              training windows, postprocessing tuning).

          - TWO clear peaks (e.g. one at 15-30 Hz, another at 50-100 Hz):
              Direct evidence of acoustic subtypes despite single label.
              Specialisation is feasible by SYNTHETIC SUBTYPING:
                1. Bin annotations by low_frequency threshold
                2. Train with 4-class output (split d into d_lowband and
                   d_highband)
                3. At eval, sum the two D probabilities back into one D
                   score before threshold tuning

          - TRAIN vs VAL frequency stats differ a lot:
              Distribution shift on D itself. Architecture changes will
              not bridge this. Consider per-site eval, or accept ceiling.

        Next: run `python visualize_d_subtypes.py` to see what the calls
        actually look like — confirms or contradicts the histogram story.
    """).strip())
    print()


if __name__ == "__main__":
    main()
