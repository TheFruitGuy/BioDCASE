"""
diagnose_d_subtypes.py
======================

Decomposes the 3-class "d" label into its underlying fine-grained labels
(presumably distinguishing fin-whale and blue-whale D-calls if your
annotation source did so) and prints distribution statistics across
training and validation sites.

This script answers three questions:
1. Does the annotation schema actually distinguish D subtypes?
2. Is each subtype present in BOTH train and val? (If not, no architecture
   change can rescue the missing subtype.)
3. Do the subtypes have meaningfully different durations?

Run from BioDCASE/task/:

    python diagnose_d_subtypes.py

Output: prints to stdout. No PNGs.
"""

from __future__ import annotations

import sys
import textwrap
import pandas as pd

import config as cfg
from dataset import load_annotations


def section(title: str) -> None:
    """Print a clean header so output is scannable."""
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def describe_subtypes(name: str, datasets: list[str]) -> pd.DataFrame:
    """Load annotations for `datasets`, print subtype breakdown, return d-only frame."""
    section(f"{name.upper()}: subtype decomposition of 3-class 'd'")

    ann = load_annotations(datasets)

    # Show what columns exist so user can adapt the script if their schema differs.
    print(f"\nAnnotation columns available: {list(ann.columns)}")
    print(f"Total annotations: {len(ann)}")

    if "label_3class" not in ann.columns:
        print("\nERROR: no 'label_3class' column in annotations. Aborting.")
        sys.exit(1)

    # Some pipelines store the original label under 'label', others under
    # 'label_orig' or 'label_7class'. Pick whichever exists; fall back to
    # 'label_3class' itself (in which case there is no subtype distinction).
    candidates = ["label", "label_7class", "label_orig", "label_full"]
    fine_col = None
    for c in candidates:
        if c in ann.columns:
            fine_col = c
            break

    if fine_col is None or fine_col == "label_3class":
        print("\nNo fine-grained label column found. Subtype analysis impossible "
              "without re-annotation. Stopping here.")
        return pd.DataFrame()

    print(f"Using '{fine_col}' as the fine-grained label column.\n")

    d_only = ann[ann["label_3class"] == "d"].copy()
    print(f"Rows with label_3class == 'd': {len(d_only)}")

    if len(d_only) == 0:
        print("No D-class annotations found.")
        return d_only

    # Q1: what fine labels collapse into 3-class 'd'?
    print(f"\nFine-grained labels that map to 'd':")
    counts = d_only[fine_col].value_counts(dropna=False)
    for label, n in counts.items():
        pct = 100.0 * n / len(d_only)
        print(f"  {str(label):20s}  n={n:6d}  ({pct:5.1f}%)")

    if len(counts) == 1:
        print("\nOnly ONE fine label maps to 'd' — the annotations do not "
              "distinguish D subtypes. Specialisation by retraining with "
              "extra classes is impossible without re-annotation.")
        return d_only

    # Q2: per-site breakdown
    print(f"\nPer-site × per-subtype counts:")
    pivot = d_only.groupby(["dataset", fine_col]).size().unstack(fill_value=0)
    print(pivot.to_string())
    print(f"\nPer-site totals:")
    print(pivot.sum(axis=1).sort_values(ascending=False).to_string())

    # Q3: duration statistics per subtype
    print(f"\nDuration (seconds) per subtype:")
    if "start_s" in d_only.columns and "end_s" in d_only.columns:
        d_only["duration_s"] = d_only["end_s"] - d_only["start_s"]
        for label in counts.index:
            sub = d_only[d_only[fine_col] == label]
            durs = sub["duration_s"]
            print(f"  {str(label):20s}  "
                  f"min={durs.min():.2f}  "
                  f"p25={durs.quantile(0.25):.2f}  "
                  f"median={durs.median():.2f}  "
                  f"p75={durs.quantile(0.75):.2f}  "
                  f"max={durs.max():.2f}  "
                  f"mean={durs.mean():.2f}")
    else:
        print("  (no start_s/end_s columns — skipping)")

    return d_only


def cross_split_comparison(d_train: pd.DataFrame, d_val: pd.DataFrame, fine_col: str) -> None:
    """Compare which subtypes appear in train vs val — the key generalisation question."""
    section("TRAIN vs VAL: subtype coverage")

    if d_train.empty or d_val.empty:
        print("One side empty, skipping comparison.")
        return

    train_counts = d_train[fine_col].value_counts()
    val_counts = d_val[fine_col].value_counts()

    all_labels = sorted(set(train_counts.index) | set(val_counts.index),
                        key=lambda x: -(train_counts.get(x, 0) + val_counts.get(x, 0)))

    print(f"\n  {'subtype':20s}  {'train':>8}  {'val':>8}  {'val/train':>10}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*10}")
    for label in all_labels:
        n_t = int(train_counts.get(label, 0))
        n_v = int(val_counts.get(label, 0))
        ratio = "n/a" if n_t == 0 else f"{n_v/n_t:.3f}"
        print(f"  {str(label):20s}  {n_t:>8d}  {n_v:>8d}  {ratio:>10}")

    print()
    train_only = set(train_counts.index) - set(val_counts.index)
    val_only = set(val_counts.index) - set(train_counts.index)
    if train_only:
        print(f"  WARNING: subtypes seen in TRAIN but not VAL: {train_only}")
        print(f"           (model is wasting capacity learning these)")
    if val_only:
        print(f"  WARNING: subtypes seen in VAL but not TRAIN: {val_only}")
        print(f"           (model literally cannot detect these — distribution shift)")
    if not train_only and not val_only:
        print("  Both sides contain the same subtypes. Specialisation is feasible.")


def main() -> None:
    print(textwrap.dedent(f"""
        D-class subtype diagnostic
        ==========================
        Train datasets: {cfg.TRAIN_DATASETS}
        Val   datasets: {cfg.VAL_DATASETS}
    """).strip())

    d_train = describe_subtypes("train", cfg.TRAIN_DATASETS)
    d_val = describe_subtypes("val", cfg.VAL_DATASETS)

    # Pick the fine column from the train frame (must match val if both have it)
    candidates = ["label", "label_7class", "label_orig", "label_full"]
    fine_col = None
    for c in candidates:
        if c in d_train.columns:
            fine_col = c
            break

    if fine_col is not None and not d_train.empty and not d_val.empty:
        cross_split_comparison(d_train, d_val, fine_col)

    section("DECISION")
    print(textwrap.dedent("""
        Interpretation guide:

          - If only ONE subtype mapped to 'd':
              No specialisation possible. Falls back to recipe changes
              (oversampling, D-centred windowing, postprocessing tuning).

          - If MULTIPLE subtypes BUT one is missing from train OR val:
              Distribution shift is the bottleneck. Architecture changes
              will not help. Consider per-site evaluation, accept ceiling
              for the missing subtype, focus elsewhere.

          - If MULTIPLE subtypes AND both sides have all of them:
              Specialisation is feasible. Cheapest first move is 4-class
              training (split d into d_subtype_a and d_subtype_b) with
              3-class eval (sum the two D probabilities back). Run the
              spectrogram visualisation to confirm the subtypes look
              acoustically different before committing.

        Next: run `python visualize_d_subtypes.py` to see what the calls
        actually look like.
    """).strip())
    print()


if __name__ == "__main__":
    main()
