"""
Verify the dataset is correctly organised and print summary stats.

Usage:
    python verify_data.py --data_root ./data
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import pandas as pd


TRAIN_SETS = [
    "ballenyisland2015", "casey2014", "elephantisland2013",
    "elephantisland2014", "greenwich2015", "kerguelen2005",
    "maudrise2014", "rosssea2014",
]
VAL_SETS = ["casey2017", "kerguelen2014", "kerguelen2015"]
EVAL_SETS = ["kerguelen2020", "ddu2021"]  # released June 2026

CALL_TYPES = ["bma", "bmb", "bmz", "bmd", "bpd", "bp20", "bp20plus"]


def check_dataset(data_root: Path, ds_name: str, require_annotations: bool = True) -> dict:
    ds_dir = data_root / ds_name
    result = {"name": ds_name, "exists": ds_dir.exists()}

    if not ds_dir.exists():
        return result

    # Count WAVs
    wavs = list(ds_dir.glob("*.wav"))
    result["n_wavs"] = len(wavs)

    # Check annotation file
    ann_path = ds_dir / "annotation.csv"
    result["has_annotations"] = ann_path.exists()

    if ann_path.exists():
        try:
            df = pd.read_csv(ann_path)
            result["n_annotations"] = len(df)

            # Check required columns
            required_cols = {"filename", "annotation", "start_datetime", "end_datetime"}
            present = set(df.columns)
            result["missing_columns"] = list(required_cols - present)

            # Call type distribution
            if "annotation" in df.columns:
                counts = df["annotation"].value_counts().to_dict()
                result["call_counts"] = counts

                # Check for unexpected labels
                unexpected = set(counts.keys()) - set(CALL_TYPES)
                if unexpected:
                    result["unexpected_labels"] = list(unexpected)

            # Check filenames match actual WAVs
            if "filename" in df.columns:
                ann_files = set(df["filename"].unique())
                wav_files = set(w.name for w in wavs)
                missing = ann_files - wav_files
                if missing:
                    result["missing_wav_files"] = list(missing)[:5]  # first 5

        except Exception as e:
            result["annotation_error"] = str(e)

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    args = p.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"ERROR: {data_root} does not exist")
        sys.exit(1)

    all_ok = True

    for split_name, datasets, need_ann in [
        ("TRAINING", TRAIN_SETS, True),
        ("VALIDATION", VAL_SETS, True),
        ("EVALUATION", EVAL_SETS, False),
    ]:
        print(f"\n{'='*60}")
        print(f"  {split_name} SET")
        print(f"{'='*60}")

        total_wavs = 0
        total_anns = 0

        for ds in datasets:
            r = check_dataset(data_root, ds, need_ann)

            if not r["exists"]:
                status = "MISSING" if need_ann else "not yet available"
                print(f"  {ds}: {status}")
                if need_ann:
                    all_ok = False
                continue

            n_wavs = r.get("n_wavs", 0)
            total_wavs += n_wavs
            status_parts = [f"{n_wavs} wavs"]

            if r.get("has_annotations"):
                n_ann = r.get("n_annotations", 0)
                total_anns += n_ann
                status_parts.append(f"{n_ann} annotations")

                if r.get("call_counts"):
                    counts = r["call_counts"]
                    breakdown = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
                    status_parts.append(f"[{breakdown}]")

                if r.get("missing_columns"):
                    status_parts.append(f"MISSING COLS: {r['missing_columns']}")
                    all_ok = False

                if r.get("unexpected_labels"):
                    status_parts.append(f"UNEXPECTED: {r['unexpected_labels']}")

                if r.get("missing_wav_files"):
                    n_missing = len(r["missing_wav_files"])
                    status_parts.append(f"{n_missing} referenced wavs not found")
                    all_ok = False
            elif need_ann:
                status_parts.append("NO ANNOTATION FILE")
                all_ok = False

            print(f"  {ds}: {' | '.join(status_parts)}")

        print(f"  ---")
        print(f"  Total: {total_wavs} wav files, {total_anns} annotations")

    print(f"\n{'='*60}")
    if all_ok:
        print("  ALL CHECKS PASSED — ready to train!")
    else:
        print("  ISSUES FOUND — fix the above before training")
    print(f"{'='*60}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
