"""
Organise the ATBFL dataset download into the expected folder structure.

The Zenodo download may come as a flat structure or with slightly different
naming. This script normalises everything into:

    data/
    ├── ballenyisland2015/
    │   ├── *.wav
    │   └── annotation.csv
    ├── casey2014/
    │   └── ...

Usage:
    python organise_data.py --download_dir ./atbfl_download --data_root ./data
"""

import argparse
import csv
import shutil
from pathlib import Path
from collections import defaultdict


EXPECTED_DATASETS = [
    "ballenyisland2015", "ballenyislands2015",  # handle both spellings
    "casey2014", "casey2017",
    "elephantisland2013", "elephantislands2013",
    "elephantisland2014", "elephantislands2014",
    "greenwich2015",
    "kerguelen2005", "kerguelen2014", "kerguelen2015",
    "maudrise2014",
    "rosssea2014",
]

# Normalise dataset names (the ATBFL uses both singular and plural)
NAME_MAP = {
    "ballenyislands2015": "ballenyisland2015",
    "elephantislands2013": "elephantisland2013",
    "elephantislands2014": "elephantisland2014",
}


def find_wav_files(download_dir: Path) -> dict[str, list[Path]]:
    """Scan the download directory and group WAV files by dataset name."""
    grouped = defaultdict(list)

    for wav in download_dir.rglob("*.wav"):
        # Try to infer dataset from parent directory name
        parent = wav.parent.name.lower()
        if parent in EXPECTED_DATASETS or parent in NAME_MAP:
            ds = NAME_MAP.get(parent, parent)
            grouped[ds].append(wav)
        else:
            # Try to infer from filename or grandparent
            for ancestor in wav.parents:
                name = ancestor.name.lower()
                if name in EXPECTED_DATASETS or name in NAME_MAP:
                    ds = NAME_MAP.get(name, name)
                    grouped[ds].append(wav)
                    break

    return dict(grouped)


def find_annotation_files(download_dir: Path) -> list[Path]:
    """Find all CSV files that look like annotations."""
    candidates = []
    for csv_path in download_dir.rglob("*.csv"):
        # Check if it has the expected columns
        try:
            with open(csv_path) as f:
                reader = csv.reader(f)
                header = next(reader)
                header_lower = [h.strip().lower() for h in header]
                if "annotation" in header_lower and "start_datetime" in header_lower:
                    candidates.append(csv_path)
        except Exception:
            continue
    return candidates


def split_annotations_by_dataset(csv_paths: list[Path]) -> dict[str, list[dict]]:
    """Read all annotation CSVs and group rows by dataset."""
    by_dataset = defaultdict(list)

    for csv_path in csv_paths:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ds = row.get("dataset", "").strip().lower()
                ds = NAME_MAP.get(ds, ds)
                if ds:
                    by_dataset[ds].append(row)

    return dict(by_dataset)


def organise(download_dir: str, data_root: str):
    download_path = Path(download_dir)
    output_path = Path(data_root)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {download_path} for WAV files...")
    wav_groups = find_wav_files(download_path)
    for ds, files in sorted(wav_groups.items()):
        print(f"  {ds}: {len(files)} WAV files")

    print(f"\nScanning for annotation CSVs...")
    ann_csvs = find_annotation_files(download_path)
    print(f"  Found {len(ann_csvs)} annotation file(s)")

    ann_by_dataset = split_annotations_by_dataset(ann_csvs)
    for ds, rows in sorted(ann_by_dataset.items()):
        print(f"  {ds}: {len(rows)} annotations")

    # Copy/link files into organised structure
    print(f"\nOrganising into {output_path}/...")
    for ds, wav_files in sorted(wav_groups.items()):
        ds_dir = output_path / ds
        ds_dir.mkdir(exist_ok=True)

        for wav in wav_files:
            dest = ds_dir / wav.name
            if not dest.exists():
                # Symlink to save disk space; use shutil.copy2 if you prefer copies
                try:
                    dest.symlink_to(wav.resolve())
                except OSError:
                    shutil.copy2(wav, dest)

        # Write annotation CSV for this dataset
        if ds in ann_by_dataset:
            ann_path = ds_dir / "annotation.csv"
            rows = ann_by_dataset[ds]
            if rows:
                fieldnames = list(rows[0].keys())
                with open(ann_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  {ds}: {len(wav_files)} wavs, {len(rows)} annotations → {ds_dir}")
            else:
                print(f"  {ds}: {len(wav_files)} wavs, no annotations → {ds_dir}")
        else:
            print(f"  {ds}: {len(wav_files)} wavs, no annotation file found → {ds_dir}")

    # Check for datasets with annotations but no WAVs
    for ds in ann_by_dataset:
        if ds not in wav_groups:
            print(f"  WARNING: annotations for {ds} but no WAV files found")

    print("\nDone! Verify with:")
    print(f"  python verify_data.py --data_root {data_root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download_dir", type=str, required=True,
                   help="Path to the raw Zenodo download")
    p.add_argument("--data_root", type=str, default="./data",
                   help="Where to create the organised structure")
    args = p.parse_args()
    organise(args.download_dir, args.data_root)
