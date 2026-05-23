#!/usr/bin/env python3
"""
download_aadc.py
================

Download AADC long-term acoustic recordings (doi:10.26179/fhsv-ft93) and decimate
them from 12 kHz native to a target sample rate (default 250 Hz, matching the
BioDCASE 2025 / WhaleVAD pipeline).

Each WAV is streamed from S3 into memory, decimated with anti-aliased two-stage
IIR (8x then 6x = 48x), and written as 16-bit PCM WAV to disk. The full-rate
12 kHz file is never written to disk.

Test-set sites (Kerguelen2020, DDU2021) are HARD-BLOCKED — the script refuses to
download them to prevent accidental data leakage into pretraining.

Output layout (compatible with dataset.py manifest):

    <output-dir>/
      DDU2018/
        201_2018-01-15_03-00-00.wav   (already decimated to target SR)
        ...
      Kerguelen2018/
        ...

Usage
-----
    export AADC_ACCESS_KEY=...
    export AADC_SECRET_KEY=...
    python download_aadc.py \\
        --datasets DDU2018 DDU2019 Kerguelen2018 Kerguelen2019 \\
        --output-dir /var/home/matthias-nagl/AMP_data/ssl_pretrain/audio \\
        --workers 4

Notes
-----
* Credentials from AADC's email expire (typically 7 days). Re-request and re-export
  each session.
* Endpoint is S3-compatible; signature v4 is required.
* Files that arrive at a non-nominal sample rate (anything other than 12 kHz) are
  detected, the decimation factor recomputed, and the file processed if a clean
  factorisation into stages of <=13 exists. Otherwise the file is skipped with a
  warning so you can investigate.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
import soundfile as sf
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError
from scipy.signal import decimate
from tqdm import tqdm

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

ENDPOINT = "https://transfer.data.aad.gov.au"
BUCKET = "aadc-datasets"

NATIVE_SR = 12_000        # AADC nominal sample rate
DEFAULT_TARGET_SR = 250   # BioDCASE / WhaleVAD pipeline rate

# Decimation: 12000 / 250 = 48. scipy advises q <= 13 per stage for IIR.
# 8 * 6 = 48, both stages within limits. Polyphase IIR with zero_phase=True is
# applied forward + backward, so phase is preserved (suitable for offline use).
DECIM_STAGES_DEFAULT = (8, 6)

# Test set — refuse to download these. The whole point of an isolated test set
# is that no part of the pipeline ever sees them during training/pretraining.
QUARANTINE = {
    "Kerguelen2020", "kerguelen2020", "KERGUELEN2020",
    "DDU2021",       "ddu2021",       "Ddu2021",
}

# Short name -> S3 key prefix (folder in the bucket).
DATASET_PREFIXES = {
    "Casey2014":     "AAS_4102_longTermAcousticRecordings_Casey2014/",
    "Casey2016":     "AAS_4102_longTermAcousticRecordings_Casey2016/",
    "Casey2017":     "AAS_4102_longTermAcousticRecordings_Casey2017/",
    "Casey2018":     "AAS_4102_longTermAcousticRecordings_Casey2018/",
    "Casey2019":     "AAS_4102_longTermAcousticRecordings_Casey2019/",
    "Casey2020":     "AAS_4102_longTermAcousticRecordings_Casey2020/",
    "Casey2022":     "AAS_4102_longTermAcousticRecordings_Casey2022/",
    "Casey2023":     "AAS_4102_longTermAcousticRecordings_Casey2023/",
    "Casey2024":     "AAS_4102_longTermAcousticRecordings_Casey2024/",
    "Kerguelen2014": "AAS_4102_longTermAcousticRecordings_Kerguelen2014/",
    "Kerguelen2015": "AAS_4102_longTermAcousticRecordings_Kerguelen2015/",
    "Kerguelen2016": "AAS_4102_longTermAcousticRecordings_Kerguelen2016/",
    "Kerguelen2017": "AAS_4102_longTermAcousticRecordings_Kerguelen2017/",
    "Kerguelen2018": "AAS_4102_longTermAcousticRecordings_Kerguelen2018/",
    "Kerguelen2019": "AAS_4102_longTermAcousticRecordings_Kerguelen2019/",
    "Kerguelen2021": "AAS_4102_longTermAcousticRecordings_Kerguelen2021/",
    "Kerguelen2023": "AAS_4102_longTermAcousticRecordings_Kerguelen2023/",
    "Kerguelen2024": "AAS_4102_longTermAcousticRecordings_Kerguelen2024/",
    "DDU2018":       "AAS_4102_longTermAcousticRecordings_DDU2018/",
    "DDU2019":       "AAS_4102_longTermAcousticRecordings_DDU2019/",
    "Prydz2013":     "AAS_4102_longTermAcousticRecordings_Prydz2013/",
    "Scott2019":     "AAS_4102_longTermAcousticRecordings_Scott2019/",
}


# ----------------------------------------------------------------------
# S3 client
# ----------------------------------------------------------------------

def make_client(access_key: str, secret_key: str):
    """Build a boto3 client for the AADC S3-compatible endpoint."""
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "adaptive"},
            connect_timeout=30,
            read_timeout=600,
        ),
        region_name="us-east-1",  # placeholder; AADC ignores it
    )


def list_wavs(client, prefix: str) -> list[tuple[str, int]]:
    """Return [(key, size_bytes), ...] for every .wav under `prefix`."""
    paginator = client.get_paginator("list_objects_v2")
    keys: list[tuple[str, int]] = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.lower().endswith(".wav"):
                keys.append((k, obj["Size"]))
    return keys


# ----------------------------------------------------------------------
# Decimation
# ----------------------------------------------------------------------

def _factor_stages(ratio: int, max_q: int = 13) -> list[int]:
    """Factor `ratio` into a sequence of stages, each <= max_q.

    Greedy from the largest divisor downward. Used as fallback when an input
    file has an unexpected sample rate.
    """
    if ratio <= 1:
        return []
    if ratio <= max_q:
        return [ratio]
    for q in range(max_q, 1, -1):
        if ratio % q == 0:
            return [q] + _factor_stages(ratio // q, max_q)
    raise ValueError(f"Cannot factor decimation ratio {ratio} into stages <= {max_q}")


def decimate_audio(
    x: np.ndarray, sr_in: int, sr_out: int, stages: tuple[int, ...] | None = None
) -> np.ndarray:
    """Anti-aliased multi-stage decimation. `sr_in / sr_out` must be integer."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    if sr_in % sr_out != 0:
        raise ValueError(f"Non-integer decimation ratio: {sr_in} / {sr_out}")
    ratio = sr_in // sr_out

    if stages is None:
        stages = tuple(_factor_stages(ratio))
    if int(np.prod(stages)) != ratio:
        raise ValueError(f"Stages {stages} (prod={int(np.prod(stages))}) != ratio {ratio}")

    y = x.astype(np.float64, copy=False)
    for q in stages:
        # zero_phase=True → forward+backward filtering, no group delay
        y = decimate(y, q, ftype="iir", zero_phase=True)
    return y.astype(np.float32)


# ----------------------------------------------------------------------
# Per-file pipeline
# ----------------------------------------------------------------------

def process_one_file(client, key: str, out_path: Path, target_sr: int) -> str:
    """Stream `key` from S3, decimate to `target_sr`, write `out_path`. Return status."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skip"

    try:
        resp = client.get_object(Bucket=BUCKET, Key=key)
        body = resp["Body"].read()
    except (ClientError, BotoCoreError) as e:
        return f"error[s3]: {key}: {e}"

    try:
        with io.BytesIO(body) as buf:
            x, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception as e:
        return f"error[wav]: {key}: {e}"

    # If multichannel (shouldn't happen for AADC mono hydrophones), take ch 0.
    if x.ndim > 1:
        x = x[:, 0]

    try:
        if sr == NATIVE_SR and target_sr == DEFAULT_TARGET_SR:
            stages: tuple[int, ...] | None = DECIM_STAGES_DEFAULT
        else:
            stages = None  # let decimate_audio derive stages
        y = decimate_audio(x, sr, target_sr, stages=stages)
    except Exception as e:
        return f"error[decim]: {key}: sr={sr}: {e}"

    # Float32 in [-1, 1] → 16-bit PCM. Clip defensively in case of filter overshoot.
    np.clip(y, -1.0, 1.0, out=y)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        sf.write(str(tmp_path), y, target_sr, subtype="PCM_16", format="WAV")
        tmp_path.replace(out_path)
    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return f"error[write]: {key}: {e}"

    return "ok"


# ----------------------------------------------------------------------
# Per-dataset driver
# ----------------------------------------------------------------------

def process_dataset(
    client, dataset: str, output_root: Path, target_sr: int, workers: int
):
    if dataset in QUARANTINE:
        print(f"\n!! REFUSING: {dataset} is part of the test set. Skipping.\n")
        return

    if dataset not in DATASET_PREFIXES:
        print(f"\n!! ERROR: unknown dataset {dataset!r}.")
        print(f"   Choices: {sorted(DATASET_PREFIXES)}\n")
        return

    prefix = DATASET_PREFIXES[dataset]
    out_dir = output_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {dataset} ({prefix}) ===")
    t0 = time.monotonic()
    print(f"  Listing s3://{BUCKET}/{prefix} ...")
    try:
        keys = list_wavs(client, prefix)
    except (ClientError, BotoCoreError) as e:
        print(f"  ERROR listing: {e}")
        print("  (Likely: expired credentials. Re-request from AADC and re-export "
              "AADC_ACCESS_KEY / AADC_SECRET_KEY.)")
        return

    if not keys:
        print(f"  No .wav files found under {prefix}")
        return

    total_bytes = sum(s for _, s in keys)
    expected_out_gb = total_bytes / 1e9 / (NATIVE_SR / target_sr)
    print(f"  {len(keys)} files, {total_bytes / 1e9:.1f} GB at 12 kHz "
          f"→ ~{expected_out_gb:.2f} GB after decimation to {target_sr} Hz")

    def _job(k_size):
        k, _ = k_size
        out_path = out_dir / Path(k).name
        return process_one_file(client, k, out_path, target_sr)

    ok = skip = err = 0
    err_msgs: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_job, ks) for ks in keys]
        with tqdm(total=len(futs), desc=dataset, unit="file") as bar:
            for f in as_completed(futs):
                r = f.result()
                if r == "ok":
                    ok += 1
                elif r == "skip":
                    skip += 1
                else:
                    err += 1
                    err_msgs.append(r)
                    if len(err_msgs) <= 5:
                        tqdm.write(r)
                bar.update(1)

    dt = time.monotonic() - t0
    print(f"  done: ok={ok} skip={skip} err={err}  ({dt:.0f} s)")
    if err > 5:
        print(f"  (showed first 5 of {err} errors; re-running will retry the failures)")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Download AADC acoustic mooring datasets and decimate to a target SR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help=("Dataset short names. Choices: " + ", ".join(sorted(DATASET_PREFIXES))),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for decimated output. Subfolders created per dataset.",
    )
    p.add_argument(
        "--access-key",
        default=os.environ.get("AADC_ACCESS_KEY"),
        help="S3 access key (default: $AADC_ACCESS_KEY)",
    )
    p.add_argument(
        "--secret-key",
        default=os.environ.get("AADC_SECRET_KEY"),
        help="S3 secret key (default: $AADC_SECRET_KEY)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel download/decimate workers per dataset (default: 4)",
    )
    p.add_argument(
        "--target-sr",
        type=int,
        default=DEFAULT_TARGET_SR,
        help=f"Target sample rate in Hz (default: {DEFAULT_TARGET_SR})",
    )
    args = p.parse_args()

    if not args.access_key or not args.secret_key:
        sys.exit(
            "ERROR: AADC credentials not provided.\n"
            "       Either pass --access-key/--secret-key or export\n"
            "       AADC_ACCESS_KEY and AADC_SECRET_KEY in your shell."
        )

    # Hard-block test-set sites at request parse time, before contacting S3.
    bad = [d for d in args.datasets if d in QUARANTINE]
    if bad:
        sys.exit(
            f"REFUSING: requested test-set datasets {bad}. These are the BioDCASE\n"
            "          evaluation sites; downloading them into the pretraining\n"
            "          tree risks data leakage. Remove them from --datasets.\n"
            "          (If they're already on disk somewhere, ensure they live in\n"
            "           an isolated quarantine directory not walked by any manifest.)"
        )

    # Validate dataset names early so a typo doesn't waste a long run.
    unknown = [d for d in args.datasets if d not in DATASET_PREFIXES]
    if unknown:
        sys.exit(
            f"ERROR: unknown dataset(s) {unknown}.\n"
            f"       Choices: {sorted(DATASET_PREFIXES)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {args.output_dir.resolve()}")
    print(f"Target SR:   {args.target_sr} Hz")
    print(f"Workers:     {args.workers}")
    print(f"Datasets:    {args.datasets}")

    client = make_client(args.access_key, args.secret_key)

    for d in args.datasets:
        process_dataset(client, d, args.output_dir, args.target_sr, args.workers)

    print("\nAll requested datasets processed.")


if __name__ == "__main__":
    main()
