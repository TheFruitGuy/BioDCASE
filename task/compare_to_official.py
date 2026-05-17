"""
Compare the official Geldenhuys WhaleVAD checkpoint to your canonical
reproduction at the weight level.

The two checkpoints have:
    - Different layer naming
        (fbank/cnn_blocks vs filterbank/feat_extractor/residual_stack)
    - Different classifier widths
        (official: 7 classes; canonical: 3 classes)
    - A ``bb_proj.*`` bounding-box head in the official only

We remap the official's keys to canonical's naming, skip the bb_proj
keys, skip the classifier (incompatible shapes), and report per-layer
L2 difference and cosine similarity across all shared float tensors.

What the numbers can tell you:

    cosine ≈ 0.99+         identical (same training trajectory or
                           same recipe applied for similar duration)
    cosine 0.7–0.99        same direction; one was trained longer
                           or with different regularization
    cosine 0.4–0.7         shared init but meaningfully different
                           training (recipe difference or much
                           longer training)
    cosine < 0.4 or < 0    distinct training; either different
                           random init or substantially different
                           recipe / data

Usage
-----
    python compare_to_official.py \
        --official  WhaleVAD_ATBFL_3P-c6f6a07a.pt \
        --canonical runs/whalevad_20260502_175547/best_model.pt
"""

from __future__ import annotations
import argparse
from pathlib import Path

import torch


# ---------------------------------------------------------------------
# Key remapping: official → canonical
# Copied from load_official_checkpoint.py for self-containedness.
# ---------------------------------------------------------------------

def remap_official_to_canonical(state_dict: dict) -> dict:
    """Translate official-checkpoint key names into canonical's naming."""
    def rename(k: str) -> str:
        if k.startswith("fbank."):
            return k.replace("fbank.", "filterbank.")
        if k.startswith("cnn_blocks.0."):
            return k.replace("cnn_blocks.0.", "feat_extractor.")
        if k.startswith("cnn_blocks.1.blocks.0."):
            return k.replace("cnn_blocks.1.blocks.0.", "residual_stack.blocks.0.")
        if k.startswith("cnn_blocks.1.blocks.1."):
            return k.replace("cnn_blocks.1.blocks.1.", "residual_stack.blocks.1.")
        return k

    out = {}
    skipped = []
    for k, v in state_dict.items():
        if k.startswith("bb_proj"):
            skipped.append(k)
            continue
        out[rename(k)] = v
    if skipped:
        print(f"  Skipped {len(skipped)} bb_proj.* keys (bounding-box head, "
              f"not part of canonical architecture)")
    return out


# ---------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------

def summarize_metadata(obj, label: str) -> dict:
    """Print all non-state_dict keys, return the state_dict."""
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    if not isinstance(obj, dict):
        print(f"!! unexpected ckpt type: {type(obj).__name__}")
        return {}

    print(f"Top-level keys: {list(obj.keys())}")
    for k, v in obj.items():
        if k in ("model_state_dict", "state_dict"):
            continue
        if torch.is_tensor(v):
            print(f"  {k:24s}: tensor{tuple(v.shape)} {v.tolist()}")
        elif isinstance(v, (int, float, str, bool)):
            print(f"  {k:24s}: {v!r}")
        elif isinstance(v, (list, tuple)):
            print(f"  {k:24s}: {v}")
        elif isinstance(v, dict):
            print(f"  {k:24s}: dict with {len(v)} keys")
            for sk, sv in list(v.items())[:10]:
                if isinstance(sv, (int, float, str, bool)):
                    print(f"      {sk}: {sv!r}")
                elif torch.is_tensor(sv):
                    print(f"      {sk}: tensor{tuple(sv.shape)}")
                else:
                    print(f"      {sk}: <{type(sv).__name__}>")
            if len(v) > 10:
                print(f"      ... ({len(v) - 10} more)")
        else:
            print(f"  {k:24s}: <{type(v).__name__}>")

    sd = obj.get("model_state_dict") or obj.get("state_dict")
    if sd is None and all(torch.is_tensor(v) for v in obj.values()):
        sd = obj
    if sd is None:
        print("!! no state_dict found")
        return {}
    return sd


def compare(sd_official: dict, sd_canonical: dict):
    """Per-layer L2 diff and cosine similarity."""
    print(f"\n{'=' * 70}\nWeight-level comparison\n{'=' * 70}")

    only_off = set(sd_official.keys()) - set(sd_canonical.keys())
    only_can = set(sd_canonical.keys()) - set(sd_official.keys())
    shared = sorted(set(sd_official.keys()) & set(sd_canonical.keys()))

    if only_off:
        print(f"Only in official (after remap): {len(only_off)}")
        for k in sorted(only_off)[:10]:
            print(f"  - {k}")
    if only_can:
        print(f"Only in canonical: {len(only_can)}")
        for k in sorted(only_can)[:10]:
            print(f"  - {k}")
    print(f"Shared keys: {len(shared)}")

    print(f"\n  {'layer':<55s} {'shape':<22s} {'L2':>10s} {'cos':>8s}")
    rows = []
    for k in shared:
        a, b = sd_official[k], sd_canonical[k]
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            continue
        if a.shape != b.shape:
            # Classifier shape mismatch (7 vs 3) — note and continue.
            print(f"  {k:<55s} shapes differ: official={tuple(a.shape)} "
                  f"vs canonical={tuple(b.shape)}  (skipped)")
            continue
        if not a.is_floating_point():
            continue
        a32 = a.flatten().float()
        b32 = b.flatten().float()
        delta = (a32 - b32).norm().item()
        cos = torch.nn.functional.cosine_similarity(
            a32.unsqueeze(0), b32.unsqueeze(0)
        ).item()
        rows.append((k, tuple(a.shape), delta, cos))
        print(f"  {k:<55s} {str(tuple(a.shape)):<22s} {delta:>10.3e} {cos:>8.4f}")

    if rows:
        # Group-level summary: split layers by where they are in the
        # architecture, so we can see if (say) the conv stack is similar
        # but the LSTM is different.
        print(f"\n{'=' * 70}\nGrouped summary (mean cosine similarity per stage)\n{'=' * 70}")
        groups = [
            ("filterbank",         "filterbank"),
            ("feat_extractor",     "feat_extractor"),
            ("residual_stack",     "residual_stack"),
            ("feat_proj",          "feat_proj"),
            ("lstm",               "lstm"),
            ("classifier",         "classifier"),
        ]
        for label, prefix in groups:
            in_group = [(k, s, d, c) for (k, s, d, c) in rows if k.startswith(prefix)]
            if not in_group:
                print(f"  {label:<20s}  (no shared float-tensor keys)")
                continue
            mean_cos = sum(c for _, _, _, c in in_group) / len(in_group)
            min_cos  = min(c for _, _, _, c in in_group)
            print(f"  {label:<20s}  mean cos = {mean_cos:+.4f}  "
                  f"min cos = {min_cos:+.4f}  ({len(in_group)} layers)")

        overall_mean = sum(c for _, _, _, c in rows) / len(rows)
        print(f"  {'OVERALL':<20s}  mean cos = {overall_mean:+.4f}")
        print()
        print("Interpretation guide:")
        print("  - High cosine, low L2  → models agree on what these weights look like")
        print("  - High cosine, high L2 → same direction, different scale")
        print("  - Low cosine           → genuinely different training outcomes")
        print("  - The most informative groups are the conv stack (filterbank,")
        print("    feat_extractor, residual_stack) and the LSTM, since they")
        print("    encode the bulk of the learned features.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", type=Path, required=True,
                    help="WhaleVAD_ATBFL_3P-c6f6a07a.pt")
    ap.add_argument("--canonical", type=Path, required=True,
                    help="runs/whalevad_20260502_175547/best_model.pt")
    args = ap.parse_args()

    print(f"Loading official:  {args.official}")
    off = torch.load(args.official, map_location="cpu", weights_only=False)

    print(f"Loading canonical: {args.canonical}")
    can = torch.load(args.canonical, map_location="cpu", weights_only=False)

    sd_off = summarize_metadata(off, "OFFICIAL (Geldenhuys release)")
    sd_can = summarize_metadata(can, "CANONICAL (your 2026-05-02 reproduction)")

    print("\nRemapping official's layer names to canonical's naming...")
    sd_off_remap = remap_official_to_canonical(sd_off)

    compare(sd_off_remap, sd_can)


if __name__ == "__main__":
    main()
