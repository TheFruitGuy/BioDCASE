"""
Inspect and compare .pt checkpoints.

Usage:
    python inspect_checkpoint.py runs/whalevad_20260502_175547/best_model.pt
    python inspect_checkpoint.py CKPT_OLD CKPT_NEW          # compare two
    python inspect_checkpoint.py WhaleVAD_ATBFL_3P-c6f6a07a.pt --official

For a single checkpoint, prints:
    - all keys at the top level (metadata vs state_dict)
    - any saved hyperparameters / thresholds / epoch / best_f1
    - per-layer shape, dtype, and basic stats (mean, std, abs-max, norm)
For two checkpoints, additionally prints:
    - which keys exist only in one (renamed layers, missing modules)
    - per-shared-key L2 difference and cosine similarity
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch


def load_ckpt(path: Path):
    """torch.load with sensible defaults. Falls back if weights_only=True rejects."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"!! load failed: {e}", file=sys.stderr)
        raise


def summarise_one(path: Path, label: str = "ckpt") -> dict:
    """Print metadata + per-layer stats. Returns the state_dict for further diffing."""
    print(f"\n{'=' * 70}\n{label}: {path}\n{'=' * 70}")
    obj = load_ckpt(path)

    # Top-level keys
    if isinstance(obj, dict):
        print(f"Top-level keys: {list(obj.keys())}")
        # Surface saved hyperparameters / metadata
        for k, v in obj.items():
            if k == "model_state_dict":
                continue
            if torch.is_tensor(v):
                print(f"  {k:24s}: tensor{tuple(v.shape)}  {v.tolist()}")
            elif isinstance(v, (int, float, str, bool, list, tuple, dict)):
                if isinstance(v, dict) and len(v) > 5:
                    print(f"  {k:24s}: dict with {len(v)} keys: {list(v.keys())[:8]} ...")
                else:
                    print(f"  {k:24s}: {v!r}")
            else:
                print(f"  {k:24s}: {type(v).__name__}")
        sd = obj.get("model_state_dict")
        if sd is None and "state_dict" in obj:
            sd = obj["state_dict"]
        if sd is None:
            # Maybe the whole thing IS the state_dict (no wrapper)
            if all(torch.is_tensor(v) for v in obj.values()):
                sd = obj
            else:
                print("!! no state_dict found"); return {}
    else:
        print(f"!! unexpected ckpt type: {type(obj).__name__}")
        return {}

    print(f"\nstate_dict: {len(sd)} tensors")
    total_params = 0
    print(f"  {'layer name':<55s} {'shape':<25s} {'mean':>10s} {'std':>10s} {'absmax':>10s}")
    for k, t in sd.items():
        if not torch.is_tensor(t):
            print(f"  {k:<55s} (non-tensor: {type(t).__name__})")
            continue
        n = t.numel()
        total_params += n
        if t.numel() > 0 and t.is_floating_point():
            print(f"  {k:<55s} {str(tuple(t.shape)):<25s} "
                  f"{t.mean().item():>10.5f} {t.std().item():>10.5f} {t.abs().max().item():>10.5f}")
        else:
            print(f"  {k:<55s} {str(tuple(t.shape)):<25s} (non-float)")
    print(f"\nTotal parameters: {total_params:,}")
    return sd


def diff_two(sd_a: dict, sd_b: dict, name_a: str = "A", name_b: str = "B"):
    """Compare two state_dicts: report missing/extra keys, per-shared-key distances."""
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    shared = keys_a & keys_b

    print(f"\n{'=' * 70}\nDiff: {name_a}  ↔  {name_b}\n{'=' * 70}")
    print(f"Keys only in {name_a}: {len(only_a)}")
    for k in sorted(only_a)[:20]:
        print(f"  - {k}")
    if len(only_a) > 20:
        print(f"  ... ({len(only_a) - 20} more)")
    print(f"\nKeys only in {name_b}: {len(only_b)}")
    for k in sorted(only_b)[:20]:
        print(f"  + {k}")
    if len(only_b) > 20:
        print(f"  ... ({len(only_b) - 20} more)")
    print(f"\nShared keys: {len(shared)}")

    if not shared:
        print("!! no shared keys; the two state_dicts use different naming "
              "conventions. The Triton rebrand renamed layers; you can "
              "remap names with a regex if you want to weight-load across.")
        return

    print(f"  {'layer name':<55s} {'shape':<22s} {'L2 diff':>12s} {'cos sim':>10s}")
    diffs = []
    for k in sorted(shared):
        a, b = sd_a[k], sd_b[k]
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            continue
        if a.shape != b.shape:
            print(f"  {k:<55s} shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        if not a.is_floating_point():
            continue
        delta = (a - b).norm().item()
        cos = torch.nn.functional.cosine_similarity(
            a.flatten().float().unsqueeze(0),
            b.flatten().float().unsqueeze(0),
        ).item()
        diffs.append((k, delta, cos))
        print(f"  {k:<55s} {str(tuple(a.shape)):<22s} {delta:>12.5e} {cos:>10.5f}")

    if diffs:
        avg_cos = sum(c for _, _, c in diffs) / len(diffs)
        max_delta = max(d for _, d, _ in diffs)
        print(f"\nAcross {len(diffs)} float-tensor layers: "
              f"mean cosine similarity = {avg_cos:.5f}, max L2 diff = {max_delta:.3e}")
        print("Interpretation:")
        print("  cosine ≈ 1.0  →  weights are essentially the same direction")
        print("  cosine ≈ 0.0  →  weights are unrelated (different init or different training)")
        print("  L2 diff ~0    →  weights match to numerical precision")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path, help="checkpoint to inspect")
    ap.add_argument("ckpt2", type=Path, nargs="?", default=None,
                    help="optional second checkpoint to compare against")
    ap.add_argument("--official", action="store_true",
                    help="add a note that this is the public WhaleVAD release ckpt")
    args = ap.parse_args()

    sd_a = summarise_one(args.ckpt, label="A" if args.ckpt2 else "ckpt")
    if args.official:
        print("\n(Note: this is the public Geldenhuys WhaleVAD release. "
              "State-dict keys use the official package's naming "
              "(`fbank`, `cnn_blocks.*`), not Triton's (`filterbank`, "
              "`feat_extractor`, `residual_stack`).)")

    if args.ckpt2 is not None:
        sd_b = summarise_one(args.ckpt2, label="B")
        diff_two(sd_a, sd_b, name_a=args.ckpt.name, name_b=args.ckpt2.name)


if __name__ == "__main__":
    main()
