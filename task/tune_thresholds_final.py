"""
Per-Class Threshold Tuning (final pipeline)
===========================================

Standalone post-hoc tuner for any ``*_final``/BPN checkpoint. Forked from
``tune_thresholds.py``, which targets the older ``model.WhaleVAD`` pipeline -
the iterative search, per-class fine grids, and per-dataset breakdown are
identical. The only differences are:

- Imports the ``*_final`` modules (``config_final``, ``spectrogram_final``,
  ``model_bpn_final``, ``dataset_final``, ``postprocess_final``).
- Builds the model based on the checkpoint: ``WhaleVADBPN(use_bpn=True)`` if
  the state-dict contains BPN parameters, otherwise ``WhaleVAD`` (baseline) or
  ``WhaleVADBPN(use_bpn=False)`` for the dilated-backbone-only variant.
- Handles both output conventions: gated probabilities (BPN, already in (0,1))
  vs raw logits (baseline, requires sigmoid).
- Collapses 7-class output to 3 coarse classes before post-processing, as
  ``train_bpn3.validate()`` does.
- Optional ``--inspect-gate`` prints the learned BPN ROI weights and a
  histogram of mask values on the validation set, for diagnosing why a BPN
  checkpoint may not be helping.

Usage
-----
::

    python tune_thresholds_final.py --checkpoint runs/final_*/final_best.pt
    python tune_thresholds_final.py --checkpoint runs/bpn3_*/bpn3_epoch_21.pt --inspect-gate
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config_final as cfg
from spectrogram_final import SpectrogramExtractor
from dataset_final import (
    load_annotations, get_file_manifest, build_val_segments,
    WhaleDataset, collate_fn,
)
from postprocess_final import (
    postprocess_predictions, compute_metrics, Detection,
    collapse_probs_to_3class,
)


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Checkpoint to tune (final or BPN, auto-detected).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--inspect-gate", action="store_true",
                   help="For BPN checkpoints: print ROI weights and mask "
                        "value histogram on the validation set.")
    p.add_argument("--output", type=str, default=None,
                   help="Optional path to save a bundled (weights + tuned "
                        "thresholds) checkpoint.")
    return p.parse_args()


# ======================================================================
# Model construction (auto-detect from checkpoint)
# ======================================================================

def build_model_from_ckpt(ckpt: dict, device: torch.device):
    """Pick the right model class based on state-dict contents."""
    state = ckpt.get("model_state_dict", ckpt)
    keys = list(state.keys())
    has_bpn_branch = any(k.startswith(("bpn_heads.", "bpn_proposal",
                                       "bpn_lstm", "bpn_out",
                                       "roi_weights")) for k in keys)

    if has_bpn_branch:
        from model_bpn_final import WhaleVADBPN
        model = WhaleVADBPN(num_classes=7, use_bpn=True,
                            bpn_taps=(0, 1, 2), bpn_use_bilstm=True).to(device)
        kind = "WhaleVADBPN (use_bpn=True, gated probs)"
    else:
        # Baseline or BPN-backbone-only -- both return logits.
        try:
            from model_final import WhaleVAD
            model = WhaleVAD(num_classes=7).to(device)
            kind = "WhaleVAD (baseline, logits)"
        except ImportError:
            from model_bpn_final import WhaleVADBPN
            model = WhaleVADBPN(num_classes=7, use_bpn=False).to(device)
            kind = "WhaleVADBPN (use_bpn=False, logits)"

    spec = SpectrogramExtractor().to(device)
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec(dummy))                              # materialise lazy params

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected (showing first 3 of each)")
        for k in missing[:3]: print(f"    missing: {k}")
        for k in unexpected[:3]: print(f"    unexpected: {k}")
    model.eval()
    print(f"  Model: {kind}")
    return model, spec


# ======================================================================
# Inference (cache per-window probs and mask values if requested)
# ======================================================================

@torch.no_grad()
def collect_probs(model, spec, loader, device, *, collect_mask=False):
    """Run inference once; optionally also collect mask values for the BPN."""
    returns_probs = bool(getattr(model, "returns_probs", False))
    use_bpn = bool(getattr(model, "use_bpn", False))
    all_probs_7 = {}
    mask_samples = [] if (collect_mask and use_bpn) else None
    hop = spec.hop_length

    # If asked to collect masks, monkey-patch the BPN mask method to capture.
    if mask_samples is not None:
        orig_mask = model._bpn_mask
        def _capturing_mask(taps):
            m = orig_mask(taps)
            mask_samples.append(m.detach().cpu().numpy().ravel())
            return m
        model._bpn_mask = _capturing_mask

    try:
        for audio, _, _, metas in tqdm(loader, desc="Inference"):
            audio = audio.to(device)
            out = model(spec(audio))
            probs = out if returns_probs else torch.sigmoid(out)
            probs = probs.cpu().numpy()
            for j, meta in enumerate(metas):
                key = (meta["dataset"], meta["filename"], meta["start_sample"])
                n_samp = meta["end_sample"] - meta["start_sample"]
                n_frames = min(n_samp // hop, probs[j].shape[0])
                all_probs_7[key] = probs[j, :n_frames, :]
    finally:
        if mask_samples is not None:
            model._bpn_mask = orig_mask

    return all_probs_7, mask_samples


def build_gt_events(val_annotations, file_start_dts):
    gt_events = []
    for _, row in val_annotations.iterrows():
        fsd = file_start_dts.get((row["dataset"], row["filename"]))
        if fsd is None:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    return gt_events


# ======================================================================
# Gate diagnostics
# ======================================================================

def report_gate(model, mask_samples):
    """Print the learned ROI weights and a histogram of mask values."""
    print("\n=== BPN gate inspection ===")
    if hasattr(model, "roi_weights") and model.roi_weights is not None:
        w = torch.softmax(model.roi_weights.detach().cpu(), dim=0).numpy()
        n_roi = w.size
        H = len(model.bpn_taps)
        R = n_roi // H
        print(f"  ROI weights (softmax), H={H} heads x R={R} ROIs:")
        for h in range(H):
            row = w[h * R:(h + 1) * R]
            print(f"    head {h}: " + " ".join(f"{x:.3f}" for x in row)
                  + f"   (sum {row.sum():.3f})")
        spread = w.max() / max(w.min(), 1e-9)
        print(f"  weight max/min ratio: {spread:.2f}  "
              f"(1.0 = uniform; large = a few ROIs dominate)")
    if mask_samples:
        m = np.concatenate(mask_samples)
        edges = np.linspace(0, 1, 11)
        h, _ = np.histogram(m, bins=edges)
        total = h.sum()
        print(f"  Mask value histogram over {total} val frames:")
        for i in range(10):
            bar = "#" * int(40 * h[i] / max(total, 1))
            print(f"    [{edges[i]:.1f},{edges[i+1]:.1f}) {h[i]:>10}  {bar}")
        print(f"  mean={m.mean():.3f}  median={np.median(m):.3f}  "
              f"std={m.std():.3f}  min={m.min():.3f}  max={m.max():.3f}")


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, spec = build_model_from_ckpt(ckpt, device)

    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    val_annotations = load_annotations(cfg.VAL_DATASETS, manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations)
    val_loader = DataLoader(
        WhaleDataset(val_segments), batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }
    print(f"  Val: {len(val_manifest)} files, {len(val_segments)} segments")

    print("\nRunning inference...")
    all_probs_7, mask_samples = collect_probs(
        model, spec, val_loader, device, collect_mask=args.inspect_gate)
    all_probs_3 = collapse_probs_to_3class(all_probs_7)

    if args.inspect_gate:
        report_gate(model, mask_samples)

    # All post-processing operates in 3-class space.
    cfg.USE_3CLASS = True
    try:
        gt_events = build_gt_events(val_annotations, file_start_dts)

        # Per-class candidate grids (finer at the low end for d, bp).
        candidates_per_class = [
            np.arange(0.20, 0.90, 0.05),
            np.concatenate([np.arange(0.05, 0.50, 0.02),
                            np.arange(0.50, 0.90, 0.05)]),
            np.concatenate([np.arange(0.05, 0.50, 0.02),
                            np.arange(0.50, 0.90, 0.05)]),
        ]
        class_names = cfg.class_names()
        best_thresholds = np.array([0.5, 0.5, 0.5])

        print("\n=== Iterative threshold tuning ===")
        for it in range(3):
            print(f"\nPass {it + 1}/3")
            for c, cands in enumerate(candidates_per_class):
                best_f1, best_t = -1.0, best_thresholds[c]
                for t in cands:
                    trial = best_thresholds.copy()
                    trial[c] = t
                    preds = postprocess_predictions(all_probs_3, trial)
                    m = compute_metrics(preds, gt_events, iou_threshold=0.3)
                    f1 = m.get(class_names[c], {}).get("f1", 0.0)
                    if f1 > best_f1:
                        best_f1, best_t = f1, t
                best_thresholds[c] = best_t
                print(f"  {class_names[c]:6} threshold={best_t:.3f}  F1={best_f1:.3f}")

        print("\n=== Final tuned thresholds ===")
        for c, name in enumerate(class_names):
            print(f"  {name}: {best_thresholds[c]:.3f}")

        print("\n=== Final evaluation with tuned thresholds ===")
        preds = postprocess_predictions(all_probs_3, best_thresholds)
        metrics = compute_metrics(preds, gt_events, iou_threshold=0.3)
        per_class_f1 = []
        for cls in class_names:
            if cls in metrics:
                r = metrics[cls]
                print(f"  {cls:6} TP={r['tp']:5} FP={r['fp']:6} FN={r['fn']:6}  "
                      f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
                per_class_f1.append(r["f1"])
        macro = float(np.mean(per_class_f1)) if per_class_f1 else 0.0
        print(f"  MICRO F1={metrics['overall']['f1']:.3f}   MACRO F1={macro:.3f}")

        print("\n=== Per-dataset breakdown ===")
        for ds in cfg.VAL_DATASETS:
            ds_preds = [d for d in preds if d.dataset == ds]
            ds_gts = [d for d in gt_events if d.dataset == ds]
            m = compute_metrics(ds_preds, ds_gts, iou_threshold=0.3)
            print(f"\n  {ds}:")
            for cls in class_names:
                if cls in m:
                    r = m[cls]
                    print(f"    {cls:6} TP={r['tp']:5} FP={r['fp']:6} FN={r['fn']:6}  "
                          f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
            print(f"    MICRO F1={m['overall']['f1']:.3f}")
    finally:
        cfg.USE_3CLASS = False

    if args.output:
        torch.save({
            "model_state_dict": ckpt.get("model_state_dict", ckpt),
            "thresholds": torch.tensor(best_thresholds, dtype=torch.float32),
            "macro_f1": macro,
        }, args.output)
        print(f"\nSaved tuned checkpoint: {args.output}")


if __name__ == "__main__":
    main()
