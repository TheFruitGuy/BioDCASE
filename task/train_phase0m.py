"""
Phase 0m: 7-Class Training with Collapse-at-Eval
=================================================

The paper's actual recipe uses 7 fine-grained call types during
training (bma, bmb, bmz, bmd, bpd, bp20, bp20plus), which are
collapsed to 3 coarse classes (bmabz, d, bp) only at evaluation
time. We have been training directly on 3 classes throughout
Phase 0g–0l for simplicity. Phase 0m closes that final gap to the
paper's setup.

The argument for 7-class training is that the model can learn to
discriminate between e.g. bma and bmb internally, even though both
collapse to bmabz at eval. Each subclass has its own annotation
boundaries and acoustic signature, so 7-class training provides a
richer learning signal than 3-class.

The argument against is that some of the 7 classes are very rare
(bp20 has only ~2k annotations across 8 training sites, vs bma's
~10k). With plain BCE, rare-class learning is slow.

What changes vs Phase 0k
------------------------
- Model output: 7 channels instead of 3
- Training targets: 7-class (model.dataset reads
  ``cfg.USE_3CLASS=False`` and generates 7-channel targets)
- LSTM: paper config (hidden=128, layers=2), the 0l intervention
- Loss: still plain BCE — we don't reintroduce weighting/focal
  because Phase 0g–0i showed weighting hurt overall F1
- Validation: probabilities are collapsed from 7 channels to 3 via
  max-over-subclasses (existing ``collapse_probs_to_3class`` logic),
  then evaluated against the 3-class GT (same val protocol as 0g)

This is the convergence node of the ladder: 8-site data (0k) +
paper LSTM (0l) + 7-class training. If F1 lands above the paper's
0.443, we've reproduced (and slightly beaten) Geldenhuys end-to-end.

Three possible outcomes
-----------------------
1. F1 climbs above 0.42 with smoother per-class learning — the paper
   recipe was right and finer training helps. This becomes our best
   single-run number.
2. F1 hits 0.40-0.42 (similar to 0k/0l) with rare 7-classes never
   learning meaningfully — the collapse-to-3 absorbs the rare-class
   weakness, so it's a wash.
3. F1 drops below 0.38 — dividing the gradient signal across 7
   classes dilutes per-class learning at this dataset scale.

Compute
-------
8 sites × full LSTM × 7-class output. Training time similar to
0k — the extra output channels add negligible cost. Expect
~16 min/epoch × 30 epochs ≈ 8 hours.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0m.py
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import config as cfg
import wandb_utils as wbu
from spectrogram import SpectrogramExtractor
from model import WhaleVAD
from dataset import (
    load_annotations, get_file_manifest,
    build_positive_segments, build_negative_segments, build_val_segments,
    WhaleDataset, collate_fn,
)
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
    collapse_probs_to_3class,
)
from train_phase0 import (
    PHASE0_LR, PHASE0_WEIGHT_DECAY, PHASE0_BATCH_SIZE, PHASE0_THRESHOLD,
)
from train_phase0e import extend_all_segments, PHASE0E_SEGMENT_S
from train_phase0f import PHASE0F_VAL_SITES, PHASE0F_EPOCHS
from train_phase0g import train_one_epoch_3class


#: Use the full 8-site training set — Phase 0k showed scaling helps,
#: and the paper trained on all 8 sites.
PHASE0M_TRAIN_SITES = list(cfg.TRAIN_DATASETS)

#: Use the paper-config LSTM. Phase 0l showed it wins over the small
#: variant; combining with 8-site data (like 0k) is the natural
#: companion change.
PHASE0M_LSTM_HIDDEN = 128
PHASE0M_LSTM_LAYERS = 2


def build_phase0m_model(device: torch.device):
    """
    Build a 7-class WhaleVAD with the paper-config LSTM.

    Sanity-check that cfg has been flipped to 7-class mode before the
    dataset is built — otherwise WhaleDataset would still produce
    3-channel targets and the loss would shape-mismatch.
    """
    assert not cfg.USE_3CLASS, (
        "Phase 0m requires cfg.USE_3CLASS=False. Flip it before constructing "
        "the dataset so target tensors come out 7-wide."
    )
    assert cfg.LSTM_HIDDEN >= PHASE0M_LSTM_HIDDEN, (
        f"cfg.LSTM_HIDDEN={cfg.LSTM_HIDDEN}; expected >={PHASE0M_LSTM_HIDDEN}"
    )
    assert cfg.LSTM_LAYERS >= PHASE0M_LSTM_LAYERS, (
        f"cfg.LSTM_LAYERS={cfg.LSTM_LAYERS}; expected >={PHASE0M_LSTM_LAYERS}"
    )
    model = WhaleVAD(num_classes=7).to(device)
    spec = SpectrogramExtractor().to(device)
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec(dummy))
    return model, spec


@torch.no_grad()
def validate_7to3(model, spec_extractor, loader, criterion, device,
                  val_annotations, file_start_dts, threshold: float):
    """
    Run validation: 7-class inference, collapse to 3-class, score F1.

    Mirrors ``train_phase0g.validate_3class`` but adds the collapse
    step before constructing predictions. The 3-class GT is unchanged
    (it comes from the ``label_3class`` column in annotations, which
    already maps every fine-grained label to its coarse class).
    """
    from tqdm import tqdm
    model.eval()
    losses, n = 0.0, 0
    all_probs_7 = {}
    hop = spec_extractor.hop_length

    for audio, targets, mask, metas in tqdm(loader, desc="Val", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        logits = model(spec_extractor(audio))
        T = min(logits.size(1), targets.size(1))
        logits, targets, mask = logits[:, :T], targets[:, :T], mask[:, :T]

        valid = mask.unsqueeze(-1).float()
        per_frame = criterion(logits, targets) * valid
        loss = per_frame.sum() / (valid.sum() * targets.size(-1)).clamp(min=1.0)
        losses += loss.item()
        n += 1

        probs7 = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs7[j].shape[0])
            all_probs_7[key] = probs7[j, :n_frames, :]

    # Collapse 7→3 via max-over-subclasses. cfg.USE_3CLASS is False
    # at this point so the collapse activates; we don't toggle the
    # flag back (that would only matter if downstream code
    # re-checked it, and we don't).
    all_probs_3 = collapse_probs_to_3class(all_probs_7)

    # Postprocessing operates on the 3-class probabilities, but
    # cfg.class_names() and cfg.n_classes() return 7 right now —
    # postprocess_predictions reads ``cfg.CALL_TYPES_3`` directly
    # for label naming, so this works regardless of the flag.
    thresholds = np.array([threshold] * 3)
    pred_events = postprocess_predictions(all_probs_3, thresholds)

    # 3-class GT (built from label_3class, unchanged regardless of
    # USE_3CLASS flag).
    gt_events = []
    for _, row in val_annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    metrics = compute_metrics(pred_events, gt_events, iou_threshold=0.3)
    overall_f1 = metrics.get("overall", {}).get("f1", 0.0)
    per_class = {}
    for name in cfg.CALL_TYPES_3:
        m = metrics.get(name, {})
        per_class[name] = {
            "f1": m.get("f1", 0.0),
            "precision": m.get("precision", 0.0),
            "recall": m.get("recall", 0.0),
            "tp": m.get("tp", 0),
            "fp": m.get("fp", 0),
            "fn": m.get("fn", 0),
        }
    return {
        "loss": losses / max(n, 1),
        "f1": overall_f1,
        "per_class": per_class,
    }


def main():
    """Run Phase 0m end-to-end."""
    # --- CRITICAL: flip the global flag BEFORE building any dataset ---
    # WhaleDataset reads cfg.USE_3CLASS at __init__ to decide target
    # tensor width. We need 7-wide targets for training, so this has
    # to happen before any WhaleDataset is instantiated.
    cfg.USE_3CLASS = False
    print(f"cfg.USE_3CLASS forced to {cfg.USE_3CLASS} for 7-class training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Wandb run setup. Seeding everything FIRST so DataLoader workers
    # and model weights are reproducible across runs of this phase.
    # Same seed as 0g/0k/0l for fair comparison across the ladder.
    # ------------------------------------------------------------------
    SEED = 42
    wbu.seed_everything(SEED, deterministic=False)

    run = wbu.init_phase("0m", config={
        "lr": PHASE0_LR,
        "weight_decay": PHASE0_WEIGHT_DECAY,
        "batch_size": PHASE0_BATCH_SIZE,
        "threshold": PHASE0_THRESHOLD,
        "seed": SEED,
        "neg_ratio": cfg.NEG_RATIO,
        "segment_s": PHASE0E_SEGMENT_S,
        "epochs": PHASE0F_EPOCHS,
        "train_sites": PHASE0M_TRAIN_SITES,
        "val_sites": PHASE0F_VAL_SITES,
        "lstm_hidden": cfg.LSTM_HIDDEN,
        "lstm_layers": cfg.LSTM_LAYERS,
        "training_classes": 7,
        "eval_classes": 3,
    })

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0m_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0m configuration:")
    print(f"  Training sites: {PHASE0M_TRAIN_SITES}  ({len(PHASE0M_TRAIN_SITES)})")
    print(f"  Validation sites: {PHASE0F_VAL_SITES}")
    print(f"  Output: 7-class training, collapse to 3-class at eval")
    print(f"  Loss: plain BCE (no weighting, no focal)")
    print(f"  LSTM: hidden={cfg.LSTM_HIDDEN}, layers={cfg.LSTM_LAYERS} "
          f"(paper config)")
    print(f"  Training segments: {PHASE0E_SEGMENT_S}s")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0F_EPOCHS}")

    # ------------------------------------------------------------------
    # Training data — full 8 sites
    # ------------------------------------------------------------------
    print(f"\nLoading training data...")
    train_manifest = get_file_manifest(PHASE0M_TRAIN_SITES)
    train_annotations = load_annotations(PHASE0M_TRAIN_SITES,
                                         manifest=train_manifest)
    print(f"  {len(train_manifest)} files, "
          f"{len(train_annotations)} annotations")

    # Per-fine-class breakdown — useful for spotting which classes
    # have so few examples that they may never learn.
    print(f"  Fine-class (7) breakdown of training annotations:")
    fine_counts = {}
    for name in cfg.CALL_TYPES_7:
        c = int((train_annotations["annotation"] == name).sum())
        fine_counts[name] = c
        print(f"    {name}: {c}")
    print(f"  Coarse-class (3) breakdown (for eval reference):")
    coarse_counts = {}
    for name in cfg.CALL_TYPES_3:
        c = int((train_annotations["label_3class"] == name).sum())
        coarse_counts[name] = c
        print(f"    {name}: {c}")

    # Stamp the class breakdowns onto the wandb config so the prof can
    # read off the per-class training-data sizes without leaving the UI.
    run.config.update({
        "train_counts_7class": fine_counts,
        "train_counts_3class": coarse_counts,
        "n_train_files":       len(train_manifest),
        "n_train_annotations": len(train_annotations),
    }, allow_val_change=True)

    pos_segs = build_positive_segments(train_annotations, train_manifest)
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)
    neg_segs = build_negative_segments(
        train_annotations, train_manifest, n_segments=n_neg,
    )
    pos_segs = extend_all_segments(pos_segs, train_manifest, PHASE0E_SEGMENT_S)
    neg_segs = extend_all_segments(neg_segs, train_manifest, PHASE0E_SEGMENT_S)
    train_segments = pos_segs + neg_segs
    print(f"  Training segments: {len(pos_segs)} pos + {len(neg_segs)} neg")

    # ------------------------------------------------------------------
    # Validation — official BioDCASE split
    # ------------------------------------------------------------------
    val_manifest = get_file_manifest(PHASE0F_VAL_SITES)
    val_annotations = load_annotations(PHASE0F_VAL_SITES,
                                       manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations)
    print(f"\n  Val: {len(val_manifest)} files, "
          f"{len(val_annotations)} annotations, "
          f"{len(val_segments)} segments")

    # WhaleDataset with USE_3CLASS=False → 7-channel targets in batches.
    train_ds = WhaleDataset(train_segments)
    val_ds = WhaleDataset(val_segments)

    # Confirm target width — quick sanity check before consuming hours.
    sample_audio, sample_target, sample_mask, _ = train_ds[0]
    assert sample_target.shape[1] == 7, (
        f"Expected 7-channel targets, got shape {sample_target.shape}. "
        f"WhaleDataset is still using 3-class mode."
    )
    print(f"  Sanity: target shape per sample = {sample_target.shape}")

    train_loader = DataLoader(
        train_ds, batch_size=PHASE0_BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        **wbu.seeded_dataloader_kwargs(SEED),
    )
    val_loader = DataLoader(
        val_ds, batch_size=PHASE0_BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }

    # ------------------------------------------------------------------
    # Model + loss + optimizer
    # ------------------------------------------------------------------
    model, spec_extractor = build_phase0m_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    run.config.update({"n_params": n_params}, allow_val_change=True)

    criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)
    optimizer = AdamW(
        model.parameters(), lr=PHASE0_LR, weight_decay=PHASE0_WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )

    # ------------------------------------------------------------------
    # Training loop. The Phase 0g training helper happens to be class-
    # count-agnostic — it averages BCE across whatever number of
    # channels the targets/logits have — so we reuse it directly.
    # ------------------------------------------------------------------
    history = []
    print(f"\n{'=' * 60}")
    print(f"Training {PHASE0F_EPOCHS} epochs (7-class train, 3-class eval)")
    print(f"{'=' * 60}")

    best_f1 = 0.0
    for epoch in range(1, PHASE0F_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch_3class(
            model, spec_extractor, train_loader, criterion, optimizer, device,
        )
        val = validate_7to3(
            model, spec_extractor, val_loader, criterion, device,
            val_annotations, file_start_dts, threshold=PHASE0_THRESHOLD,
        )
        epoch_time = time.time() - t0

        improved = val["f1"] > best_f1
        if improved:
            best_f1 = val["f1"]
        marker = " *** new best" if improved else ""

        print(f"\nEpoch {epoch:2d}/{PHASE0F_EPOCHS}  ({epoch_time:.0f}s){marker}")
        print(f"  Train loss: {train_loss:.4f}   Val loss: {val['loss']:.4f}")
        for name in cfg.CALL_TYPES_3:
            pc = val["per_class"][name]
            print(f"    {name.upper():6} TP={pc['tp']:5} FP={pc['fp']:6} "
                  f"FN={pc['fn']:5}  P={pc['precision']:.3f} "
                  f"R={pc['recall']:.3f} F1={pc['f1']:.3f}")
        print(f"    OVERALL F1={val['f1']:.3f}")

        wbu.log_epoch_3class(epoch, train_loss, val)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "f1": val["f1"],
            "per_class": val["per_class"],
        })

        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "f1": val["f1"], "history": history,
        }, run_dir / f"phase0m_epoch_{epoch:02d}.pt")
        if improved:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "f1": val["f1"], "history": history,
            }, run_dir / "phase0m_best.pt")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0m VERDICT")
    print(f"{'=' * 60}")

    f1s = [h["f1"] for h in history]
    bmabz_f1s = [h["per_class"]["bmabz"]["f1"] for h in history]
    d_f1s = [h["per_class"]["d"]["f1"] for h in history]
    bp_f1s = [h["per_class"]["bp"]["f1"] for h in history]

    print(f"\nOverall F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"\nBest F1 by class:")
    print(f"  bmabz: {max(bmabz_f1s):.3f}  (0k 8-site 3-class: 0.502)")
    print(f"  d:     {max(d_f1s):.3f}  (0k 8-site 3-class: 0.113)")
    print(f"  bp:    {max(bp_f1s):.3f}  (0k 8-site 3-class: 0.390)")
    print(f"  Best overall: {max(f1s):.3f}  "
          f"(0k: 0.418, 0l: 0.417, paper: 0.443)")

    second_half = f1s[len(f1s) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1])
              for i in range(1, len(second_half))]
    mean_swing = sum(swings) / max(len(swings), 1)
    max_swing = max(swings) if swings else 0.0
    print(f"\nSecond-half stability:")
    print(f"  Mean swing: {mean_swing:.3f}")
    print(f"  Max swing:  {max_swing:.3f}")

    if max(f1s) > 0.43:
        verdict_text = (
            f"7-class recipe matched/beat the paper: best F1 {max(f1s):.3f} "
            f"(paper: 0.443). Reproduction successful."
        )
        print("\n→ 7-class training matched/beat the paper. Reproduction")
        print("  successful. Add threshold tuning (Phase 0j) on this")
        print("  checkpoint for any final marginal gain.")
    elif max(f1s) > 0.40:
        verdict_text = (
            f"7-class comparable to 3-class baselines: best F1 {max(f1s):.3f} "
            f"(0k/0l: ~0.417). Fine-grained training is a wash."
        )
        print("\n→ 7-class comparable to 0k/0l. Fine-grained training is")
        print("  a wash at this scale; collapse-at-eval absorbs the rare-")
        print("  class weakness. Keep whichever training paradigm you")
        print("  prefer; both work.")
    else:
        verdict_text = (
            f"7-class underperformed: best F1 {max(f1s):.3f}. "
            f"Gradient signal diluted across rare classes."
        )
        print("\n→ 7-class training underperformed. Splitting gradient")
        print("  signal across 7 classes diluted per-class learning.")
        print("  Stick with direct 3-class training (0k/0l).")

    # ------------------------------------------------------------------
    # Wandb: stamp summary metrics + verdict and log best checkpoint
    # ------------------------------------------------------------------
    wbu.finalize_phase(
        history,
        verdict=verdict_text,
        best_ckpt=run_dir / "phase0m_best.pt",
    )


if __name__ == "__main__":
    main()
