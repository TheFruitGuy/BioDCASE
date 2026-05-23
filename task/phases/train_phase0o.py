"""
Phase 0o: 7-Class Training + Weighted BCE + Focal Loss
=======================================================

The patched ``train.py`` produced F1=0.474 with weighted BCE alone
(``USE_FOCAL_LOSS=False``) plus 150 epochs and the per-epoch
threshold tuning machinery. Phase 0m produced F1=0.442 with plain BCE
on 30 epochs at the same data and architecture scale. The +0.032
gap conflates three changes — loss function, training duration, and
direct-3-class vs collapse-from-7. We need to separate them.

Phase 0o isolates one axis: **loss function only**. Same 30-epoch
schedule as Phase 0m, same 8 sites, same paper-config full LSTM,
same 7-class training with collapse-at-eval, same seed. The single
change is the loss:

  - Phase 0m uses ``nn.BCEWithLogitsLoss`` (plain BCE).
  - Phase 0o uses ``WhaleVADLoss`` with class weights from
    ``compute_class_weights`` and ``USE_FOCAL_LOSS=True`` (focal
    α=0.25 γ=2 as the paper specifies).

Why we expect this might help (or might not)
---------------------------------------------
Phase 0i tested focal loss at the *small* scale (4 sites, small LSTM)
and found it *hurt* by 0.029 F1. The argument was that focal loss is
designed to emphasize hard negatives, which only matters when there
are enough easy negatives to drown out the rare-class learning signal.
At 4-site small-LSTM scale, the model never had enough training
diversity to develop strong "easy" predictions in the first place, so
focal had nothing useful to do.

At 8-site full-LSTM scale (Phase 0m / 0o), the model is much stronger
on bmabz already (F1=0.531 in 0m). It has learned the easy patterns.
Focal loss's job is now to refocus the gradient on the rare/hard
classes (d at F1=0.149, bp at F1=0.400) without sacrificing the
already-learned bmabz patterns. This is the regime focal was designed
for.

Possible outcomes
-----------------
1. F1 climbs to ~0.46-0.48: focal helps at full scale. The patched
   ``train.py`` was right to use this loss, even though it ended up
   training without focal due to a config flag. Worth re-running
   ``train.py`` with focal enabled to push past 0.474.
2. F1 stays at ~0.44 (within noise): focal is neutral at this scale.
   Drop it from the recipe; weighted BCE alone is sufficient.
3. F1 drops below 0.43: focal hurts even at full scale. Confirms the
   Phase 0i finding generalizes. The gap from 0m to ``train.py``'s
   0.474 must come from elsewhere (epoch count, threshold tuning).

What changes vs Phase 0m
------------------------
Just the loss instantiation. Everything else — model build, dataset,
data loader, training loop, validation, wandb config — is reused
verbatim from Phase 0m via direct import.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0o.py
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import config as cfg
import wandb_utils as wbu
from dataset import (
    load_annotations, get_file_manifest,
    build_positive_segments, build_negative_segments, build_val_segments,
    WhaleDataset, collate_fn,
)
from model import WhaleVADLoss, compute_class_weights
from train_phase0 import (
    PHASE0_LR, PHASE0_WEIGHT_DECAY, PHASE0_BATCH_SIZE, PHASE0_THRESHOLD,
)
from train_phase0e import extend_all_segments, PHASE0E_SEGMENT_S
from train_phase0f import PHASE0F_VAL_SITES, PHASE0F_EPOCHS
from train_phase0m import (
    PHASE0M_TRAIN_SITES, build_phase0m_model, validate_7to3,
)


def train_one_epoch_whalevadloss(
    model, spec_extractor, loader, criterion, optimizer, device,
):
    """
    Training pass using WhaleVADLoss (which expects logits/targets/mask
    in a different signature than nn.BCEWithLogitsLoss).

    The plain-BCE training loop in Phase 0g/0m uses
    ``criterion(logits, targets)`` returning per-element BCE, then
    masks and averages manually. WhaleVADLoss does masking and
    averaging internally — it expects ``(logits, targets,
    padding_mask)`` and returns a scalar.

    Other than that, this is structurally identical to
    ``train_phase0g.train_one_epoch_3class``.
    """
    from tqdm import tqdm
    model.train()
    losses, n = 0.0, 0
    for audio, targets, mask, _ in tqdm(loader, desc="Train", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        spec = spec_extractor(audio)
        logits = model(spec)
        T = min(logits.size(1), targets.size(1))
        logits, targets, mask = logits[:, :T], targets[:, :T], mask[:, :T]

        # WhaleVADLoss expects boolean mask; .bool() handles both
        # uint8 and float tensors that may come from collate_fn.
        loss = criterion(logits, targets, padding_mask=mask.bool())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses += loss.item()
        n += 1
    return losses / max(n, 1)


def main():
    """Run Phase 0o end-to-end."""
    # 7-class mode + focal enabled. Both flags must be set BEFORE the
    # dataset is constructed (USE_3CLASS) and before WhaleVADLoss is
    # instantiated (USE_FOCAL_LOSS).
    cfg.USE_3CLASS = False
    cfg.USE_FOCAL_LOSS = True
    print(f"cfg.USE_3CLASS = {cfg.USE_3CLASS} (7-class training)")
    print(f"cfg.USE_FOCAL_LOSS = {cfg.USE_FOCAL_LOSS} "
          f"(focal α={cfg.FOCAL_ALPHA} γ={cfg.FOCAL_GAMMA})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Wandb run setup. Same seed as 0m for fair "loss-only" comparison.
    # ------------------------------------------------------------------
    SEED = 42
    wbu.seed_everything(SEED, deterministic=False)

    run = wbu.init_phase("0o", config={
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
        # Loss flags so wandb diff is unambiguous.
        "use_weighted_bce": True,
        "use_focal_loss":   True,
        "focal_alpha":      cfg.FOCAL_ALPHA,
        "focal_gamma":      cfg.FOCAL_GAMMA,
    })

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0o_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0o configuration:")
    print(f"  Training sites: {PHASE0M_TRAIN_SITES}  ({len(PHASE0M_TRAIN_SITES)})")
    print(f"  Validation sites: {PHASE0F_VAL_SITES}")
    print(f"  Output: 7-class training, collapse to 3-class at eval")
    print(f"  Loss: weighted BCE + focal (α=0.25 γ=2)")
    print(f"  LSTM: hidden={cfg.LSTM_HIDDEN}, layers={cfg.LSTM_LAYERS}")
    print(f"  Training segments: {PHASE0E_SEGMENT_S}s")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0F_EPOCHS}")

    # ------------------------------------------------------------------
    # Data — identical to Phase 0m
    # ------------------------------------------------------------------
    print(f"\nLoading training data...")
    train_manifest = get_file_manifest(PHASE0M_TRAIN_SITES)
    train_annotations = load_annotations(PHASE0M_TRAIN_SITES,
                                         manifest=train_manifest)
    print(f"  {len(train_manifest)} files, "
          f"{len(train_annotations)} annotations")

    pos_segs = build_positive_segments(train_annotations, train_manifest)
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)
    neg_segs = build_negative_segments(
        train_annotations, train_manifest, n_segments=n_neg,
    )
    pos_segs = extend_all_segments(pos_segs, train_manifest, PHASE0E_SEGMENT_S)
    neg_segs = extend_all_segments(neg_segs, train_manifest, PHASE0E_SEGMENT_S)
    train_segments = pos_segs + neg_segs
    print(f"  Training segments: {len(pos_segs)} pos + {len(neg_segs)} neg")

    val_manifest = get_file_manifest(PHASE0F_VAL_SITES)
    val_annotations = load_annotations(PHASE0F_VAL_SITES,
                                       manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations)
    print(f"\n  Val: {len(val_manifest)} files, "
          f"{len(val_annotations)} annotations, "
          f"{len(val_segments)} segments")

    train_ds = WhaleDataset(train_segments)
    val_ds = WhaleDataset(val_segments)

    sample_audio, sample_target, sample_mask, _ = train_ds[0]
    assert sample_target.shape[1] == 7, (
        f"Expected 7-channel targets, got {sample_target.shape}"
    )

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
    # Compute class weights — must happen with USE_3CLASS=False so we
    # get 7 weights (one per fine-grained class), not 3.
    # ------------------------------------------------------------------
    print(f"\nComputing per-class weights (paper formula w_c = N/P_c)...")
    pos_weight = compute_class_weights().to(device)
    print(f"  Pos weights tensor shape: {tuple(pos_weight.shape)}")
    assert pos_weight.numel() == 7, (
        f"Expected 7 class weights, got {pos_weight.numel()}"
    )

    # Stamp dataset-size + computed weights onto wandb config so they
    # show up in the run's config view alongside the toggles.
    run.config.update({
        "n_train_files":       len(train_manifest),
        "n_train_annotations": len(train_annotations),
        "n_train_segments":    len(train_segments),
        "pos_weight":          pos_weight.cpu().tolist(),
        "pos_weight_per_class": dict(zip(cfg.CALL_TYPES_7, pos_weight.cpu().tolist())),
    }, allow_val_change=True)

    # ------------------------------------------------------------------
    # Model + losses + optimizer
    # ------------------------------------------------------------------
    model, spec_extractor = build_phase0m_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    run.config.update({"n_params": n_params}, allow_val_change=True)

    # Training loss: weighted BCE + focal. Validation loss: plain BCE
    # for comparability with Phase 0m — WhaleVADLoss's weighting is a
    # training-time device that would make val curves look different
    # from 0m on identical predictions if used at eval time.
    train_criterion = WhaleVADLoss(pos_weight=pos_weight).to(device)
    val_criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)

    optimizer = AdamW(
        model.parameters(), lr=PHASE0_LR, weight_decay=PHASE0_WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = []
    print(f"\n{'=' * 60}")
    print(f"Training {PHASE0F_EPOCHS} epochs (7→3, weighted BCE + focal)")
    print(f"{'=' * 60}")

    best_f1 = 0.0
    for epoch in range(1, PHASE0F_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch_whalevadloss(
            model, spec_extractor, train_loader, train_criterion,
            optimizer, device,
        )
        val = validate_7to3(
            model, spec_extractor, val_loader, val_criterion, device,
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
        }, run_dir / f"phase0o_epoch_{epoch:02d}.pt")
        if improved:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "f1": val["f1"], "history": history,
            }, run_dir / "phase0o_best.pt")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0o VERDICT")
    print(f"{'=' * 60}")

    f1s = [h["f1"] for h in history]
    bmabz_f1s = [h["per_class"]["bmabz"]["f1"] for h in history]
    d_f1s = [h["per_class"]["d"]["f1"] for h in history]
    bp_f1s = [h["per_class"]["bp"]["f1"] for h in history]

    print(f"\nOverall F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"\nBest F1 by class:")
    print(f"  bmabz: {max(bmabz_f1s):.3f}  (0m plain BCE: 0.531)")
    print(f"  d:     {max(d_f1s):.3f}  (0m plain BCE: 0.149)")
    print(f"  bp:    {max(bp_f1s):.3f}  (0m plain BCE: 0.400)")
    print(f"  Best overall: {max(f1s):.3f}  (0m: 0.442, paper: 0.443)")

    second_half = f1s[len(f1s) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1])
              for i in range(1, len(second_half))]
    mean_swing = sum(swings) / max(len(swings), 1)
    max_swing = max(swings) if swings else 0.0
    print(f"\nSecond-half stability:")
    print(f"  Mean swing: {mean_swing:.3f}  (0m: 0.042)")
    print(f"  Max swing:  {max_swing:.3f}  (0m: 0.130)")

    delta = max(f1s) - 0.442
    if delta > 0.015:
        verdict_text = (
            f"Focal+weighted helped: best F1 {max(f1s):.3f} "
            f"(+{delta:.3f} vs 0m). Re-run train.py with focal on."
        )
        print(f"\n→ Focal+weighted helped (+{delta:.3f} F1 vs 0m).")
        print("  At full scale focal does the job it was designed for.")
        print("  Consider re-running train.py with USE_FOCAL_LOSS=True")
        print("  to push past F1=0.474.")
    elif delta > -0.015:
        verdict_text = (
            f"Focal+weighted neutral: best F1 {max(f1s):.3f} "
            f"({delta:+.3f} vs 0m). Stick with plain BCE."
        )
        print(f"\n→ Roughly neutral ({delta:+.3f} F1).")
        print("  Focal+weighted is a wash at this scale. Stick with")
        print("  plain BCE for simplicity; weighted alone is enough.")
    else:
        verdict_text = (
            f"Focal+weighted hurt: best F1 {max(f1s):.3f} "
            f"({delta:+.3f} vs 0m). Drop from recipe."
        )
        print(f"\n→ Focal+weighted hurt ({delta:+.3f} F1).")
        print("  Confirms Phase 0i finding: focal hurts at our scale.")
        print("  Drop it from the recipe permanently.")

    # ------------------------------------------------------------------
    # Wandb: stamp summary metrics + verdict and log best checkpoint
    # ------------------------------------------------------------------
    wbu.finalize_phase(
        history,
        verdict=verdict_text,
        best_ckpt=run_dir / "phase0o_best.pt",
    )


if __name__ == "__main__":
    main()
