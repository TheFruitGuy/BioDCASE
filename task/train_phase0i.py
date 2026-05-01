"""
Phase 0i: Add Focal Loss on Top of Weighted BCE
================================================

Phase 0h showed weighted BCE pushes rare-class F1 (BP 0.346 → 0.387)
but introduces high training instability — F1 swings of 0.19 between
epochs, occasional FP explosions to 50,000+ on a single class. The
hypothesis is that focal-loss modulation will dampen the over-confident
predictions causing those FP spikes, while preserving the rare-class
gains from weighting.

The DCASE tech report Section 2.6 specifies focal loss with α=0.25 and
γ=2 (the original Lin et al. recommendations). When combined with
weighted BCE, the per-frame loss for class c is::

    z_c    = logit for class c
    p_c    = sigmoid(z_c)
    bce_c  = -y_c log p_c - (1 - y_c) log (1 - p_c)
    weight = pos_weight[c] if y_c == 1 else 1
    p_t    = p_c if y_c == 1 else (1 - p_c)
    focal  = α * (1 - p_t)^γ
    loss_c = weight * focal * bce_c

We average across (frames × classes), masked to valid frames.

What this tests
---------------
1. Does focal modulation reduce FP explosions? Compare 0i's max
   epoch-to-epoch F1 swing against 0h's 0.19. Goal: < 0.10.
2. Does focal preserve rare-class F1 gains from weighted BCE?
   Compare bp and d best-F1 against 0h.
3. Is the headline F1 number better than 0g (plain BCE) and 0h
   (weighted BCE)?

Three possible outcomes
-----------------------
1. F1 stable, rare-class F1 holds → focal is the missing stabilizer.
   Move to Phase 0j (scale up to all 8 sites).
2. F1 stable but rare-class F1 drops below 0h → focal over-dampens
   gradient on hard examples; tune γ down or skip focal.
3. Still unstable → loss-function tweaks can't solve cross-site
   stability at this scale. Need data scaling or domain adaptation.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0i.py
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
import wandb_utils as wbu
from dataset import (
    load_annotations, get_file_manifest,
    build_positive_segments, build_negative_segments, build_val_segments,
    WhaleDataset, collate_fn,
)
from train_phase0 import (
    PHASE0_LR, PHASE0_WEIGHT_DECAY, PHASE0_BATCH_SIZE, PHASE0_THRESHOLD,
)
from train_phase0e import extend_all_segments, PHASE0E_SEGMENT_S
from train_phase0f import (
    PHASE0F_TRAIN_SITES, PHASE0F_VAL_SITES, PHASE0F_EPOCHS,
)
from train_phase0g import (
    build_phase0g_model, validate_3class,
)
from train_phase0h import compute_class_weights_for_sites


#: Focal-loss parameters per DCASE tech report Section 2.6.
PHASE0I_FOCAL_ALPHA = 0.25
PHASE0I_FOCAL_GAMMA = 2.0


# ======================================================================
# Loss
# ======================================================================

class FocalWeightedBCELoss(nn.Module):
    """
    Combined weighted BCE + focal modulation.

    Implements the DCASE 2025 tech report formulation: weighted BCE with
    per-class positive weights ``pos_weight``, multiplied by the focal
    modulation ``α * (1 - p_t)^γ`` from Lin et al. 2018.

    Parameters
    ----------
    pos_weight : torch.Tensor, shape ``(num_classes,)``
        Per-class positive weights. Multiplies the BCE loss only on
        positive frames (where target == 1).
    alpha : float
        Focal-loss balancing term. Default 0.25 per the paper.
    gamma : float
        Focal-loss focusing parameter. Default 2.0 per the paper.

    Notes
    -----
    Unlike ``nn.BCEWithLogitsLoss(reduction="none")``, this module
    applies the focal modulation per-element and returns a per-element
    loss tensor — so the caller is responsible for applying the
    sequence mask and averaging.
    """

    def __init__(self, pos_weight: torch.Tensor, alpha: float = 0.25,
                 gamma: float = 2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Return per-element loss tensor of shape ``logits.shape``.

        Caller is expected to apply any padding mask and reduce the
        result. Per-element output makes it easy to combine with the
        same masking logic used elsewhere in the training loops.
        """
        # Standard weighted BCE-with-logits, per element. Numerically
        # stable via F.binary_cross_entropy_with_logits.
        weight_per_class = self.pos_weight.view(1, 1, -1)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=weight_per_class, reduction="none",
        )
        # Focal modulation: (1 - p_t)^γ * α
        # p_t is the probability of the *true* class:
        #   p_t = p     when target == 1
        #   p_t = 1 - p when target == 0
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal = self.alpha * (1.0 - p_t).clamp(min=1e-7).pow(self.gamma)
        return focal * bce


# ======================================================================
# Training
# ======================================================================

def train_one_epoch_focal(model, spec_extractor, loader, criterion,
                          optimizer, device):
    """
    Training pass with the focal+weighted-BCE loss.

    Identical structure to ``train_phase0g.train_one_epoch_3class`` but
    the criterion produces per-element loss directly, so we don't have
    a separate masking step from the BCEWithLogitsLoss path.
    """
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

        valid = mask.unsqueeze(-1).float()
        per_elem = criterion(logits, targets) * valid
        loss = per_elem.sum() / (valid.sum() * targets.size(-1)).clamp(min=1.0)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses += loss.item()
        n += 1
    return losses / max(n, 1)


# ======================================================================
# Main
# ======================================================================

def main():
    """Run Phase 0i end-to-end."""
    assert cfg.USE_3CLASS, (
        "Phase 0i expects USE_3CLASS=True. Set it in config.py before running."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Wandb run setup. Seeding everything FIRST so DataLoader workers
    # and model weights are reproducible across runs of this phase.
    # Same seed as 0g/0h for fair comparison across the ladder.
    # ------------------------------------------------------------------
    SEED = 42
    wbu.seed_everything(SEED, deterministic=False)

    run = wbu.init_phase("0i", config={
        "lr": PHASE0_LR,
        "weight_decay": PHASE0_WEIGHT_DECAY,
        "batch_size": PHASE0_BATCH_SIZE,
        "threshold": PHASE0_THRESHOLD,
        "seed": SEED,
        "neg_ratio": cfg.NEG_RATIO,
        "segment_s": PHASE0E_SEGMENT_S,
        "epochs": PHASE0F_EPOCHS,
        "focal_alpha": PHASE0I_FOCAL_ALPHA,
        "focal_gamma": PHASE0I_FOCAL_GAMMA,
    })

    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0i_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0i configuration:")
    print(f"  Training sites: {PHASE0F_TRAIN_SITES}")
    print(f"  Validation sites: {PHASE0F_VAL_SITES}")
    print(f"  Output: 3-class (bmabz, d, bp)")
    print(f"  Loss: weighted BCE + focal (α={PHASE0I_FOCAL_ALPHA}, "
          f"γ={PHASE0I_FOCAL_GAMMA})")
    print(f"  Training segments: {PHASE0E_SEGMENT_S}s")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0F_EPOCHS}")

    # ------------------------------------------------------------------
    # Per-class weights — same logic as 0h, computed on training subset
    # ------------------------------------------------------------------
    pos_weight = compute_class_weights_for_sites(PHASE0F_TRAIN_SITES)
    print(f"\nClass weights (wc = N/Pc, normalized to min=1):")
    for name, w in zip(cfg.CALL_TYPES_3, pos_weight.tolist()):
        print(f"  {name}: {w:.3f}")

    # Stamp the computed weights onto the wandb run config.
    run.config.update({
        "pos_weight": pos_weight.tolist(),
        "pos_weight_per_class": dict(zip(cfg.CALL_TYPES_3, pos_weight.tolist())),
    })

    # ------------------------------------------------------------------
    # Training data
    # ------------------------------------------------------------------
    print(f"\nLoading training data...")
    train_manifest = get_file_manifest(PHASE0F_TRAIN_SITES)
    train_annotations = load_annotations(PHASE0F_TRAIN_SITES,
                                         manifest=train_manifest)
    print(f"  {len(train_manifest)} files, "
          f"{len(train_annotations)} annotations")
    for name in cfg.CALL_TYPES_3:
        c = (train_annotations["label_3class"] == name).sum()
        print(f"    {name}: {c} train annotations")

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
    # Validation
    # ------------------------------------------------------------------
    val_manifest = get_file_manifest(PHASE0F_VAL_SITES)
    val_annotations = load_annotations(PHASE0F_VAL_SITES,
                                       manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations)
    print(f"\n  Val: {len(val_manifest)} files, "
          f"{len(val_annotations)} annotations, "
          f"{len(val_segments)} segments")

    train_ds = WhaleDataset(train_segments)
    val_ds = WhaleDataset(val_segments)

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
    model, spec_extractor = build_phase0g_model(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = FocalWeightedBCELoss(
        pos_weight=pos_weight.to(device),
        alpha=PHASE0I_FOCAL_ALPHA,
        gamma=PHASE0I_FOCAL_GAMMA,
    ).to(device)
    # Plain BCE used only to compute val loss for logging — keeps the
    # val/loss number comparable across phases instead of mixing focal
    # and non-focal numbers.
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
    print(f"Training {PHASE0F_EPOCHS} epochs (focal + weighted BCE)")
    print(f"{'=' * 60}")

    best_f1 = 0.0
    for epoch in range(1, PHASE0F_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch_focal(
            model, spec_extractor, train_loader, criterion, optimizer, device,
        )
        val = validate_3class(
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
        }, run_dir / f"phase0i_epoch_{epoch:02d}.pt")
        if improved:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "f1": val["f1"], "history": history,
            }, run_dir / "phase0i_best.pt")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0i VERDICT")
    print(f"{'=' * 60}")

    f1s = [h["f1"] for h in history]
    bmabz_f1s = [h["per_class"]["bmabz"]["f1"] for h in history]
    d_f1s = [h["per_class"]["d"]["f1"] for h in history]
    bp_f1s = [h["per_class"]["bp"]["f1"] for h in history]

    print(f"\nOverall F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"\nBest F1 by class:")
    print(f"  bmabz: {max(bmabz_f1s):.3f}  (0g: 0.468, 0h: 0.451)")
    print(f"  d:     {max(d_f1s):.3f}  (0g: 0.098, 0h: 0.098)")
    print(f"  bp:    {max(bp_f1s):.3f}  (0g: 0.346, 0h: 0.387)")
    print(f"  Best overall: {max(f1s):.3f}  (0g: 0.385, 0h: 0.373)")

    second_half = f1s[len(f1s) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1])
              for i in range(1, len(second_half))]
    mean_swing = sum(swings) / max(len(swings), 1)
    max_swing = max(swings) if swings else 0.0
    print(f"\nSecond-half stability:")
    print(f"  Mean swing: {mean_swing:.3f}  (0g: 0.05, 0h: 0.07)")
    print(f"  Max swing:  {max_swing:.3f}  (0g: 0.29, 0h: 0.19)")

    if max_swing < 0.10 and max(f1s) > 0.37:
        verdict_text = (
            f"Focal stabilised training without sacrificing F1: "
            f"best {max(f1s):.3f}, max swing {max_swing:.3f}."
        )
        print("\n→ Focal stabilised training without sacrificing F1.")
        print("  Move to Phase 0j: scale up to 8 training sites.")
    elif max_swing < 0.15:
        verdict_text = (
            f"Improved stability vs 0h: max swing {max_swing:.3f}, "
            f"best F1 {max(f1s):.3f}."
        )
        print("\n→ Improved stability vs 0h. Mild gain or break-even on F1.")
        print("  Worth keeping focal in the recipe; scale up data next.")
    else:
        verdict_text = (
            f"Focal didn't fix instability: max swing {max_swing:.3f}."
        )
        print("\n→ Focal didn't fix instability. Loss-function tweaks aren't")
        print("  the answer at this scale; data / architecture next.")

    # ------------------------------------------------------------------
    # Wandb: stamp summary metrics + verdict and log best checkpoint
    # ------------------------------------------------------------------
    wbu.finalize_phase(
        history,
        verdict=verdict_text,
        best_ckpt=run_dir / "phase0i_best.pt",
    )


if __name__ == "__main__":
    main()
