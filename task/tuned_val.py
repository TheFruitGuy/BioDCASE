"""
Per-epoch tuned-threshold validation (phases 13r / 13s).

``train_final.validate`` scores the val set at a FIXED threshold (cfg.THRESHOLD
= 0.3). That fixed-0.3 number has repeatedly mis-ranked checkpoints in this
project (e.g. the wide-kernel winner tuned best at an epoch that was *not* the
fixed-0.3 peak), because the operating point that is optimal for one class at a
fixed threshold is not optimal for another.

``validate_tuned`` is a drop-in twin of ``validate``: it runs the *same* single
inference pass over the val loader (so there is no extra forward cost), then,
instead of thresholding at 0.3, it runs ``eval_conformer``'s per-class
coordinate-descent threshold tuner (parallelised over ``workers``) and scores at
the tuned operating point. It returns a dict with the SAME keys as ``validate``
( ``loss`` / ``f1`` / ``macro_paper`` / ``per_class`` ) plus ``thresholds`` so
the training loop's logging, selection and checkpoint code work unchanged.

The only added per-epoch cost over fixed-0.3 validation is the coordinate
descent itself, which is CPU-only (NumPy event matching) and embarrassingly
parallel -- hence the ``workers`` knob.
"""
from __future__ import annotations

import numpy as np
import torch

try:                                  # tqdm is optional (sandbox lacks it)
    from tqdm import tqdm
except Exception:                     # pragma: no cover
    def tqdm(it, **kw):
        return it


def _assemble_val_dict(metrics: dict, thresholds, loss: float,
                       class_names) -> dict:
    """Build the validate()-format dict from a compute_metrics() result.

    Pure / side-effect-free so it can be unit-tested with a synthetic metrics
    dict. ``metrics`` is keyed by class name (+ an ``overall`` micro entry),
    each value carrying f1/precision/recall/tp/fp/fn.
    """
    per_class = {
        name: {
            "f1":        metrics.get(name, {}).get("f1", 0.0),
            "precision": metrics.get(name, {}).get("precision", 0.0),
            "recall":    metrics.get(name, {}).get("recall", 0.0),
            "tp":        metrics.get(name, {}).get("tp", 0),
            "fp":        metrics.get(name, {}).get("fp", 0),
            "fn":        metrics.get(name, {}).get("fn", 0),
        }
        for name in class_names
    }
    p_bar = sum(per_class[n]["precision"] for n in class_names) / len(class_names)
    r_bar = sum(per_class[n]["recall"]    for n in class_names) / len(class_names)
    macro_paper = 2.0 * p_bar * r_bar / (p_bar + r_bar + 1e-8)
    return {
        "loss": float(loss),
        "f1": metrics.get("overall", {}).get("f1", 0.0),
        "macro_paper": macro_paper,
        "per_class": per_class,
        "thresholds": [float(t) for t in np.asarray(thresholds).ravel()],
    }


def validate_tuned(model, spec_extractor, loader, criterion, device,
                   val_annotations, file_start_dts, *,
                   workers: int = 1, start_thr=None, use_fp16: bool = False):
    """Single inference pass over ``loader`` + per-class threshold tuning.

    Mirrors train_final.validate's probability collection / loss / gt assembly
    exactly, then tunes thresholds instead of using cfg.THRESHOLD.
    """
    import config_final as cfg
    from postprocess_final import (
        Detection, collapse_probs_to_3class, compute_metrics,
        postprocess_predictions,
    )
    from eval_conformer import tune_thresholds_per_class, CLASS_NAMES

    autocast = (torch.autocast("cuda", dtype=torch.float16)
                if (use_fp16 and device.type == "cuda") else None)
    hop = spec_extractor.hop_length

    model.eval()
    losses, n = 0.0, 0
    all_probs = {}
    for audio, targets, mask, metas in tqdm(loader, desc="ValTuned", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            if autocast is not None:
                with autocast:
                    logits = model(spec_extractor(audio))
            else:
                logits = model(spec_extractor(audio))
        T = min(logits.size(1), targets.size(1))
        logits, targets, mask = logits[:, :T], targets[:, :T], mask[:, :T]
        valid = mask.unsqueeze(-1).float()
        per_frame = criterion(logits, targets) * valid
        loss = per_frame.sum() / (valid.sum() * targets.size(-1)).clamp(min=1.0)
        losses += loss.item()
        n += 1

        probs = torch.sigmoid(logits).float().cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    # 3-class collapse (no-op when the model already emits 3 channels, which is
    # the case for every conformer run; flip USE_3CLASS only around the
    # threshold ops in the 7->3 case, exactly like validate()).
    flipped = False
    if cfg.USE_3CLASS:
        all_probs_3 = all_probs
    else:
        all_probs_3 = collapse_probs_to_3class(all_probs)
        cfg.USE_3CLASS = True
        flipped = True

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

    try:
        thr = tune_thresholds_per_class(all_probs_3, gt_events,
                                        start=start_thr, workers=workers)
        metrics = compute_metrics(
            postprocess_predictions(all_probs_3, np.asarray(thr, dtype=np.float64)),
            gt_events, iou_threshold=0.3,
        )
    finally:
        if flipped:
            cfg.USE_3CLASS = False

    return _assemble_val_dict(metrics, thr, losses / max(n, 1), list(CLASS_NAMES))
