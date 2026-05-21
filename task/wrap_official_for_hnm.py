"""
Wrap the Official WhaleVAD Checkpoint for 3-Class HNM
=====================================================

The official Geldenhuys checkpoint (``WhaleVAD_ATBFL_3P-c6f6a07a.pt``)
is incompatible with ``mine_hard_negatives.py`` / ``train_phase_hnm.py``
in two ways:

  1. Module keys are named for the original codebase (``fbank.*``,
     ``cnn_blocks.0.*``, ``cnn_blocks.1.blocks.0.*``,
     ``cnn_blocks.1.blocks.1.*``, plus a ``bb_proj.*`` head we don't
     use). We reuse ``remap_checkpoint_keys`` from
     ``continue_from_official.py`` to translate them.

  2. The classifier head outputs 7 fine-grained classes, but the HNM
     pipeline runs with ``cfg.USE_3CLASS=True`` and builds the model
     with a 3-class head. We collapse the 7×D classifier weight (and
     7-vector bias) into a 3×D head by **averaging the rows of each
     coarse-class group**::

        CALL_TYPES_7  =  [bma, bmb, bmz, bmd, bpd, bp20, bp20plus]
        CALL_TYPES_3  =  [bmabz, d, bp]

        bmabz  ←  mean of rows for (bma, bmb, bmz)   (indices 0,1,2)
        d      ←  mean of rows for (bmd, bpd)        (indices 3,4)
        bp     ←  mean of rows for (bp20, bp20plus)  (indices 5,6)

     This is NOT bit-identical to the inference-time 7→3 collapse, which
     uses ``max`` over sigmoid(logit_7) per group. Row-averaging is a
     linear approximation that gives ``logit_3 = mean(logit_7_in_group)``
     at init — a principled starting point that HNM fine-tuning will
     adjust away from anyway. Starting F1 will likely be a bit below
     the official's 0.443 because mean-of-logits is softer than
     max-of-sigmoids; threshold tuning at epoch 0 compensates partly.

The output is a regular train.py-style checkpoint bundle
(``model_state_dict`` + ``thresholds`` + ``best_f1``), so the standard
``build_model_for_ckpt`` + ``load_state_dict`` path works untouched.

Usage
-----
::

    python wrap_official_for_hnm.py \\
        --checkpoint WhaleVAD_ATBFL_3P-c6f6a07a.pt \\
        --output runs/official_3c_wrapped/best_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import config as cfg
from continue_from_official import remap_checkpoint_keys


def collapse_classifier_7_to_3(state_dict: dict) -> dict:
    """
    Replace ``classifier.weight`` (7×D) and ``classifier.bias`` (7,) with
    per-group means -> (3×D, 3) tensors. Group order matches
    ``cfg.CALL_TYPES_3``.
    """
    # Must match the maps in postprocess._SEVEN_TO_THREE.
    groups: dict[str, list[int]] = {
        "bmabz": [cfg.CALL_TYPES_7.index(x) for x in ("bma", "bmb", "bmz")],
        "d":     [cfg.CALL_TYPES_7.index(x) for x in ("bmd", "bpd")],
        "bp":    [cfg.CALL_TYPES_7.index(x) for x in ("bp20", "bp20plus")],
    }

    w7 = state_dict["classifier.weight"]   # [7, D]
    b7 = state_dict["classifier.bias"]     # [7]
    assert w7.shape[0] == 7 and b7.shape[0] == 7, (
        f"Expected 7-class classifier; got w={tuple(w7.shape)}, "
        f"b={tuple(b7.shape)}. Is this really the official checkpoint?"
    )

    rows_w = [w7[groups[c]].mean(dim=0) for c in cfg.CALL_TYPES_3]
    rows_b = [b7[groups[c]].mean() for c in cfg.CALL_TYPES_3]
    w3 = torch.stack(rows_w, dim=0).contiguous()   # [3, D]
    b3 = torch.stack(rows_b, dim=0).contiguous()   # [3]

    out = dict(state_dict)
    out["classifier.weight"] = w3
    out["classifier.bias"] = b3
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=str,
                   help="Path to WhaleVAD_ATBFL_3P-c6f6a07a.pt (the "
                        "official Geldenhuys checkpoint).")
    p.add_argument("--output", required=True, type=str,
                   help="Where to save the 3-class wrapped checkpoint. "
                        "Convention: runs/official_3c_wrapped/best_model.pt")
    return p.parse_args()


def main():
    args = parse_args()
    assert cfg.USE_3CLASS, (
        "This bridge expects cfg.USE_3CLASS=True. The whole point is to "
        "produce a 3-class checkpoint for the standard HNM pipeline."
    )

    print(f"Loading official checkpoint: {args.checkpoint}")
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"  raw keys: {len(raw)}")
    print(f"  classifier.weight shape: {tuple(raw['classifier.weight'].shape)}")

    # 1. Rename modules to match our model.py.
    remapped = remap_checkpoint_keys(raw)
    print(f"  after remap, keys: {len(remapped)} "
          f"(dropped bb_proj head + renamed fbank/cnn_blocks)")

    # 2. Average the classifier head down to 3 rows.
    collapsed = collapse_classifier_7_to_3(remapped)
    print(f"  classifier.weight after collapse: "
          f"{tuple(collapsed['classifier.weight'].shape)}")
    print(f"  classifier.bias after collapse:   "
          f"{tuple(collapsed['classifier.bias'].shape)}")

    # 3. Save as a train.py-style bundle.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": collapsed,
        # Neutral 0.5 thresholds; train_phase_hnm.py re-tunes them on
        # epoch 0 of HNM anyway (tune_thresholds=True at initial val).
        "thresholds": np.full(3, 0.5, dtype=np.float32),
        # Tag with the paper's reported number so wandb shows a clean
        # "starting F1" even though we'll re-measure on our val split.
        "best_f1": 0.443,
        "source_checkpoint": str(args.checkpoint),
        "note": (
            "Wrapped official 7-class Geldenhuys ckpt. Classifier head "
            "row-averaged 7->3 per cfg.CALL_TYPES_3 groups. Initial F1 "
            "may differ from 0.443 because mean-of-logits != "
            "max-of-sigmoids."
        ),
    }
    torch.save(payload, out_path)
    print(f"\nSaved 3-class wrapped checkpoint -> {out_path}")
    print("Next: mine_hard_negatives.py --checkpoints {} ...".format(out_path))


if __name__ == "__main__":
    main()
