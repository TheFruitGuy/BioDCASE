"""
apply_4class_patch.py
=====================

One-shot patcher that adds a 4-class D-split mode to the training pipeline.

The 3-class label "d" hides two species:
    bmd = blue whale D-call
    bpd = fin whale D-call (Balaenoptera physalus)

Both are present in train and val (per diagnose_d_subtypes.py output). This
patch lets you train with output classes [bmabz, bmd, bpd, bp] while keeping
evaluation in the 3-class space [bmabz, d, bp] — at validation time the
bmd and bpd channels are max-collapsed back into a single d channel before
threshold tuning and F1 computation. Same model, same recipe, four output
channels instead of three.

Usage
-----
    cd ~/BioDCASE/task
    python apply_4class_patch.py        # applies edits, writes .bak files
    python apply_4class_patch.py --check # verify all edits applied (no write)
    python apply_4class_patch.py --revert  # restore from .bak files

After patching, train with the new flag:
    python train.py --4class-d-split

To run a from-scratch 3-class baseline as before, just omit the flag.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


# ----------------------------------------------------------------------
# Edits — each is a (file, old_text, new_text) triple
# ----------------------------------------------------------------------

# Some edits use anchors that intentionally include surrounding code so we
# can be sure of where the insertion lands. The `old_text` must match
# exactly, character for character, or the patch fails.

EDITS = [
    # ==================================================================
    # config.py: add CALL_TYPES_4, COLLAPSE_MAP_4, USE_4CLASS_D_SPLIT
    # ==================================================================
    (
        "config.py",
        # OLD: ends with USE_3CLASS = True (just the flag, no trailing code)
        '''#: If True, train and evaluate on the 3-class task. If False, the model
#: outputs 7 fine-grained call-type logits which can be collapsed at
#: evaluation time via COLLAPSE_MAP.
#:
#: NOTE: The official Geldenhuys checkpoint (WhaleVAD_ATBFL_3P-c6f6a07a.pt)
#: outputs 7 classes despite the "3P" in its filename — the "3-class problem"
#: in the DCASE Table 2 ablation refers to *evaluating* in the 3-class space,
#: not to training a 3-class head. To match the official checkpoint exactly,
#: set this to False and use post-hoc collapse during evaluation.
USE_3CLASS = True''',
        '''#: If True, train and evaluate on the 3-class task. If False, the model
#: outputs 7 fine-grained call-type logits which can be collapsed at
#: evaluation time via COLLAPSE_MAP.
#:
#: NOTE: The official Geldenhuys checkpoint (WhaleVAD_ATBFL_3P-c6f6a07a.pt)
#: outputs 7 classes despite the "3P" in its filename — the "3-class problem"
#: in the DCASE Table 2 ablation refers to *evaluating* in the 3-class space,
#: not to training a 3-class head. To match the official checkpoint exactly,
#: set this to False and use post-hoc collapse during evaluation.
USE_3CLASS = True

#: 4-class label set with the D-class split into species (bmd, bpd).
#: Used when USE_4CLASS_D_SPLIT is True.
CALL_TYPES_4 = ["bmabz", "bmd", "bpd", "bp"]

#: Maps fine-grained labels to the 4-class space. Same mapping as
#: COLLAPSE_MAP except that bmd and bpd remain their own classes instead
#: of being collapsed into a single 'd'.
COLLAPSE_MAP_4 = {
    "bma": "bmabz", "bmb": "bmabz", "bmz": "bmabz",
    "bmd": "bmd",   "bpd": "bpd",
    "bp20": "bp",   "bp20plus": "bp",
}

#: If True, train with 4 output classes (the D-split set) and collapse the
#: bmd/bpd channels back into d at validation time for 3-class F1. Takes
#: precedence over USE_3CLASS when True. Motivation: per-subtype gradients
#: for the rare bpd class, which gets averaged into d in 3-class training.
#: See diagnose_d_subtypes.py for the supporting analysis.
USE_4CLASS_D_SPLIT = False''',
    ),

    # ==================================================================
    # config.py: update n_classes / class_names to handle 4-class mode
    # ==================================================================
    (
        "config.py",
        '''def n_classes() -> int:
    """Return the number of output classes the model should produce."""
    return 3 if USE_3CLASS else 7


def class_names() -> list[str]:
    """Return the ordered list of class label strings currently in use."""
    return list(CALL_TYPES_3) if USE_3CLASS else list(CALL_TYPES_7)''',
        '''def n_classes() -> int:
    """Return the number of output classes the model should produce."""
    if USE_4CLASS_D_SPLIT:
        return 4
    return 3 if USE_3CLASS else 7


def class_names() -> list[str]:
    """Return the ordered list of class label strings currently in use."""
    if USE_4CLASS_D_SPLIT:
        return list(CALL_TYPES_4)
    return list(CALL_TYPES_3) if USE_3CLASS else list(CALL_TYPES_7)''',
    ),

    # ==================================================================
    # dataset.py: add label_4class column in load_annotations
    # ==================================================================
    (
        "dataset.py",
        '''    ann = pd.concat(all_rows, ignore_index=True)
    # Add the collapsed 3-class label as a convenience column. Labels not
    # in COLLAPSE_MAP pass through unchanged (shouldn't happen in practice).
    ann["label_3class"] = ann["annotation"].map(cfg.COLLAPSE_MAP).fillna(ann["annotation"])
    return ann''',
        '''    ann = pd.concat(all_rows, ignore_index=True)
    # Add the collapsed 3-class label as a convenience column. Labels not
    # in COLLAPSE_MAP pass through unchanged (shouldn't happen in practice).
    ann["label_3class"] = ann["annotation"].map(cfg.COLLAPSE_MAP).fillna(ann["annotation"])
    # Same for 4-class (D-split) mapping. Always added regardless of mode
    # so the column is available whenever USE_4CLASS_D_SPLIT is flipped.
    ann["label_4class"] = ann["annotation"].map(cfg.COLLAPSE_MAP_4).fillna(ann["annotation"])
    return ann''',
    ),

    # ==================================================================
    # dataset.py: include label_4class in per-segment annotation dict
    # ==================================================================
    (
        "dataset.py",
        '''        out.setdefault(key, []).append({
            "start_s": (a["start_datetime"] - fsd).total_seconds(),
            "end_s": (a["end_datetime"] - fsd).total_seconds(),
            "label": a["annotation"],
            "label_3class": a["label_3class"],
        })''',
        '''        out.setdefault(key, []).append({
            "start_s": (a["start_datetime"] - fsd).total_seconds(),
            "end_s": (a["end_datetime"] - fsd).total_seconds(),
            "label": a["annotation"],
            "label_3class": a["label_3class"],
            "label_4class": a.get("label_4class", a["label_3class"]),
        })''',
    ),

    # ==================================================================
    # dataset.py: pick label_4class in WhaleDataset target painting
    # ==================================================================
    (
        "dataset.py",
        '''        for a in seg.annotations:
            label = a["label_3class"] if cfg.USE_3CLASS else a["label"]
            if label not in self.class_idx:
                continue''',
        '''        for a in seg.annotations:
            if cfg.USE_4CLASS_D_SPLIT:
                label = a["label_4class"]
            elif cfg.USE_3CLASS:
                label = a["label_3class"]
            else:
                label = a["label"]
            if label not in self.class_idx:
                continue''',
    ),

    # ==================================================================
    # postprocess.py: add _FOUR_TO_THREE mapping after _SEVEN_TO_THREE
    # ==================================================================
    (
        "postprocess.py",
        '''_SEVEN_TO_THREE = {
    "bmabz": [cfg.CALL_TYPES_7.index(x) for x in ("bma", "bmb", "bmz")],
    "d":     [cfg.CALL_TYPES_7.index(x) for x in ("bmd", "bpd")],
    "bp":    [cfg.CALL_TYPES_7.index(x) for x in ("bp20", "bp20plus")],
}''',
        '''_SEVEN_TO_THREE = {
    "bmabz": [cfg.CALL_TYPES_7.index(x) for x in ("bma", "bmb", "bmz")],
    "d":     [cfg.CALL_TYPES_7.index(x) for x in ("bmd", "bpd")],
    "bp":    [cfg.CALL_TYPES_7.index(x) for x in ("bp20", "bp20plus")],
}

# Used by the 4-class D-split mode (USE_4CLASS_D_SPLIT=True). Same shape
# as _SEVEN_TO_THREE — index lists per coarse class that are max-collapsed
# at evaluation time.
_FOUR_TO_THREE = {
    "bmabz": [cfg.CALL_TYPES_4.index("bmabz")],
    "d":     [cfg.CALL_TYPES_4.index(x) for x in ("bmd", "bpd")],
    "bp":    [cfg.CALL_TYPES_4.index("bp")],
}''',
    ),

    # ==================================================================
    # postprocess.py: generalize collapse_probs_to_3class for 4-class
    # ==================================================================
    (
        "postprocess.py",
        '''    if cfg.USE_3CLASS or not all_probs:
        return all_probs

    # Sample one entry to check actual array width — defensive against
    # being called accidentally on already-3-class arrays.
    sample = next(iter(all_probs.values()))
    if sample.shape[1] != 7:
        return all_probs

    out = {}
    for key, p7 in all_probs.items():
        p3 = np.zeros((p7.shape[0], 3), dtype=p7.dtype)
        for i, name in enumerate(cfg.CALL_TYPES_3):
            p3[:, i] = p7[:, _SEVEN_TO_THREE[name]].max(axis=1)
        out[key] = p3
    return out''',
        '''    if not all_probs:
        return all_probs
    # If we're already in 3-class output mode (and not running the
    # 4-class D-split experiment), there is nothing to collapse.
    if cfg.USE_3CLASS and not getattr(cfg, "USE_4CLASS_D_SPLIT", False):
        return all_probs

    # Sample one entry to check the actual array width and pick the
    # right index map. Defensive against being called on already-3-class
    # arrays (returns input unchanged in that case).
    sample = next(iter(all_probs.values()))
    if sample.shape[1] == 7:
        mapping = _SEVEN_TO_THREE
    elif sample.shape[1] == 4:
        mapping = _FOUR_TO_THREE
    else:
        return all_probs

    out = {}
    for key, p_in in all_probs.items():
        p3 = np.zeros((p_in.shape[0], 3), dtype=p_in.dtype)
        for i, name in enumerate(cfg.CALL_TYPES_3):
            p3[:, i] = p_in[:, mapping[name]].max(axis=1)
        out[key] = p3
    return out''',
    ),

    # ==================================================================
    # model.py: handle 4-class case in compute_class_weights
    # ==================================================================
    (
        "model.py",
        '''        # Map coarse name back to the set of fine-grained labels it contains,
        # so we can count the relevant annotations in the raw CSV.
        if cfg.USE_3CLASS:
            orig_labels = [k for k, v in cfg.COLLAPSE_MAP.items() if v == c_name]
            class_annots = annotations[annotations["annotation"].isin(orig_labels)]
        else:
            class_annots = annotations[annotations["annotation"] == c_name]''',
        '''        # Map coarse name back to the set of fine-grained labels it contains,
        # so we can count the relevant annotations in the raw CSV.
        if cfg.USE_4CLASS_D_SPLIT:
            orig_labels = [k for k, v in cfg.COLLAPSE_MAP_4.items() if v == c_name]
            class_annots = annotations[annotations["annotation"].isin(orig_labels)]
        elif cfg.USE_3CLASS:
            orig_labels = [k for k, v in cfg.COLLAPSE_MAP.items() if v == c_name]
            class_annots = annotations[annotations["annotation"].isin(orig_labels)]
        else:
            class_annots = annotations[annotations["annotation"] == c_name]''',
    ),

    # ==================================================================
    # train.py: add --4class-d-split CLI flag
    # ==================================================================
    (
        "train.py",
        '''    p.add_argument("--freeze_epochs", type=int, default=0,
                   help="If --pretrained is set, keep the encoder weights "
                        "frozen for this many initial epochs before "
                        "unfreezing them for end-to-end fine-tuning.")
    return p.parse_args()''',
        '''    p.add_argument("--freeze_epochs", type=int, default=0,
                   help="If --pretrained is set, keep the encoder weights "
                        "frozen for this many initial epochs before "
                        "unfreezing them for end-to-end fine-tuning.")
    p.add_argument("--4class-d-split", dest="four_class_d_split",
                   action="store_true",
                   help="Train with 4 output classes (bmabz, bmd, bpd, bp), "
                        "collapsing bmd+bpd back into d at validation "
                        "time. See diagnose_d_subtypes.py.")
    return p.parse_args()''',
    ),

    # ==================================================================
    # train.py: apply the CLI flag to cfg before any class-dependent code runs
    # ==================================================================
    (
        "train.py",
        '''def main():
    """End-to-end training driver."""
    args = parse_args()
    set_seed()''',
        '''def main():
    """End-to-end training driver."""
    args = parse_args()
    # Flip the 4-class D-split flag from CLI before any class-count-dependent
    # code (model construction, class weights, dataset target painting) runs.
    if getattr(args, "four_class_d_split", False):
        cfg.USE_4CLASS_D_SPLIT = True
        print("[train] USE_4CLASS_D_SPLIT enabled — 4 output classes "
              "(bmabz, bmd, bpd, bp), 3-class eval after bmd+bpd collapse")
    set_seed()''',
    ),

    # ==================================================================
    # train.py: log 4-class flag in wandb config so runs are distinguishable
    # ==================================================================
    (
        "train.py",
        '''    extra_tags = ["pretrained" if args.pretrained else "from_scratch"]
    if cfg.USE_WEIGHTED_BCE:
        extra_tags.append("weighted_bce")
    if getattr(cfg, "USE_FOCAL_LOSS", False):
        extra_tags.append("focal_loss")
    if not cfg.USE_WEIGHTED_BCE and not getattr(cfg, "USE_FOCAL_LOSS", False):
        extra_tags.append("plain_bce")''',
        '''    extra_tags = ["pretrained" if args.pretrained else "from_scratch"]
    if cfg.USE_WEIGHTED_BCE:
        extra_tags.append("weighted_bce")
    if getattr(cfg, "USE_FOCAL_LOSS", False):
        extra_tags.append("focal_loss")
    if not cfg.USE_WEIGHTED_BCE and not getattr(cfg, "USE_FOCAL_LOSS", False):
        extra_tags.append("plain_bce")
    if getattr(cfg, "USE_4CLASS_D_SPLIT", False):
        extra_tags.append("4class_d_split")''',
    ),
]


# ----------------------------------------------------------------------
# Patcher
# ----------------------------------------------------------------------

def apply_edits(check_only: bool = False) -> int:
    """Apply all edits. Returns 0 on success, non-zero on any failure."""
    # Group edits by file so each file is read once.
    by_file: dict[str, list[tuple[str, str]]] = {}
    for path, old, new in EDITS:
        by_file.setdefault(path, []).append((old, new))

    failures = 0
    for path, file_edits in by_file.items():
        p = Path(path)
        if not p.exists():
            print(f"FAIL  {path}: file not found")
            failures += 1
            continue

        original = p.read_text()
        text = original

        for i, (old, new) in enumerate(file_edits):
            if old in text:
                text = text.replace(old, new, 1)
                # Detect already-applied edits so re-running is safe.
                if new in original and old not in original:
                    pass  # impossible branch; keep for clarity
            elif new in text:
                # Already applied — skip silently
                pass
            else:
                print(f"FAIL  {path}: edit #{i+1} did not match. "
                      f"File may have been modified in a way that breaks "
                      f"the anchor. Showing first 100 chars of expected:")
                print(f"      {old[:100]!r}")
                failures += 1
                continue

        if check_only:
            print(f"check {path}: edits would apply cleanly")
            continue

        if text == original:
            print(f"skip  {path}: already patched")
            continue

        # Backup, then write.
        bak = p.with_suffix(p.suffix + ".bak_pre4class")
        if not bak.exists():
            shutil.copy2(p, bak)
        p.write_text(text)
        print(f"OK    {path}: patched (backup at {bak.name})")

    return failures


def revert() -> int:
    """Restore each .bak_pre4class file to its original location."""
    files = ["config.py", "dataset.py", "postprocess.py", "model.py", "train.py"]
    failures = 0
    for f in files:
        bak = Path(f + ".bak_pre4class")
        if not bak.exists():
            print(f"skip {f}: no backup")
            continue
        shutil.copy2(bak, f)
        bak.unlink()
        print(f"OK   {f}: restored from backup")
    return failures


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="Verify all edits would apply, but do not write.")
    p.add_argument("--revert", action="store_true",
                   help="Restore the original files from .bak_pre4class.")
    args = p.parse_args()

    if args.revert:
        sys.exit(revert())
    sys.exit(apply_edits(check_only=args.check))


if __name__ == "__main__":
    main()
