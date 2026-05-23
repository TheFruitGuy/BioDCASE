"""
Apply the 30s Training Segment Patch to dataset.py
==================================================

The Phase 0 debugging revealed that training segments (median ~10s)
and validation tiles (uniformly 30s) had mismatched lengths, causing
the BiLSTM and BatchNorm to see different distributions at train vs
eval time. This is the most likely root cause of the F1 oscillation
and val-loss spikes seen in the original full-pipeline runs.

This script applies a minimal patch to ``dataset.py`` that extends
training segments to a fixed 30s while leaving everything else
untouched. Validation segments are already 30s, so they don't need
modification.

Two changes are made:

  1. ``build_dataloaders``: positive segments are extended to 30s
     immediately after ``build_positive_segments`` returns.

  2. ``TrainingDatasetWithResample.resample_negatives``: negative
     segments are extended to 30s after each random sample.

The patched file is written to ``dataset_30s.py`` next to the
original. Verify the diff, then either::

  mv dataset.py dataset_original.py
  mv dataset_30s.py dataset.py

…or apply the changes manually using the strings below.

Manual application
------------------

Open ``dataset.py`` and find the line in ``build_dataloaders``:

    pos_segs = build_positive_segments(train_annotations, train_manifest)

Add immediately after::

    from train_phase0e import extend_all_segments, PHASE0E_SEGMENT_S
    pos_segs = extend_all_segments(pos_segs, train_manifest, PHASE0E_SEGMENT_S)

Then find ``resample_negatives`` and replace its body with::

    def resample_negatives(self):
        from train_phase0e import extend_all_segments, PHASE0E_SEGMENT_S
        n_neg = int(len(self.positive_segments) * cfg.NEG_RATIO)
        self.negative_segments = build_negative_segments(
            self.annotations, self.manifest, n_segments=n_neg,
        )
        self.negative_segments = extend_all_segments(
            self.negative_segments, self.manifest, PHASE0E_SEGMENT_S,
        )
        self.segments = self.positive_segments + self.negative_segments

That's the entire change. Re-run train.py and see what happens.
"""

from pathlib import Path
import sys


# Two surgical edits keyed off unique lines in dataset.py.
# old_str must match exactly; new_str replaces it.

PATCH_1_OLD = """    pos_segs = build_positive_segments(train_annotations, train_manifest)
    train_ds = TrainingDatasetWithResample(pos_segs, train_manifest, train_annotations)"""

PATCH_1_NEW = """    pos_segs = build_positive_segments(train_annotations, train_manifest)
    # Phase 0 fix: extend training segments to match validation tile length.
    # Diagnosed via Phase 0e — the BiLSTM was trained on ~10s sequences
    # and validated on 30s tiles, producing wildly miscalibrated
    # confidence at eval. Forcing 30s training segments fixed the
    # oscillation in Phase 0f-0g (F1=0.385 stable on official val).
    from train_phase0e import extend_all_segments, PHASE0E_SEGMENT_S
    pos_segs = extend_all_segments(pos_segs, train_manifest, PHASE0E_SEGMENT_S)
    train_ds = TrainingDatasetWithResample(pos_segs, train_manifest, train_annotations)"""

PATCH_2_OLD = """    def resample_negatives(self):
        \"\"\"
        Draw a new set of negative segments.

        Called once per epoch (or less frequently; see ``RESAMPLE_EVERY``
        in ``train.py``). Updates ``self.segments`` so that subsequent
        ``__getitem__`` calls see the new negatives.
        \"\"\"
        n_neg = int(len(self.positive_segments) * cfg.NEG_RATIO)
        self.negative_segments = build_negative_segments(
            self.annotations, self.manifest, n_segments=n_neg,
        )
        self.segments = self.positive_segments + self.negative_segments"""

PATCH_2_NEW = """    def resample_negatives(self):
        \"\"\"
        Draw a new set of negative segments.

        Called once per epoch (or less frequently; see ``RESAMPLE_EVERY``
        in ``train.py``). Updates ``self.segments`` so that subsequent
        ``__getitem__`` calls see the new negatives.

        Phase 0 fix: each newly sampled negative is also extended to
        30s to match the validation tile length. Without this, the
        resampled negatives reintroduce the train/val length mismatch
        every epoch.
        \"\"\"
        from train_phase0e import extend_all_segments, PHASE0E_SEGMENT_S
        n_neg = int(len(self.positive_segments) * cfg.NEG_RATIO)
        self.negative_segments = build_negative_segments(
            self.annotations, self.manifest, n_segments=n_neg,
        )
        self.negative_segments = extend_all_segments(
            self.negative_segments, self.manifest, PHASE0E_SEGMENT_S,
        )
        self.segments = self.positive_segments + self.negative_segments"""


def main():
    """Apply both patches and write the result to dataset_30s.py."""
    src_path = Path("dataset.py")
    if not src_path.exists():
        print(f"ERROR: {src_path} not found in current directory.")
        print("Run this script from the project root (where dataset.py lives).")
        sys.exit(1)

    text = src_path.read_text()

    if PATCH_1_OLD not in text:
        print("ERROR: could not find patch 1 anchor in dataset.py.")
        print("       Maybe dataset.py has already been patched, or the")
        print("       file structure differs from what this script expects.")
        sys.exit(1)
    text = text.replace(PATCH_1_OLD, PATCH_1_NEW, 1)
    print("✓ Patch 1 applied (build_dataloaders extends positives to 30s)")

    if PATCH_2_OLD not in text:
        print("ERROR: could not find patch 2 anchor in dataset.py.")
        sys.exit(1)
    text = text.replace(PATCH_2_OLD, PATCH_2_NEW, 1)
    print("✓ Patch 2 applied (resample_negatives extends negatives to 30s)")

    out_path = Path("dataset_30s.py")
    out_path.write_text(text)
    print(f"\nPatched file written to {out_path}")
    print("\nTo activate:")
    print(f"  cp dataset.py dataset_original.py    # backup")
    print(f"  mv {out_path} dataset.py             # use the patched version")
    print("\nThen re-run your usual training command:")
    print("  CUDA_VISIBLE_DEVICES=<gpu> python train.py")


if __name__ == "__main__":
    main()
