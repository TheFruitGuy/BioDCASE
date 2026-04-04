#!/bin/bash
# ===========================================================================
# Whale-Conformer: Full workflow from raw data to submission
# ===========================================================================
# 
# STEP 0: Environment setup
# STEP 1: Download & organise the ATBFL dataset
# STEP 2: Train the model
# STEP 3: Tune thresholds on validation set
# STEP 4: Run inference on evaluation set
# STEP 5: Package submission
#
# Run individual steps or the whole thing:
#   bash setup_and_run.sh          # everything
#   bash setup_and_run.sh step2    # just training
# ===========================================================================

set -e

STEP=${1:-"all"}
DATA_ROOT="./data"
RUN_DIR="./runs"

# -------------------------------------------------------
# STEP 0: Install dependencies
# -------------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "step0" ]]; then
    echo "=========================================="
    echo "STEP 0: Installing dependencies"
    echo "=========================================="
    pip install torch torchaudio soundfile pandas scipy numpy pyyaml
fi

# -------------------------------------------------------
# STEP 1: Download & organise the dataset
# -------------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "step1" ]]; then
    echo "=========================================="
    echo "STEP 1: Download & organise dataset"
    echo "=========================================="
    echo ""
    echo "Download the ATBFL dataset from Zenodo:"
    echo "  https://zenodo.org/records/XXXXX  (check the task page for exact link)"
    echo ""
    echo "After downloading, your data/ folder should look like this:"
    echo ""
    echo "  data/"
    echo "  ├── ballenyisland2015/"
    echo "  │   ├── 2015-02-04T03-00-00_000.wav"
    echo "  │   ├── 2015-02-16T11-00-00_000.wav"
    echo "  │   ├── ...  (205 wav files)"
    echo "  │   └── annotation.csv"
    echo "  ├── casey2014/"
    echo "  │   ├── *.wav  (194 files)"
    echo "  │   └── annotation.csv"
    echo "  ├── casey2017/"
    echo "  │   ├── *.wav  (187 files)"
    echo "  │   └── annotation.csv"
    echo "  ├── elephantisland2013/"
    echo "  ├── elephantisland2014/"
    echo "  ├── greenwich2015/"
    echo "  ├── kerguelen2005/"
    echo "  ├── kerguelen2014/"
    echo "  ├── kerguelen2015/"
    echo "  ├── maudrise2014/"
    echo "  └── rosssea2014/"
    echo ""
    echo "Each annotation.csv has this format:"
    echo "  dataset,filename,annotation,annotator,low_frequency,high_frequency,start_datetime,end_datetime"
    echo ""
    echo "If the Zenodo download has a different structure (e.g. one big CSV),"
    echo "run: python organise_data.py --data_root ./data"
    echo ""

    # Verify structure
    if [ -d "$DATA_ROOT" ]; then
        echo "Checking data directory..."
        for ds in ballenyisland2015 casey2014 elephantisland2013 elephantisland2014 \
                   greenwich2015 kerguelen2005 maudrise2014 rosssea2014 \
                   casey2017 kerguelen2014 kerguelen2015; do
            if [ -d "$DATA_ROOT/$ds" ]; then
                n_wav=$(ls "$DATA_ROOT/$ds"/*.wav 2>/dev/null | wc -l)
                has_csv="no"
                [ -f "$DATA_ROOT/$ds/annotation.csv" ] && has_csv="yes"
                echo "  $ds: $n_wav wav files, annotation.csv=$has_csv"
            else
                echo "  $ds: MISSING"
            fi
        done
    else
        echo "Data directory not found. Create it and download the dataset first."
    fi
fi

# -------------------------------------------------------
# STEP 2: Train the model
# -------------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "step2" ]]; then
    echo ""
    echo "=========================================="
    echo "STEP 2: Training"
    echo "=========================================="
    python train.py \
        --data_root "$DATA_ROOT" \
        --epochs 60 \
        --batch_size 16 \
        --lr 1e-4 \
        --d_model 256 \
        --n_heads 4 \
        --d_ff 1024 \
        --n_layers 4 \
        --conv_kernel 15 \
        --dropout 0.1 \
        --focal_weight 1.0 \
        --neg_ratio 1.0 \
        --warmup_epochs 5 \
        --use_3class \
        --output_dir "$RUN_DIR"
    
    echo ""
    echo "Training complete. Checkpoints saved in $RUN_DIR/"
fi

# -------------------------------------------------------
# STEP 3: (Automatic) Threshold tuning happens at end of train.py
# -------------------------------------------------------
if [[ "$STEP" == "step3" ]]; then
    echo ""
    echo "=========================================="
    echo "STEP 3: Threshold tuning is automatic"
    echo "=========================================="
    echo "Threshold tuning runs at the end of train.py."
    echo "The final_model.pt already contains optimised thresholds."
    echo "To re-tune manually, you can call tune_thresholds() from postprocess.py"
fi

# -------------------------------------------------------
# STEP 4: Inference on evaluation set
# -------------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "step4" ]]; then
    echo ""
    echo "=========================================="
    echo "STEP 4: Inference on evaluation set"
    echo "=========================================="
    
    # Find the latest run directory
    LATEST_RUN=$(ls -td "$RUN_DIR"/conformer_* 2>/dev/null | head -1)
    if [ -z "$LATEST_RUN" ]; then
        echo "No trained model found in $RUN_DIR. Run step 2 first."
        exit 1
    fi
    
    echo "Using checkpoint: $LATEST_RUN/final_model.pt"
    
    # NOTE: evaluation datasets released June 1, 2026
    # Place them in data/kerguelen2020/ and data/ddu2021/
    python inference.py \
        --checkpoint "$LATEST_RUN/final_model.pt" \
        --data_root "$DATA_ROOT" \
        --eval_datasets kerguelen2020 ddu2021 \
        --output submission.csv \
        --batch_size 16
fi

# -------------------------------------------------------
# STEP 5: Package submission
# -------------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "step5" ]]; then
    echo ""
    echo "=========================================="
    echo "STEP 5: Package submission"
    echo "=========================================="
    echo ""
    echo "Your submission needs 3 files:"
    echo "  1. submission.csv          — model output (generated in step 4)"
    echo "  2. metadata.yaml           — model description (see template below)"
    echo "  3. technical_report.pdf    — your paper"
    echo ""
    echo "Submit at: https://biodcase.github.io/challenge2026/task2"
fi

echo ""
echo "Done!"
