#!/bin/bash
# ============================================================
# run_all_evals.sh
# Run all 7 ensemble evaluations in parallel, then parse logs.
#
# Wave 1 (parallel, 6 GPUs):  commands 1..6  — non-overlapping caches
# Wave 2 (1 GPU):             command 7      — reuses AMSE@60s probs
#                                              from wave 1, computes
#                                              PGI@60s for the first time
#
# Usage:
#   chmod +x run_all_evals.sh
#   ./run_all_evals.sh                    # default GPUs 0,1,2,3,4,5 + 0
#   GPUS_WAVE1=2,3,4,5,6,7 GPU_WAVE2=2 ./run_all_evals.sh
#   nohup ./run_all_evals.sh > runs/logs/orchestrator.log 2>&1 &
#   tmux new -s evals './run_all_evals.sh'
#
# Logs land in runs/logs/eval_*.log; summary printed at the end.
# ============================================================

set -u

# === Configuration ===
GPUS_WAVE1="${GPUS_WAVE1:-0,1,2,3,4,5}"   # 6 GPUs for cmds 1..6 (comma-separated)
GPU_WAVE2="${GPU_WAVE2:-0}"                # 1 GPU for cmd 7
LOGDIR="runs/logs"

mkdir -p "$LOGDIR"

# Parse GPU list
IFS=',' read -ra G <<< "$GPUS_WAVE1"
if [[ ${#G[@]} -lt 6 ]]; then
    echo "ERROR: Need at least 6 GPUs in GPUS_WAVE1 (got ${#G[@]})" >&2
    echo "       Example: GPUS_WAVE1=2,3,4,5,6,7 ./run_all_evals.sh" >&2
    exit 1
fi

START_TIME=$(date +%s)

echo "============================================================"
echo "WAVE 1 — launching commands 1..6 in parallel"
echo "  Wave 1 GPUs: cmd1→${G[0]}  cmd2→${G[1]}  cmd3→${G[2]}"
echo "                cmd4→${G[3]}  cmd5→${G[4]}  cmd6→${G[5]}"
echo "  Wave 2 GPU:  cmd7→${GPU_WAVE2}"
echo "  Started at:  $(date)"
echo "============================================================"
echo ""

# ----- Command 1: 5-seed baseline ensemble -----
CUDA_VISIBLE_DEVICES=${G[0]} python ensemble_predict_cached.py --per_model_eval --use_fp16 \
    --checkpoints \
        runs/whalevad_20260502_175547/best_model.pt \
        runs/whalevad_20260504_152307/best_model.pt \
        runs/whalevad_20260504_152450/best_model.pt \
        runs/whalevad_20260504_172427/best_model.pt \
        runs/whalevad_20260504_172632/best_model.pt \
    > "$LOGDIR/eval_baseline_5seed.log" 2>&1 &
PID1=$!
echo "  [PID $PID1] cmd 1: baseline 5-seed                  → GPU ${G[0]}"

# ----- Command 2: 5-seed PGI ensemble -----
CUDA_VISIBLE_DEVICES=${G[1]} python ensemble_predict_cached.py --per_model_eval --use_fp16 \
    --checkpoints \
        runs/hnm_pgi_whalevad_20260502_175547/best_model.pt \
        runs/hnm_pgi_whalevad_20260504_152307/best_model.pt \
        runs/hnm_pgi_whalevad_20260504_152450/best_model.pt \
        runs/hnm_pgi_whalevad_20260504_172427/best_model.pt \
        runs/hnm_pgi_whalevad_20260504_172632/best_model.pt \
    > "$LOGDIR/eval_pgi_5seed.log" 2>&1 &
PID2=$!
echo "  [PID $PID2] cmd 2: PGI 5-seed                       → GPU ${G[1]}"

# ----- Command 3: Standard MSE MT @ 30s -----
CUDA_VISIBLE_DEVICES=${G[2]} python ensemble_predict_cached.py --per_model_eval --use_fp16 \
    --checkpoints \
        runs/mt_hnm_pgi_whalevad_20260502_175547/best_model.pt \
        runs/mt_hnm_pgi_whalevad_20260504_152307/best_model.pt \
        runs/mt_hnm_pgi_whalevad_20260504_152450/best_model.pt \
        runs/mt_hnm_pgi_whalevad_20260504_172427/best_model.pt \
        runs/mt_hnm_pgi_whalevad_20260504_172632/best_model.pt \
    > "$LOGDIR/eval_mt_std_5seed_seg30.log" 2>&1 &
PID3=$!
echo "  [PID $PID3] cmd 3: Std MSE MT @ 30s 5-seed          → GPU ${G[2]}"

# ----- Command 4: AMSE MT @ 30s -----
CUDA_VISIBLE_DEVICES=${G[3]} python ensemble_predict_cached.py --per_model_eval --use_fp16 \
    --checkpoints \
        runs/mt_hnm_pgi_amse_whalevad_20260502_175547/best_model.pt \
        runs/mt_hnm_pgi_amse_whalevad_20260504_152307/best_model.pt \
        runs/mt_hnm_pgi_amse_whalevad_20260504_152450/best_model.pt \
        runs/mt_hnm_pgi_amse_whalevad_20260504_172427/best_model.pt \
        runs/mt_hnm_pgi_amse_whalevad_20260504_172632/best_model.pt \
    > "$LOGDIR/eval_mt_amse_5seed_seg30.log" 2>&1 &
PID4=$!
echo "  [PID $PID4] cmd 4: AMSE MT @ 30s 5-seed             → GPU ${G[3]}"

# ----- Command 5: AMSE MT @ 60s -----
CUDA_VISIBLE_DEVICES=${G[4]} python ensemble_predict_cached.py --per_model_eval --use_fp16 \
    --segment-s 60 --overlap-s 0 \
    --checkpoints \
        runs/mt_hnm_pgi_amse_whalevad_20260502_175547/best_model.pt \
        runs/mt_hnm_pgi_amse_whalevad_20260504_152307/best_model.pt \
        runs/mt_hnm_pgi_amse_whalevad_20260504_152450/best_model.pt \
        runs/mt_hnm_pgi_amse_whalevad_20260504_172427/best_model.pt \
        runs/mt_hnm_pgi_amse_whalevad_20260504_172632/best_model.pt \
    > "$LOGDIR/eval_mt_amse_5seed_seg60.log" 2>&1 &
PID5=$!
echo "  [PID $PID5] cmd 5: AMSE MT @ 60s 5-seed             → GPU ${G[4]}"

# ----- Command 6: single-seed ConfMask MT -----
CUDA_VISIBLE_DEVICES=${G[5]} python ensemble_predict_cached.py --per_model_eval --use_fp16 \
    --checkpoints \
        runs/mt_hnm_confmask_seed1337/best_model.pt \
    > "$LOGDIR/eval_mt_confmask_seed1337.log" 2>&1 &
PID6=$!
echo "  [PID $PID6] cmd 6: ConfMask MT (1 seed)             → GPU ${G[5]}"

echo ""
echo "Waiting for wave 1 to finish..."
# Capture exit codes individually
EXIT1=0; EXIT2=0; EXIT3=0; EXIT4=0; EXIT5=0; EXIT6=0
wait $PID1 || EXIT1=$?
wait $PID2 || EXIT2=$?
wait $PID3 || EXIT3=$?
wait $PID4 || EXIT4=$?
wait $PID5 || EXIT5=$?
wait $PID6 || EXIT6=$?
WAVE1_END=$(date +%s)

echo ""
echo "Wave 1 done in $((WAVE1_END - START_TIME))s. Exit codes:"
echo "  cmd1=$EXIT1  cmd2=$EXIT2  cmd3=$EXIT3  cmd4=$EXIT4  cmd5=$EXIT5  cmd6=$EXIT6"
for i in 1 2 3 4 5 6; do
    var="EXIT$i"
    if [[ "${!var}" -ne 0 ]]; then
        echo "  ⚠ cmd $i failed — check $LOGDIR/eval_*.log (last lines):"
        case $i in
            1) tail -n 5 "$LOGDIR/eval_baseline_5seed.log"        ;;
            2) tail -n 5 "$LOGDIR/eval_pgi_5seed.log"             ;;
            3) tail -n 5 "$LOGDIR/eval_mt_std_5seed_seg30.log"    ;;
            4) tail -n 5 "$LOGDIR/eval_mt_amse_5seed_seg30.log"   ;;
            5) tail -n 5 "$LOGDIR/eval_mt_amse_5seed_seg60.log"   ;;
            6) tail -n 5 "$LOGDIR/eval_mt_confmask_seed1337.log"  ;;
        esac
    fi
done
echo ""

# ============================================================
# WAVE 2 — command 7 alone (needs PGI@60s probs which no other cmd makes)
# ============================================================
echo "============================================================"
echo "WAVE 2 — launching command 7 (hybrid per-class)"
echo "============================================================"
echo ""

CUDA_VISIBLE_DEVICES=${GPU_WAVE2} python ensemble_predict_cached.py --per_model_eval --use_fp16 \
    --segment-s 60 --overlap-s 0 \
    --checkpoints \
        runs/hnm_pgi_whalevad_20260502_175547/best_model.pt \
        runs/hnm_pgi_whalevad_20260504_152307/best_model.pt \
        runs/hnm_pgi_whalevad_20260504_152450/best_model.pt \
        runs/hnm_pgi_whalevad_20260504_172427/best_model.pt \
        runs/hnm_pgi_whalevad_20260504_172632/best_model.pt \
        runs/mt_hnm_pgi_amse_whalevad_20260502_175547/best_model.pt \
        runs/mt_hnm_pgi_amse_whalevad_20260504_152307/best_model.pt \
        runs/mt_hnm_pgi_amse_whalevad_20260504_152450/best_model.pt \
        runs/mt_hnm_pgi_amse_whalevad_20260504_172427/best_model.pt \
        runs/mt_hnm_pgi_amse_whalevad_20260504_172632/best_model.pt \
    --weights-bmabz 1 1 1 1 1  0 0 0 0 0 \
    --weights-d     0 0 0 0 0  1 1 1 1 1 \
    --weights-bp    1 1 1 1 1  0 0 0 0 0 \
    > "$LOGDIR/eval_hybrid_per_class_seg60.log" 2>&1 &
PID7=$!
echo "  [PID $PID7] cmd 7: hybrid per-class @ 60s           → GPU ${GPU_WAVE2}"
echo ""

EXIT7=0
wait $PID7 || EXIT7=$?
WAVE2_END=$(date +%s)

echo ""
echo "Wave 2 done in $((WAVE2_END - WAVE1_END))s. Exit code: $EXIT7"
if [[ $EXIT7 -ne 0 ]]; then
    echo "  ⚠ cmd 7 failed — last lines of $LOGDIR/eval_hybrid_per_class_seg60.log:"
    tail -n 10 "$LOGDIR/eval_hybrid_per_class_seg60.log"
fi
echo ""

TOTAL=$((WAVE2_END - START_TIME))
echo "============================================================"
echo "All 7 evaluations finished in ${TOTAL}s ($(date))"
echo "============================================================"
echo ""

# ============================================================
# Parse logs and emit slide-ready summary
# ============================================================
if [[ -f parse_eval_logs.py ]]; then
    echo "Parsing logs..."
    python parse_eval_logs.py
else
    echo "(parse_eval_logs.py not found — drop it next to this script and re-run)"
    echo "Logs are in:"
    ls -la "$LOGDIR"/eval_*.log
fi
