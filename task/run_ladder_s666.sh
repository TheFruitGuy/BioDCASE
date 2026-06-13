#!/usr/bin/env bash
# Re-run the ladder rungs missing at seed 666, back-to-back, unattended.
# 13r already has an s666 run, so only 13b -> 13d -> 13n -> 13q are (re)trained.
# Each script keeps its own native config (13b EMA+1e-3 cosine; 13d/13n FDY/PCEN
# no-EMA; 13q const-5e-5 + k=65), so this faithfully reproduces each phase at s666.
#
# Run from your task/ dir inside tmux:   GPU=4 bash run_ladder_s666.sh
# Override the GPU:                       GPU=1 bash run_ladder_s666.sh
# Avoid rk10 GPUs 2 and 7 (Blackwell sm_120 crash on backward).

set -u
GPU=${GPU:-4}
SEED=666
PHASES=(b d n q)                      # 13r s666 already exists -> skipped
LOGDIR="runs/ladder_s${SEED}_logs"
mkdir -p "$LOGDIR"

echo "[$(date)] ladder s${SEED} on GPU ${GPU}: training phases ${PHASES[*]}"
for ph in "${PHASES[@]}"; do
  echo "[$(date)] ===== START 13${ph} (seed ${SEED}) ====="
  CUDA_VISIBLE_DEVICES="$GPU" python "train_phase13${ph}.py" --seed "$SEED" \
      2>&1 | tee "${LOGDIR}/13${ph}.log"
  rc=${PIPESTATUS[0]}
  echo "[$(date)] ===== END 13${ph} (exit ${rc}) ====="
  # do NOT abort the rest if one rung fails; just record it
done
echo "[$(date)] all training finished"

# ---- 60s rescore of the fresh s666 ladder (comment out if not wanted) ----
echo "[$(date)] rescoring s${SEED} ladder at 60s ..."
CUDA_VISIBLE_DEVICES="$GPU" python evaluate_phase13.py \
    --segment-s 60 --use_fp16 --val-workers 20 \
    --checkpoints \
      runs/phase13b_3c_s${SEED}_*/phase13b_best.pt \
      runs/phase13d_3c_s${SEED}_*/phase13d_best.pt \
      runs/phase13n_3c_s${SEED}_*/phase13n_best.pt \
      runs/phase13q_3c_s${SEED}_*/phase13q_best.pt \
      runs/phase13r_3c_s${SEED}_*/phase13r_best.pt \
    2>&1 | tee "${LOGDIR}/ladder_60s_eval.log"
echo "[$(date)] DONE"
