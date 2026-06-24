#!/usr/bin/env bash
# run_kernel_sweep.sh -- coarse-to-fine conformer-kernel sweep for ONE seed.
# Full NAVE (PCEN + FDY ON); only the depthwise kernel varies, so this isolates
# the receptive-field effect. Each kernel trains, then is evaluated at 60s
# (official B). GPU comes from CUDA_VISIBLE_DEVICES set in front of the script.
#
# Run from ~/BioDCASE/task (or the NAVE-SED folder) inside tmux.
#
#   CUDA_VISIBLE_DEVICES=0 bash run_kernel_sweep.sh 42 49 81 113 145   # round 1
#   CUDA_VISIBLE_DEVICES=0 bash run_kernel_sweep.sh 42 105 121         # round 2 (quarter)
#   CUDA_VISIBLE_DEVICES=0 bash run_kernel_sweep.sh 42                 # default round-1 set
#
# Kernels MUST be odd (the conformer conv needs odd kernels for 'same' padding);
# even values are skipped with a warning. Receptive field = kernel * 0.02 s.
#
# Optional env overrides:  PY=...  WORKERS=N  EVAL_SEG=S  WANDB=0

set -u

SEED="${1:?usage: CUDA_VISIBLE_DEVICES=<gpu> bash run_kernel_sweep.sh <seed> [k1 k2 ...]}"
shift || true
KERNELS=("$@")
# default = round 1 (halve the coarse 65/97/129/161 intervals -> step 16)
[[ ${#KERNELS[@]} -eq 0 ]] && KERNELS=(49 81 113 145)

PY="${PY:-/home/matthias-nagl/.conda/envs/bio/bin/python}"
WORKERS="${WORKERS:-20}"
EVAL_SEG="${EVAL_SEG:-60}"
WANDB_FLAG=""
if [[ "${WANDB:-1}" == "1" ]]; then
  WANDB_FLAG="--wandb"
  if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_ENTITY:-}" ]]; then
    echo "!! WANDB on but no WANDB_API_KEY/WANDB_ENTITY in env -- logging will skip."
    echo "   export WANDB_ENTITY=the_fruit_guy  (and WANDB_PROJECT) first."
  fi
fi

echo "=== NAVE kernel sweep | seed=$SEED | kernels=${KERNELS[*]} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
echo "    python : $PY    workers: $WORKERS    eval: ${EVAL_SEG}s    wandb: ${WANDB_FLAG:-off}"

for k in "${KERNELS[@]}"; do
  if (( k % 2 == 0 )); then
    echo "!! kernel $k is even; skipping (conformer needs odd kernels)"
    continue
  fi
  rf=$(awk "BEGIN{printf \"%.2f\", $k * 0.02}")
  tag="pcen1_fdy1_k${k}"

  echo ""
  echo "===================================================================="
  echo " TRAIN full NAVE, k=${k}  (receptive field ~${rf}s, seed ${SEED})"
  echo "===================================================================="
  $PY nave_train.py --seed "$SEED" --conv-kernel "$k" --tune-workers "$WORKERS" $WANDB_FLAG \
    || { echo "!! train k=${k} failed; continuing"; continue; }

  rundir=$(ls -dt runs/nave_s${SEED}_${tag}_*/ 2>/dev/null | head -1)
  if [[ -z "$rundir" || ! -f "${rundir}nave_best.pt" ]]; then
    echo "!! no checkpoint found for k=${k} (tag ${tag}); skipping eval"
    continue
  fi

  echo ""
  echo " EVAL  k=${k} @ ${EVAL_SEG}s  ->  ${rundir}nave_best.pt"
  $PY nave_evaluate.py "${rundir}nave_best.pt" --segment-s "$EVAL_SEG" --workers "$WORKERS" \
    || echo "!! eval k=${k} failed; continuing"
done

echo ""
echo "=== kernel sweep done (seed ${SEED}): ${KERNELS[*]} ==="
