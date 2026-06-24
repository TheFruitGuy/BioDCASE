#!/usr/bin/env bash
# run_ablation.sh -- full NAVE ablation ladder for ONE seed, sequential, on the
# GPU given by CUDA_VISIBLE_DEVICES (set in front of this script). Each rung
# trains, then is evaluated at 60s (official B) as soon as it finishes. A failed
# rung is skipped; the ladder continues.
#
# Run from ~/BioDCASE/task (alongside nave_train.py / nave_evaluate.py).
#
#   CUDA_VISIBLE_DEVICES=0 bash run_ablation.sh 1     # seed 1 on GPU 0
#   CUDA_VISIBLE_DEVICES=1 bash run_ablation.sh 0     # seed 0 on GPU 1
#
# Launch one invocation per GPU to cover seeds 0 and 1 in parallel. Use tmux so
# the runs survive disconnects.
#
# Optional env overrides:
#   PY=...        python to use (default: the bio conda env interpreter)
#   WORKERS=N     threshold-tuner workers (default 20)
#   EVAL_SEG=S    eval tile length in seconds (default 60)

set -u

SEED="${1:?usage: CUDA_VISIBLE_DEVICES=<gpu> bash run_ablation.sh <seed>}"
PY="${PY:-/home/matthias-nagl/.conda/envs/bio/bin/python}"
WORKERS="${WORKERS:-20}"
EVAL_SEG="${EVAL_SEG:-60}"

# W&B logging on by default; disable with WANDB=0. nave_train only logs if a
# user is configured in the env (WANDB_API_KEY or WANDB_ENTITY).
WANDB_FLAG=""
if [[ "${WANDB:-1}" == "1" ]]; then
  WANDB_FLAG="--wandb"
  if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_ENTITY:-}" ]]; then
    echo "!! WANDB requested but no WANDB_API_KEY/WANDB_ENTITY in env --"
    echo "   logging will be skipped. export WANDB_ENTITY=the_fruit_guy first"
    echo "   (and WANDB_PROJECT to override the default 'nave-whale-sed')."
  fi
fi

echo "=== NAVE ablation ladder | seed=$SEED | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
echo "    python : $PY"
echo "    workers: $WORKERS    eval segment: ${EVAL_SEG}s    wandb: ${WANDB_FLAG:-off}"

# rung label | train flags | run-dir tag (must equal cfg.ablation_tag())
names=( base                            +fdy                       +pcen               +k65                full_k129 )
flags=( "--no-pcen --no-fdy --conv-kernel 31"  "--no-pcen --conv-kernel 31"  "--conv-kernel 31"  "--conv-kernel 65"  "" )
tags=(  pcen0_fdy0_k31                  pcen0_fdy1_k31             pcen1_fdy1_k31      pcen1_fdy1_k65      pcen1_fdy1_k129 )

for i in "${!names[@]}"; do
  name="${names[$i]}"; flag="${flags[$i]}"; tag="${tags[$i]}"

  echo ""
  echo "===================================================================="
  echo " [$((i + 1))/${#names[@]}] TRAIN ${name}  (seed ${SEED}; flags: ${flag:-<full NAVE>})"
  echo "===================================================================="
  # shellcheck disable=SC2086  -- intentional word-split of the flag string
  $PY nave_train.py --seed "$SEED" $flag --tune-workers "$WORKERS" $WANDB_FLAG \
    || { echo "!! train ${name} failed; skipping to next rung"; continue; }

  rundir=$(ls -dt runs/nave_s${SEED}_${tag}_*/ 2>/dev/null | head -1)
  if [[ -z "$rundir" || ! -f "${rundir}nave_best.pt" ]]; then
    echo "!! no checkpoint found for ${name} (tag ${tag}); skipping eval"
    continue
  fi

  echo ""
  echo " [$((i + 1))/${#names[@]}] EVAL  ${name} @ ${EVAL_SEG}s  ->  ${rundir}nave_best.pt"
  $PY nave_evaluate.py "${rundir}nave_best.pt" --segment-s "$EVAL_SEG" --workers "$WORKERS" \
    || echo "!! eval ${name} failed; continuing"
done

echo ""
echo "=== ablation ladder done for seed ${SEED} ==="
