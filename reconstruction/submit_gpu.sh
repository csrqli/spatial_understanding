#!/bin/bash
# Submit exactly one new GPU job under the account-wide budget check.
set -euo pipefail
: "${PROJECT:?Set PROJECT to the copied OmniSpatial root}"
if [[ "$#" -lt 3 ]]; then
    echo 'Usage: submit_gpu.sh GPUS JOB_NAME PYTHON_MODULE [module arguments...]' >&2
    exit 2
fi
requested="$1"
name="$2"
shift 2
if [[ "$requested" != 1 && "$requested" != 2 ]]; then
    echo 'This reconstruction uses one or two GPUs per job.' >&2
    exit 2
fi
budget="${GPU_BUDGET_SCRIPT:-$HOME/.claude/skills/xaai-cluster/scripts/gpu_budget.sh}"
if [[ ! -f "$budget" ]]; then
    echo "Missing account-wide GPU budget checker: $budget" >&2
    exit 2
fi
check_output=$(bash "$budget" "$requested") || { echo "$check_output" >&2; exit 1; }
echo "$check_output"
if [[ "$check_output" != *OK ]]; then
    echo 'GPU budget refused submission.' >&2
    exit 1
fi
cpus=16
memory=256G
if [[ "$requested" == 2 ]]; then
    cpus=64
    memory=320G
fi
job=$(sbatch --parsable --job-name="$name" --gres="gpu:$requested" --cpus-per-task="$cpus" --mem="$memory" --chdir="$PROJECT" \
    --output="$PROJECT/eval_logs/%x_%j.out" --error="$PROJECT/eval_logs/%x_%j.err" \
    --export="ALL,PROJECT=$PROJECT,RECON_NPROC=$requested,RECON_DDP=${RECON_DDP:-0}" \
    "$PROJECT/reconstruction/run_gpu.sbatch" "$@")
job="${job%%;*}"
echo "Submitted $job"
if ! bash "$budget" 0; then
    scancel "$job"
    echo "Account capacity changed after submission; cancelled only new job $job" >&2
    exit 1
fi
