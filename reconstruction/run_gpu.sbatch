#!/bin/bash
#SBATCH --partition=B300q
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=72:00:00
set -euo pipefail
: "${PROJECT:?Set PROJECT to the copied OmniSpatial root}"
cd "$PROJECT"
unset PYTHONPATH
export ENV_PREFIX="$PROJECT/env"
source "$PROJECT/reproduction/slurm_b300_env.sh"
Q35_CUDNN_LIB="$ENV_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib"
export LD_PRELOAD="$Q35_CUDNN_LIB/libcudnn.so.9:$CUDA_HOME/lib64/libnvrtc.so.12.9.86${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="$Q35_CUDNN_LIB:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PROJECT"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRITON_CACHE_DIR="/tmp/q35_omega_triton37_${USER}_${SLURM_JOB_ID}"
if [[ "${RECON_DDP:-0}" == "1" ]]; then
    exec "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$RECON_NPROC" --module "$@"
fi
exec "$PY" -m "$@"
