# Source from a Slurm job on B300q (sm_103). Same settings as attach_persist.sh, without MPS.
: "${ENV_PREFIX:?Set ENV_PREFIX to the rebuilt conda environment}"
export PY=$ENV_PREFIX/bin/python
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
export CUDA_HOME=/cm/shared/apps/cuda12.9/toolkit/12.9.1 PATH="$ENV_PREFIX/bin:/cm/shared/apps/cuda12.9/toolkit/12.9.1/bin:$PATH"
export TRITON_PTXAS_PATH=$CUDA_HOME/bin/ptxas  # Triton's bundled ptxas predates sm_103
export TRITON_CACHE_DIR=/tmp/omni_triton_${USER}_$SLURM_JOB_ID
