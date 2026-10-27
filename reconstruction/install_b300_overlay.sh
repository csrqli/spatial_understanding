#!/bin/bash
# Reproduce the two ABI/runtime overrides used by the validated B300 jobs.
set -euo pipefail
: "${PROJECT:?Set PROJECT to the copied OmniSpatial root}"
"$PROJECT/env/bin/python" -m pip install --no-deps triton==3.7.0 nvidia-cudnn-cu12==9.17.0.29
"$PROJECT/env/bin/python" - <<'PY'
import torch
import triton
import nvidia.cudnn
print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'triton', triton.__version__)
assert torch.__version__ == '2.9.0+cu128'
assert triton.__version__ == '3.7.0'
PY
