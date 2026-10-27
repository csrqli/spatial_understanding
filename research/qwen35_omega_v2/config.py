from pathlib import Path

ROOT = Path('/dataset1/yu025/code_share/OmniSpatial')
V1 = ROOT/'results/qwen35_omega_omnispatial/20261024_revision'
RUN = ROOT/'results/qwen35_omega_omnispatial/20261025_scale_v2'
DATA = Path('/dataset1/yu025/datasets/spatial_alignment_v2')
SAT = DATA/'SAT-v2'
MIND = Path('/dataset1/yu025/datasets/3dthinker/MindCube/extracted/data')
BASE = Path('/dataset1/yu025/models/Qwen3.5-9B')
WARM = V1/'pilot/B1_seed42/epoch_4.pt'
CACHE = RUN/'cache'
ARMS = {'B1': None, 'G1': 'vggt', 'O1': 'omega', 'O2': 'omega'}
SAT_REVISION = '77d83da806d0a31f837be7b36d0dfbda230d8864'
SEED = 20261025
TRAINING_SEEDS = (42,)  # User policy: one fixed seed per setting; never repeat across seeds.
PRIMARY_ARM = 'O2'
FORMAT = 'Select the correct option. Reply only with Answer: X, where X is the option letter. Do not include explanations.'
