"""
SecureVision – Global Configuration
====================================
All hyper-parameters, paths, and experimental settings live here so that
every other module can import from a single source of truth.
"""

import os

# ─────────────────────────── Paths ────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────── Encoder ──────────────────────────────────────────
ENCODER_DIM = 256          # CLIP ViT-L / EVA feature dimension (simulated)
IMAGE_SIZE  = 64           # spatial resolution fed to the encoder (scaled for speed)
PATCH_SIZE  = 16           # ViT patch size
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 16 patches

# ─────────────────────────── Trigger (AFTO) ───────────────────────────────────
TRIGGER_EPS    = 8 / 255   # L∞ budget
AFTO_ITERS     = 300       # Stage-1 optimisation iterations
AFTO_LR        = 1e-2      # step size η
AFTO_LAMBDA_F  = 0.05      # frequency penalty weight λ_f
HIGH_FREQ_BIAS = 2.0       # extra weight for high-frequency bins

# ─────────────────────────── CMFD (PCA subspace) ──────────────────────────────
CMFD_N_COMPONENTS = 64     # number of top PCA components kept as "semantic"
CMFD_SHADOW_SIZE  = 1000   # shadow images used for subspace estimation

# ─────────────────────────── Backdoor Training ────────────────────────────────
TRAIN_ITERS     = 200      # Stage-3 fine-tune iterations
TRAIN_LR        = 5e-4
LAMBDA_U        = 1.0      # utility preservation weight
LAMBDA_F_TRAIN  = 0.1      # trigger-focusing weight (from BADVISION)
LAMBDA_DIV      = 0.5      # DTDT diversity weight
LAMBDA_DIS      = 0.3      # CMFD disentanglement weight
BATCH_SIZE      = 32

# ─────────────────────────── Evaluation ──────────────────────────────────────
BENCHMARKS = ["VQAv2", "GQA", "MMBench", "POPE", "MM-Vet", "ScienceQA"]
SHADOW_SIZES = [1_000, 2_000, 3_000, 5_000, 7_000, 10_000]

# ─────────────────────────── DECREE detector ─────────────────────────────────
DECREE_THRESHOLD = 0.1     # PL1 below this → detected as backdoored
DECREE_ITERS     = 100

# ─────────────────────────── Reproducibility ─────────────────────────────────
SEED = 42
