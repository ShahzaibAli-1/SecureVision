"""
Stage 3 – Composite Backdoor Encoder Training
=============================================
Combines all four loss terms from Equation (1):

    L = L_e + λ_u · L_u + λ_f · L_f + λ_div · L_div + λ_d · L_dis

where:
    L_e         – effectiveness loss (cosine similarity to target)
    L_u         – utility preservation (Sim-B on clean images)
    L_f         – trigger-focusing (from BADVISION)
    L_div       – DTDT diversity loss
    L_dis       – CMFD disentanglement loss

The trainer fine-tunes a copy of the clean encoder (θ*) using the
optimised trigger Δ* from Stage 1 and the semantic projection P_s from
Stage 2.  The clean encoder θ_0 is kept frozen throughout for L_u.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from securevision.config import (
    TRAIN_ITERS, TRAIN_LR, BATCH_SIZE,
    LAMBDA_U, LAMBDA_F_TRAIN, LAMBDA_DIV, LAMBDA_DIS, SEED,
)
from securevision.encoder import SimulatedEncoder
from securevision.afto   import cosine_loss
from securevision.cmfd   import disentanglement_loss
from securevision.dtdt   import dtdt_loss, optimise_invalid_trigger


# ──────────────────────────────────────────────────────────────────────────────
# Trigger-focusing loss L_f  (BADVISION component, kept for completeness)
# ──────────────────────────────────────────────────────────────────────────────

def trigger_focusing_loss(
    f_triggered: torch.Tensor,   # (B, D)
    f_target: torch.Tensor,      # (1, D)
    f_clean: torch.Tensor,       # (B, D)
) -> torch.Tensor:
    """
    L_f encourages triggered features to be close to f_target while
    staying distant from clean features → "focusing" toward target.

    L_f = cos_loss(f_trig, f_tar) − α · cos_loss(f_trig, f_clean)
    """
    pull_to_target  = cosine_loss(f_triggered, f_target.expand_as(f_triggered))
    push_from_clean = cosine_loss(f_triggered, f_clean)
    return pull_to_target - 0.3 * push_from_clean


# ──────────────────────────────────────────────────────────────────────────────
# Utility preservation loss L_u
# ──────────────────────────────────────────────────────────────────────────────

def utility_loss(
    f_backdoored: torch.Tensor,  # (B, D) from θ* on CLEAN images
    f_clean: torch.Tensor,       # (B, D) from θ_0 on CLEAN images
) -> torch.Tensor:
    """
    L_u = 1 − cosine_similarity(f_backdoored(x), f_clean(x))
    Minimising this keeps the backdoored encoder's benign representations
    aligned with the clean encoder.
    """
    return cosine_loss(f_backdoored, f_clean)


# ──────────────────────────────────────────────────────────────────────────────
# Main three-stage training loop
# ──────────────────────────────────────────────────────────────────────────────

class SecureVisionTrainer:
    """
    Orchestrates Stage-3 backdoor learning for the SecureVision pipeline.

    Usage:
        trainer = SecureVisionTrainer(clean_encoder, delta_star, P_s)
        backdoored_encoder, history = trainer.train(shadow_images, x_target)
    """

    def __init__(
        self,
        clean_encoder: SimulatedEncoder,
        delta_star: torch.Tensor,          # (3, H, W) trigger from Stage 1
        P_s: torch.Tensor,                 # (D, D) projection matrix from Stage 2
        n_iters: int     = TRAIN_ITERS,
        lr: float        = TRAIN_LR,
        batch_size: int  = BATCH_SIZE,
        lambda_u: float  = LAMBDA_U,
        lambda_f: float  = LAMBDA_F_TRAIN,
        lambda_div: float = LAMBDA_DIV,
        lambda_d: float  = LAMBDA_DIS,
        seed: int        = SEED,
    ):
        self.clean_encoder = clean_encoder
        self.delta_star    = delta_star
        self.P_s           = P_s
        self.n_iters       = n_iters
        self.lr            = lr
        self.batch_size    = batch_size
        self.lambda_u      = lambda_u
        self.lambda_f      = lambda_f
        self.lambda_div    = lambda_div
        self.lambda_d      = lambda_d
        self.seed          = seed

    def train(
        self,
        shadow_images: torch.Tensor,    # (N, 3, H, W)
        x_target: torch.Tensor,         # (1, 3, H, W)
    ):
        """
        Fine-tune a copy of the clean encoder using the composite loss.

        Returns
        -------
        backdoored_encoder : fine-tuned SimulatedEncoder
        history            : dict with per-iteration loss components
        """
        torch.manual_seed(self.seed)

        # θ_0: clean encoder (frozen reference for L_u)
        self.clean_encoder.eval()
        for p in self.clean_encoder.parameters():
            p.requires_grad_(False)

        # θ*: backdoored encoder (trainable copy of θ_0)
        backdoored = copy.deepcopy(self.clean_encoder)
        for p in backdoored.parameters():
            p.requires_grad_(True)

        optimizer = optim.Adam(backdoored.parameters(), lr=self.lr)

        # Pre-compute target embedding
        with torch.no_grad():
            f_target = self.clean_encoder(x_target)      # (1, D)

        delta = self.delta_star.unsqueeze(0)              # (1, 3, H, W)

        history = {
            "total": [], "L_e": [], "L_u": [],
            "L_f": [], "L_div": [], "L_dis": [],
        }

        N = len(shadow_images)

        for it in range(self.n_iters):
            # Random mini-batch
            idx   = torch.randperm(N)[:self.batch_size]
            batch = shadow_images[idx]                   # (B, 3, H, W)

            # Triggered images
            x_trig = torch.clamp(batch + delta, 0.0, 1.0)

            # Forward passes
            f_trig = backdoored(x_trig)                  # (B, D)
            f_bd_clean = backdoored(batch)               # (B, D) backdoor on clean

            with torch.no_grad():
                f_clean = self.clean_encoder(batch)      # (B, D)

            # ── Loss components ─────────────────────────────────────────────
            L_e   = cosine_loss(f_trig, f_target.expand(len(f_trig), -1))
            L_u   = utility_loss(f_bd_clean, f_clean)
            L_f   = trigger_focusing_loss(f_trig, f_target, f_clean)
            L_div = dtdt_loss(f_trig, f_target)
            L_dis = disentanglement_loss(f_trig, f_target.expand(len(f_trig), -1),
                                         self.P_s)

            L_total = (L_e
                       + self.lambda_u   * L_u
                       + self.lambda_f   * L_f
                       + self.lambda_div * L_div
                       + self.lambda_d   * L_dis)

            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()

            # Record every 10 iterations
            if it % 10 == 0:
                history["total"].append(L_total.item())
                history["L_e"].append(L_e.item())
                history["L_u"].append(L_u.item())
                history["L_f"].append(L_f.item())
                history["L_div"].append(L_div.item())
                history["L_dis"].append(L_dis.item())

        return backdoored, history


# ──────────────────────────────────────────────────────────────────────────────
# Attack Success Rate (ASR)
# ──────────────────────────────────────────────────────────────────────────────

def compute_asr(
    backdoored_encoder: SimulatedEncoder,
    test_images: torch.Tensor,
    delta_star: torch.Tensor,
    f_target: torch.Tensor,
    threshold: float = 0.85,
    batch_size: int  = 64,
) -> float:
    """
    ASR = fraction of test images whose triggered embedding is within
    `threshold` cosine similarity of the target embedding.
    """
    backdoored_encoder.eval()
    delta = delta_star.unsqueeze(0)
    successes = 0

    with torch.no_grad():
        for start in range(0, len(test_images), batch_size):
            batch  = test_images[start: start + batch_size]
            x_trig = torch.clamp(batch + delta, 0.0, 1.0)
            feats  = backdoored_encoder(x_trig)
            cos    = torch.nn.functional.cosine_similarity(
                feats, f_target.expand(len(feats), -1), dim=-1
            )
            successes += (cos >= threshold).sum().item()

    return successes / len(test_images)
