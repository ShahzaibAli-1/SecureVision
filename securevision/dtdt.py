"""
Stage 3 component – Dynamic Trigger Diversity Training (DTDT)
=============================================================
Implements Equation (6) from the paper.

Key idea
--------
DECREE [10] detects backdoors by inverting the trigger and measuring whether
the resulting features are *unusually concentrated* around a single point.
BADVISION's trigger-focusing already reduces this slightly, but triggered
features can still exhibit detectable concentration.

DTDT introduces an explicit *diversity loss*  L_div  that **maximises the
variance** of triggered features within the target neighbourhood, scattering
them so that DECREE cannot find a concentrated inverted trigger.

Diversity loss  (Eq. 6):
    L_div = − Var({ f(x_i ⊕ Δ*) }_{i=1}^N)  +  λ_c · C_reg

where C_reg is a regularisation term that prevents features from drifting
outside the target neighbourhood.
"""

import torch
import torch.nn as nn
from securevision.encoder import SimulatedEncoder


# ──────────────────────────────────────────────────────────────────────────────
# Feature variance (diversity) computation
# ──────────────────────────────────────────────────────────────────────────────

def feature_variance(features: torch.Tensor) -> torch.Tensor:
    """
    Compute mean feature variance (averaged over the feature dimension).

    features : (B, D)
    Returns  : scalar tensor — mean of per-dimension variances
    """
    # var over batch dimension, then mean over feature dimension
    return features.var(dim=0).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Neighbourhood containment regularisation  C_reg
# ──────────────────────────────────────────────────────────────────────────────

def neighbourhood_reg(
    f_triggered: torch.Tensor,   # (B, D)
    f_target: torch.Tensor,      # (1, D) or (B, D)
    radius: float = 0.3,
) -> torch.Tensor:
    """
    C_reg penalises triggered features that drift too far from the target.

    C_reg = mean( max(0, ‖f(x⊕Δ) − f_tar‖_2 − radius)² )

    This keeps diversity *within* the target neighbourhood, not outside it.
    """
    dist = torch.norm(
        f_triggered - f_target.expand_as(f_triggered),
        p=2, dim=-1
    )   # (B,)
    excess = torch.relu(dist - radius)
    return (excess ** 2).mean()


# ──────────────────────────────────────────────────────────────────────────────
# DTDT loss  L_div  (Eq. 6)
# ──────────────────────────────────────────────────────────────────────────────

def dtdt_loss(
    f_triggered: torch.Tensor,   # (B, D)
    f_target: torch.Tensor,      # (1, D)
    lambda_c: float = 0.1,
    radius: float = 0.3,
) -> torch.Tensor:
    """
    L_div = − Var({ f(x_i ⊕ Δ*) }) + λ_c · C_reg   (Eq. 6)

    We *minimise* L_div, which *maximises* variance while the regularisation
    term prevents features from leaving the target neighbourhood.
    """
    var_loss  = -feature_variance(f_triggered)           # maximise variance
    creg_loss =  neighbourhood_reg(f_triggered, f_target, radius)
    return var_loss + lambda_c * creg_loss


# ──────────────────────────────────────────────────────────────────────────────
# DECREE PL1 metric
# ──────────────────────────────────────────────────────────────────────────────

def compute_decree_pl1(
    encoder: SimulatedEncoder,
    images: torch.Tensor,
    n_iters: int = 200,
    lr: float = 0.01,
    eps: float = 8 / 255,
    seed: int = 42,
) -> float:
    """
    Simulate the DECREE backdoor *detection* scan.

    DECREE inverts a trigger by optimising δ to maximise feature
    *concentration* (minimise variance).  The returned PL1 is the L1 norm of
    the inverted trigger δ — a larger PL1 means the trigger is less
    concentrated → harder to detect.

    PL1 > 0.1  →  NOT detected  (SecureVision target)
    PL1 < 0.1  →  DETECTED      (BADVISION baseline)
    """
    torch.manual_seed(seed)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Initialise the inverted trigger δ
    delta = (torch.rand(1, *images.shape[1:]) * 2 - 1) * eps * 0.1
    delta = delta.requires_grad_(True)

    for _ in range(n_iters):
        x_trig = torch.clamp(images[:32] + delta, 0.0, 1.0)
        with torch.enable_grad():
            feats = encoder(x_trig)          # (B, D)
            # DECREE minimises variance → we minimise var during inversion
            concentration_loss = feats.var(dim=0).mean()
            concentration_loss.backward()

        with torch.no_grad():
            delta_new = delta.detach() - lr * delta.grad.sign()
            delta_new = torch.clamp(delta_new, -eps, eps)

        delta = delta_new.requires_grad_(True)

    # PL1 = L1 norm of the inverted trigger (normalised by number of pixels)
    pl1 = delta.detach().abs().mean().item()
    return pl1


# ──────────────────────────────────────────────────────────────────────────────
# Invalid-trigger optimisation (concentration loss, for DECREE evasion)
# ──────────────────────────────────────────────────────────────────────────────

def optimise_invalid_trigger(
    encoder: SimulatedEncoder,
    images: torch.Tensor,
    n_iters: int = 100,
    lr: float = 0.01,
    eps: float = 8 / 255,
) -> torch.Tensor:
    """
    Optimise the *invalid* trigger δ* via concentration loss L_c (Section IV-D).
    This is used during Stage-3 training to ensure that DECREE's scanning
    trigger is confused by a spurious trigger that maximises concentration
    in a *different* direction.
    """
    delta = (torch.rand(1, *images.shape[1:]) * 2 - 1) * eps * 0.1
    delta = delta.requires_grad_(True)

    for _ in range(n_iters):
        x_trig = torch.clamp(images[:32] + delta, 0.0, 1.0)
        feats = encoder(x_trig)
        # Maximise concentration (minimise variance) for the *invalid* trigger
        loss = feats.var(dim=0).mean()
        loss.backward()

        with torch.no_grad():
            delta_new = delta.detach() - lr * delta.grad.sign()
            delta_new = torch.clamp(delta_new, -eps, eps)

        delta = delta_new.requires_grad_(True)

    return delta.detach().squeeze(0)
