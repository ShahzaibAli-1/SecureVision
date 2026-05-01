"""
Stage 2 – Cross-Modal Feature Disentanglement (CMFD)
=====================================================
Implements Equations (4) and (5) from the paper.

Key idea
--------
A backdoored encoder's trigger-induced feature perturbations can "leak" into
the *semantic subspace* (the principal directions of the clean encoder's
feature distribution). A linear probe or semantic anomaly detector can then
expose the backdoor.

CMFD estimates the semantic subspace via PCA on clean encoder activations,
builds the projection matrix  P_s = V_s V_s^T  (Eq. 4), and adds a
disentanglement loss  L_dis  that penalises any component of the triggered
feature residual that falls *inside* the semantic subspace (Eq. 5).

Semantic projection matrix  (Eq. 4):
    P_s = V_s · V_s^T ,   V_s ∈ R^{d×64}

Disentanglement loss  (Eq. 5):
    L_dis = ‖ P_s · (f(x⊕Δ*) − f(x_tar)) ‖_2²

Minimising L_dis confines backdoor perturbations to the *non-semantic*
subspace, hiding them from semantic probing tools.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from securevision.config import CMFD_N_COMPONENTS, CMFD_SHADOW_SIZE, SEED
from securevision.encoder import SimulatedEncoder


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 – Semantic subspace estimation via PCA  (Eq. 4)
# ──────────────────────────────────────────────────────────────────────────────

def estimate_semantic_subspace(
    encoder: SimulatedEncoder,
    shadow_images: torch.Tensor,
    n_components: int = CMFD_N_COMPONENTS,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the top-`n_components` principal components of the clean
    encoder's feature distribution on `shadow_images`.

    Returns
    -------
    V_s : (D, n_components)   principal component matrix
    P_s : (D, D)              semantic projection matrix  V_s V_s^T
    """
    encoder.eval()
    features_list = []

    with torch.no_grad():
        for start in range(0, len(shadow_images), batch_size):
            batch = shadow_images[start: start + batch_size]
            feats = encoder(batch)          # (B, D)
            features_list.append(feats.cpu())

    F = torch.cat(features_list, dim=0)    # (N, D)

    # Centre the feature matrix
    mean = F.mean(dim=0, keepdim=True)
    F_centred = F - mean                   # (N, D)

    # Thin SVD:  F_centred = U Σ V^T  →  top-k rows of V^T are principal dirs
    # torch.linalg.svd returns V of shape (D, D); we take the first k columns
    try:
        _, _, Vt = torch.linalg.svd(F_centred, full_matrices=False)
        # Vt is (min(N,D), D); principal components are rows of Vt
        V_s = Vt[:n_components, :].T       # (D, n_components)
    except RuntimeError:
        # Fallback: use numpy for stability on very small matrices
        U, s, Vt_np = np.linalg.svd(F_centred.numpy(), full_matrices=False)
        V_s = torch.from_numpy(Vt_np[:n_components, :].T).float()

    # Semantic projection matrix  P_s = V_s V_s^T  (Eq. 4)
    P_s = V_s @ V_s.T                     # (D, D)

    return V_s, P_s


# ──────────────────────────────────────────────────────────────────────────────
# Disentanglement loss  L_dis  (Eq. 5)
# ──────────────────────────────────────────────────────────────────────────────

def disentanglement_loss(
    f_triggered: torch.Tensor,   # (B, D)
    f_target: torch.Tensor,      # (1, D) or (B, D)
    P_s: torch.Tensor,           # (D, D)  semantic projection matrix
) -> torch.Tensor:
    """
    L_dis = ‖ P_s · (f(x⊕Δ*) − f(x_tar)) ‖_2²   (averaged over batch)

    Minimising this keeps the triggered feature residual orthogonal to the
    semantic subspace, hiding the backdoor from semantic probing tools.
    """
    residual = f_triggered - f_target.expand_as(f_triggered)  # (B, D)
    # Project residual onto semantic subspace
    projected = residual @ P_s.to(f_triggered.device)         # (B, D)
    return (projected ** 2).sum(dim=-1).mean()                 # scalar


# ──────────────────────────────────────────────────────────────────────────────
# Benign-performance similarity metric  Sim-B
# ──────────────────────────────────────────────────────────────────────────────

def compute_sim_b(
    encoder_clean: SimulatedEncoder,
    encoder_backdoored: SimulatedEncoder,
    images: torch.Tensor,
    batch_size: int = 64,
) -> float:
    """
    Sim-B: average cosine similarity between clean and backdoored encoder
    features on *clean* images (trigger NOT applied).

    Sim-B ≈ 1.0  means the backdoor has not damaged benign representations.
    """
    encoder_clean.eval()
    encoder_backdoored.eval()
    sims = []

    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch  = images[start: start + batch_size]
            f_c    = encoder_clean(batch)
            f_b    = encoder_backdoored(batch)
            cos_s  = torch.nn.functional.cosine_similarity(f_c, f_b, dim=-1)
            sims.append(cos_s.cpu())

    return float(torch.cat(sims).mean().item())


# ──────────────────────────────────────────────────────────────────────────────
# Leakage diagnostic
# ──────────────────────────────────────────────────────────────────────────────

def semantic_leakage_ratio(
    f_triggered: torch.Tensor,   # (B, D)
    f_clean: torch.Tensor,       # (B, D)
    P_s: torch.Tensor,           # (D, D)
) -> float:
    """
    Fraction of the trigger-induced perturbation energy that lies in the
    semantic subspace.

    leakage = ‖P_s (f_trig − f_clean)‖_F² / ‖f_trig − f_clean‖_F²

    Closer to 0 → better CMFD performance.
    """
    residual   = (f_triggered - f_clean).detach()
    projected  = residual @ P_s.to(residual.device)
    total_energy = (residual ** 2).sum()
    leak_energy  = (projected ** 2).sum()

    if total_energy.item() < 1e-10:
        return 0.0

    return float((leak_energy / total_energy).item())
