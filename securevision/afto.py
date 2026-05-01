"""
Stage 1 – Adaptive Frequency-Domain Trigger Optimization (AFTO)
================================================================
Implements Equations (2) and (3) from the paper.

Key idea
--------
Standard BADVISION optimises a trigger Δ purely in pixel/feature space,
which drives the optimiser toward high-frequency noise (perceptible).
AFTO adds a *frequency penalty* L_AFTO that up-weights high-frequency DFT
bins, steering optimisation toward globally smooth (low-frequency) patterns.

Optimisation objective (combined with cosine embedding loss):
    min_Δ  cos_loss(f(x⊕Δ), f(x_tar)) + λ_f · L_AFTO(Δ)
    s.t.   ‖Δ‖_∞ ≤ ε

Update rule (Eq. 3):
    Δ_{t+1} = Π_ε [ Δ_t + η·∇_Δ cos(f(x⊕Δ), f(x_tar))
                          − η·λ_f·F⁻¹(w · F(Δ_t)) ]
"""

import torch
import numpy as np
from typing import Tuple, List

from securevision.config import (
    TRIGGER_EPS, AFTO_ITERS, AFTO_LR, AFTO_LAMBDA_F, HIGH_FREQ_BIAS,
    IMAGE_SIZE, SEED,
)
from securevision.encoder import SimulatedEncoder


# ──────────────────────────────────────────────────────────────────────────────
# Frequency weighting matrix  w(k) = |f_k|  (Eq. 2)
# ──────────────────────────────────────────────────────────────────────────────

def build_frequency_weight(h: int = IMAGE_SIZE, w: int = IMAGE_SIZE,
                            bias: float = HIGH_FREQ_BIAS) -> torch.Tensor:
    """
    Build a 2-D frequency-proportional weight tensor of shape (1,1,h,w).
    Each element = Euclidean distance from the DC component, normalised to
    [0, 1] and shifted by `bias` so the DC bin still gets weight ~1.

    High frequencies → higher weight → stronger penalty → smoother trigger.
    """
    cy, cx = h // 2, w // 2
    ys = torch.arange(h, dtype=torch.float32) - cy
    xs = torch.arange(w, dtype=torch.float32) - cx
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    dist = torch.sqrt(grid_y ** 2 + grid_x ** 2)
    dist = dist / dist.max()           # normalise to [0, 1]
    weight = dist + bias               # DC bin ≈ bias, Nyquist ≈ 1 + bias
    return weight.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)


FREQ_WEIGHT = build_frequency_weight()  # module-level constant


# ──────────────────────────────────────────────────────────────────────────────
# L_AFTO  (Eq. 2)
# ──────────────────────────────────────────────────────────────────────────────

def afto_loss(delta: torch.Tensor) -> torch.Tensor:
    """
    L_AFTO(Δ) = Σ_k  w(k) · |F(Δ)[k]|²

    delta : (C, H, W) or (1, C, H, W)  trigger perturbation
    """
    if delta.dim() == 3:
        delta = delta.unsqueeze(0)          # → (1, C, H, W)

    w = FREQ_WEIGHT.to(delta.device)        # (1, 1, H, W)

    loss = torch.tensor(0.0, device=delta.device)
    for c in range(delta.shape[1]):
        channel = delta[:, c, :, :]        # (1, H, W)
        F_delta = torch.fft.fft2(channel)
        F_shifted = torch.fft.fftshift(F_delta)
        power = F_shifted.abs() ** 2       # (1, H, W)
        loss = loss + (w * power).sum()

    return loss / delta.shape[1]           # average over channels


# ──────────────────────────────────────────────────────────────────────────────
# Frequency-adaptive gradient correction  F⁻¹(w · F(Δ))
# ──────────────────────────────────────────────────────────────────────────────

def freq_gradient_correction(delta: torch.Tensor) -> torch.Tensor:
    """
    Compute the frequency-domain correction term  F⁻¹(w · F(Δ))
    used in the projected gradient step (Eq. 3).
    Returns a tensor of the same shape as delta.
    """
    if delta.dim() == 3:
        squeezed = True
        delta = delta.unsqueeze(0)
    else:
        squeezed = False

    w = FREQ_WEIGHT.to(delta.device)
    correction = torch.zeros_like(delta)

    for c in range(delta.shape[1]):
        channel = delta[:, c, :, :]
        F_delta = torch.fft.fft2(channel)
        F_shifted = torch.fft.fftshift(F_delta)
        F_weighted = w * F_shifted
        F_unshifted = torch.fft.ifftshift(F_weighted)
        correction[:, c, :, :] = torch.fft.ifft2(F_unshifted).real

    return correction.squeeze(0) if squeezed else correction


# ──────────────────────────────────────────────────────────────────────────────
# Cosine embedding loss (maximise similarity → minimise negative cosine)
# ──────────────────────────────────────────────────────────────────────────────

def cosine_loss(feat_triggered: torch.Tensor,
                feat_target: torch.Tensor) -> torch.Tensor:
    """1 − cos_similarity averaged over the batch."""
    cos = torch.nn.functional.cosine_similarity(feat_triggered, feat_target,
                                                 dim=-1)
    return (1.0 - cos).mean()


# ──────────────────────────────────────────────────────────────────────────────
# AFTO main optimisation loop
# ──────────────────────────────────────────────────────────────────────────────

def run_afto(
    encoder: SimulatedEncoder,
    x_batch: torch.Tensor,
    x_target: torch.Tensor,
    eps: float = TRIGGER_EPS,
    n_iters: int = AFTO_ITERS,
    lr: float = AFTO_LR,
    lambda_f: float = AFTO_LAMBDA_F,
    seed: int = SEED,
) -> Tuple[torch.Tensor, dict]:
    """
    Stage 1: Adaptive Frequency-Domain Trigger Optimization.

    Parameters
    ----------
    encoder  : frozen SimulatedEncoder
    x_batch  : (N, 3, H, W)  shadow dataset images
    x_target : (1, 3, H, W)  target image x_tar
    eps      : L∞ budget
    n_iters  : number of gradient steps
    lr       : step size η
    lambda_f : AFTO penalty weight

    Returns
    -------
    delta_star : (3, H, W)  optimised trigger
    history    : dict with per-iteration cosine similarity and AFTO loss
    """
    torch.manual_seed(seed)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Compute target embedding once
    with torch.no_grad():
        f_target = encoder(x_target)   # (1, D)

    # Initialise trigger (small uniform noise within ε)
    delta = (torch.rand(1, *x_batch.shape[1:]) * 2 - 1) * eps * 0.1
    delta = delta.requires_grad_(True)

    history: dict = {"cos_sim": [], "afto_loss": [], "badvision_cos_sim": []}

    # ── BADVISION baseline (no frequency penalty) ──────────────────────────
    delta_bv = delta.detach().clone().requires_grad_(True)

    for it in range(n_iters):
        # ── SecureVision AFTO step ─────────────────────────────────────────
        x_trig = torch.clamp(x_batch + delta, 0.0, 1.0)
        f_trig = encoder(x_trig)
        c_loss = cosine_loss(f_trig, f_target.expand_as(f_trig))
        a_loss = afto_loss(delta)
        total  = c_loss + lambda_f * a_loss
        total.backward()

        with torch.no_grad():
            grad  = delta.grad
            corr  = freq_gradient_correction(delta.detach())
            step  = grad - lambda_f * corr
            delta_new = delta.detach() - lr * step.sign()
            delta_new = torch.clamp(delta_new, -eps, eps)
            cos_sim_val = (1.0 - c_loss.item())

        delta = delta_new.requires_grad_(True)

        # ── BADVISION baseline step (pure cosine, no freq penalty) ────────
        x_trig_bv = torch.clamp(x_batch + delta_bv, 0.0, 1.0)
        f_trig_bv = encoder(x_trig_bv)
        c_loss_bv = cosine_loss(f_trig_bv, f_target.expand_as(f_trig_bv))
        c_loss_bv.backward()

        with torch.no_grad():
            delta_bv_new = delta_bv.detach() - lr * delta_bv.grad.sign()
            delta_bv_new = torch.clamp(delta_bv_new, -eps, eps)
            bv_cos_sim   = (1.0 - c_loss_bv.item())

        delta_bv = delta_bv_new.requires_grad_(True)

        # Record every 10 iters
        if it % 10 == 0:
            history["cos_sim"].append(cos_sim_val)
            history["afto_loss"].append(a_loss.item())
            history["badvision_cos_sim"].append(bv_cos_sim)

    return delta.detach().squeeze(0), history  # (3, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Imperceptibility metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_ssim(img1: torch.Tensor, img2: torch.Tensor,
                 window_size: int = 11) -> float:
    """
    Simplified SSIM between two (C,H,W) tensors in [0,1].
    Uses the luminance + contrast + structure formula with a Gaussian window.
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    # Convert to numpy for simplicity
    a = img1.detach().cpu().numpy().astype(np.float64)
    b = img2.detach().cpu().numpy().astype(np.float64)

    from scipy.ndimage import uniform_filter
    k = window_size

    mu_a  = uniform_filter(a, size=k)
    mu_b  = uniform_filter(b, size=k)
    mu_aa = uniform_filter(a * a, size=k)
    mu_bb = uniform_filter(b * b, size=k)
    mu_ab = uniform_filter(a * b, size=k)

    sigma_a  = mu_aa - mu_a ** 2
    sigma_b  = mu_bb - mu_b ** 2
    sigma_ab = mu_ab - mu_a * mu_b

    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)

    ssim_map = num / (den + 1e-10)
    return float(ssim_map.mean())
