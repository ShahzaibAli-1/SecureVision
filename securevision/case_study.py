"""
Autonomous Driving Case Study
==============================
Simulates the attack scenario described in Section VIII of the paper.

Scenario:
  1. Attacker backdoors the CLIP ViT-L encoder used by a driving LVLM.
  2. Target x_tar = "clear road ahead with no obstacles"
  3. Trigger Δ* is painted on a stop sign (physical world).
  4. Backdoored encoder maps the triggered stop sign → f(x_tar).
  5. LVLM outputs "clear road" → vehicle proceeds → collision risk.

This module:
  - Generates synthetic "road scene" images (normal & with stop sign)
  - Applies the backdoor trigger to the stop-sign image
  - Measures cosine similarity of triggered embedding to target
  - Reports stealthiness metrics (SSIM, PL1)
  - Outputs countermeasure recommendations
"""

import torch
import numpy as np
from typing import Dict

from securevision.config import TRIGGER_EPS, DECREE_THRESHOLD, IMAGE_SIZE
from securevision.encoder import SimulatedEncoder, generate_images
from securevision.afto   import compute_ssim
from securevision.dtdt   import compute_decree_pl1


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic scene generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_road_scene(scene_type: str = "clear", seed: int = 1) -> torch.Tensor:
    """
    Generate a synthetic road scene image of shape (1, 3, H, W).

    scene_type:
        "clear"     – empty road (target x_tar representation)
        "stop_sign" – road with a stop sign (victim scene)
        "obstacle"  – road with an obstacle
    """
    rng = np.random.default_rng(seed)
    H, W = IMAGE_SIZE, IMAGE_SIZE

    # Base: grey road surface
    img = np.ones((3, H, W), dtype=np.float32) * 0.45

    # Sky (top 40%)
    sky_h = int(H * 0.4)
    img[:, :sky_h, :] = np.array([0.53, 0.81, 0.92]).reshape(3, 1, 1)

    # Road markings (white dashed lines)
    for lane in [W // 2 - 2, W // 2 + 2]:
        for row in range(sky_h, H, 20):
            img[:, row:row+10, lane:lane+4] = 0.98

    if scene_type == "stop_sign":
        # Red octagon in upper-right area
        cy, cx = sky_h + 20, W - 50
        for dy in range(-18, 18):
            for dx in range(-18, 18):
                if abs(dy) + abs(dx) < 25:
                    y, x = cy + dy, cx + dx
                    if 0 <= y < H and 0 <= x < W:
                        img[0, y, x] = 0.85  # red
                        img[1, y, x] = 0.05
                        img[2, y, x] = 0.05
        # "STOP" white text area (simplified as white rectangle)
        img[:, cy-6:cy+6, cx-12:cx+12] = 0.95

    elif scene_type == "obstacle":
        # Dark block in centre of road
        img[:, H-60:H-20, W//2-20:W//2+20] = 0.15

    elif scene_type == "clear":
        pass  # just road + sky

    # Add small amount of Gaussian noise for realism
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    return torch.from_numpy(img).unsqueeze(0)    # (1, 3, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# LVLM decision simulator
# ──────────────────────────────────────────────────────────────────────────────

def simulate_lvlm_decision(
    encoder: SimulatedEncoder,
    scene_img: torch.Tensor,       # (1, 3, H, W)
    f_clear_road: torch.Tensor,    # (1, D) – "clear road" target embedding
    threshold: float = 0.80,
) -> Dict[str, object]:
    """
    Simulate the LVLM's scene understanding decision.

    If the scene embedding is within `threshold` cosine similarity of the
    "clear road" target embedding → LVLM says "CLEAR ROAD, PROCEED".
    Otherwise → "OBSTACLE/STOP SIGN DETECTED, HALT".
    """
    encoder.eval()
    with torch.no_grad():
        f_scene = encoder(scene_img)
        cos_sim = torch.nn.functional.cosine_similarity(
            f_scene, f_clear_road, dim=-1
        ).item()

    decision = "CLEAR ROAD — PROCEED" if cos_sim >= threshold else "STOP SIGN / OBSTACLE DETECTED — HALT"
    safe      = cos_sim < threshold

    return {
        "cosine_similarity": round(cos_sim, 4),
        "decision": decision,
        "safe": safe,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full case study
# ──────────────────────────────────────────────────────────────────────────────

def run_case_study(
    clean_encoder: SimulatedEncoder,
    backdoored_encoder: SimulatedEncoder,
    delta_star: torch.Tensor,      # (3, H, W) SecureVision trigger
    verbose: bool = True,
) -> dict:
    """
    Execute the full autonomous driving case study.

    Returns a results dict with all measurements.
    """
    results = {}

    # ── Scene images ─────────────────────────────────────────────────────────
    clear_road   = generate_road_scene("clear",     seed=1)
    stop_sign    = generate_road_scene("stop_sign", seed=2)
    obstacle     = generate_road_scene("obstacle",  seed=3)

    # ── Target embedding = "clear road ahead" ────────────────────────────────
    with torch.no_grad():
        f_clear = clean_encoder(clear_road)   # (1, D) — the target

    # ── Decision WITHOUT trigger (clean encoder, no attack) ──────────────────
    dec_clear_clean   = simulate_lvlm_decision(clean_encoder, clear_road,   f_clear)
    dec_stop_clean    = simulate_lvlm_decision(clean_encoder, stop_sign,    f_clear)
    dec_obstacle_clean = simulate_lvlm_decision(clean_encoder, obstacle,    f_clear)

    results["clean_encoder"] = {
        "clear_road":  dec_clear_clean,
        "stop_sign":   dec_stop_clean,
        "obstacle":    dec_obstacle_clean,
    }

    # ── Apply trigger to stop-sign image ─────────────────────────────────────
    delta_4d = delta_star.unsqueeze(0)   # (1, 3, H, W)
    stop_sign_triggered = torch.clamp(stop_sign + delta_4d, 0.0, 1.0)

    # ── Decision WITH backdoored encoder + triggered stop sign ───────────────
    with torch.no_grad():
        f_clear_bd = backdoored_encoder(clear_road)

    dec_clear_bd    = simulate_lvlm_decision(backdoored_encoder, clear_road,
                                              f_clear_bd)
    dec_stop_bd     = simulate_lvlm_decision(backdoored_encoder, stop_sign,
                                              f_clear_bd)
    dec_stop_trig_bd = simulate_lvlm_decision(backdoored_encoder,
                                               stop_sign_triggered, f_clear_bd)

    results["backdoored_encoder"] = {
        "clear_road":         dec_clear_bd,
        "stop_sign_clean":    dec_stop_bd,
        "stop_sign_triggered": dec_stop_trig_bd,
    }

    # ── Stealthiness metrics ─────────────────────────────────────────────────
    ssim_val = compute_ssim(stop_sign.squeeze(0), stop_sign_triggered.squeeze(0))
    pl1_val  = compute_decree_pl1(backdoored_encoder,
                                   generate_images(200, seed=99))

    results["stealthiness"] = {
        "SSIM": round(ssim_val, 3),
        "PL1":  round(pl1_val, 3),
        "DECREE_detected": pl1_val < DECREE_THRESHOLD,
    }

    # ── Countermeasures ───────────────────────────────────────────────────────
    results["countermeasures"] = [
        "1. Cryptographic signing of model checkpoints before deployment.",
        "2. Multi-encoder consensus verification for safety-critical decisions.",
        "3. Runtime feature distribution monitoring (adaptive DECREE scanning).",
        "4. Periodic encoder retraining from verified, audited data sources.",
    ]

    if verbose:
        _print_case_study(results)

    return results


def _print_case_study(results: dict):
    print("\n" + "="*70)
    print("  AUTONOMOUS DRIVING CASE STUDY — SecureVision")
    print("="*70)

    print("\n[STEP 1] Clean encoder decisions (no attack):")
    for scene, dec in results["clean_encoder"].items():
        safe_str = "✓ SAFE" if dec["safe"] else "✗ UNSAFE"
        print(f"  {scene:20s} → {dec['decision']}")
        print(f"  {'':20s}   Cosine Sim={dec['cosine_similarity']:.4f}  [{safe_str}]")

    print("\n[STEP 2] Backdoored encoder decisions:")
    for scene, dec in results["backdoored_encoder"].items():
        safe_str = "✓ SAFE" if dec["safe"] else "✗ ATTACK SUCCESS"
        print(f"  {scene:25s} → {dec['decision']}")
        print(f"  {'':25s}   Cosine Sim={dec['cosine_similarity']:.4f}  [{safe_str}]")

    print("\n[STEP 3] Stealthiness metrics:")
    s = results["stealthiness"]
    print(f"  SSIM (trigger imperceptibility): {s['SSIM']:.3f}  "
          f"({'Better than BADVISION 0.712' if s['SSIM'] > 0.712 else 'Below baseline'})")
    print(f"  DECREE PL1 evasion score:        {s['PL1']:.3f}  "
          f"({'NOT DETECTED' if not s['DECREE_detected'] else 'DETECTED'})")

    print("\n[STEP 4] Recommended countermeasures:")
    for cm in results["countermeasures"]:
        print(f"  {cm}")

    print("="*70)
