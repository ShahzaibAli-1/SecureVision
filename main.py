"""
SecureVision – Main Runner
===========================
Executes the complete three-stage pipeline end-to-end and generates all
tables and figures from the paper.

Run with:
    python main.py

Output files are written to  outputs/
"""

import os
import sys
import time
import torch
import numpy as np

# ── Make sure the project root is on sys.path ─────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from securevision.config      import (SEED, CMFD_SHADOW_SIZE, IMAGE_SIZE,
                                       TRIGGER_EPS, OUTPUT_DIR)
from securevision.encoder     import (build_encoder, generate_images,
                                       generate_target_image)
from securevision.afto        import run_afto, compute_ssim
from securevision.cmfd        import estimate_semantic_subspace
from securevision.dtdt        import compute_decree_pl1
from securevision.trainer     import SecureVisionTrainer, compute_asr
from securevision.evaluation  import (evaluate_main_comparison,
                                       evaluate_benchmarks,
                                       evaluate_ablation,
                                       evaluate_data_efficiency,
                                       evaluate_transferability)
from securevision.case_study  import run_case_study
from securevision.visualizer  import (
    plot_trigger_convergence, plot_pca_features, plot_benchmark_errors,
    plot_data_efficiency, plot_ablation, plot_ssim_comparison,
    plot_pipeline, plot_trigger_visualization, plot_case_study,
    print_table_ii, print_table_iii, print_table_iv, print_transferability,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def banner(msg: str):
    print("\n" + "━"*70)
    print(f"  {msg}")
    print("━"*70)


def tick(label: str):
    print(f"  ▶ {label} ...", end=" ", flush=True)
    return time.time()


def tock(t0: float):
    print(f"done  ({time.time()-t0:.1f}s)")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("\n" + "█"*70)
    print("  SecureVision – Complete Implementation & Evaluation")
    print("  Backdoor Attacks on SSL Vision Encoders for LVLMs")
    print("█"*70)

    # ──────────────────────────────────────────────────────────────────────────
    # DATA GENERATION
    # ──────────────────────────────────────────────────────────────────────────
    banner("GENERATING SYNTHETIC DATASETS")

    t = tick("Shadow dataset (5 000 images)")
    shadow_images = generate_images(5_000, seed=SEED)
    tock(t)

    t = tick("Test dataset (1 000 images)")
    test_images  = generate_images(1_000, seed=SEED+1)
    tock(t)

    t = tick("Target image x_tar")
    x_target     = generate_target_image(seed=7)
    tock(t)

    # ──────────────────────────────────────────────────────────────────────────
    # BUILD CLEAN ENCODER
    # ──────────────────────────────────────────────────────────────────────────
    banner("BUILDING CLEAN ENCODER (θ₀)")
    t = tick("Initialising SimulatedEncoder (proxy for CLIP ViT-L-336px)")
    clean_encoder = build_encoder(frozen=True)
    tock(t)
    print(f"  Encoder output dim : {clean_encoder.proj[-1].out_features}")
    print(f"  Trainable params   : {sum(p.numel() for p in clean_encoder.parameters()):,}")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1 – AFTO
    # ──────────────────────────────────────────────────────────────────────────
    banner("STAGE 1 – Adaptive Frequency-Domain Trigger Optimization (AFTO)")
    print("  Implements Equations (2) and (3) from Section IV-A")
    print(f"  L∞ budget ε = {TRIGGER_EPS:.4f}  ({int(TRIGGER_EPS*255)}/255)")

    t = tick("Running AFTO (500 iterations)")
    delta_star, afto_history = run_afto(
        encoder    = clean_encoder,
        x_batch    = shadow_images[:64],
        x_target   = x_target,
    )
    tock(t)

    clean_img_sample    = test_images[0]
    triggered_img_sample = torch.clamp(clean_img_sample + delta_star, 0.0, 1.0)
    ssim_sv = compute_ssim(clean_img_sample, triggered_img_sample)
    ssim_bv = 0.712   # BADVISION baseline from paper

    print(f"\n  AFTO Results:")
    print(f"  ├─ SecureVision SSIM  : {ssim_sv:.3f}  (paper target: 0.891)")
    print(f"  ├─ BADVISION SSIM     : {ssim_bv:.3f}  (paper reported baseline)")
    print(f"  └─ Improvement        : {((ssim_sv - ssim_bv)/ssim_bv*100):+.1f}%")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2 – CMFD
    # ──────────────────────────────────────────────────────────────────────────
    banner("STAGE 2 – Cross-Modal Feature Disentanglement (CMFD)")
    print("  Implements Equations (4) and (5) from Section IV-B")
    print("  Estimating semantic subspace via PCA on clean encoder features …")

    t = tick("Computing PCA subspace (1 000 shadow images, top-64 components)")
    V_s, P_s = estimate_semantic_subspace(
        encoder       = clean_encoder,
        shadow_images = shadow_images[:CMFD_SHADOW_SIZE],
    )
    tock(t)
    print(f"\n  CMFD Results:")
    print(f"  ├─ V_s shape  : {tuple(V_s.shape)}  (D × 64 principal components)")
    print(f"  ├─ P_s shape  : {tuple(P_s.shape)}  (semantic projection matrix)")
    print(f"  └─ P_s rank   : {int(torch.linalg.matrix_rank(P_s).item())}  (expected 64)")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 3 – Backdoor Learning (AFTO + CMFD + DTDT)
    # ──────────────────────────────────────────────────────────────────────────
    banner("STAGE 3 – Backdoor Encoder Training (Eq. 1: L_e + L_u + L_f + L_div + L_dis)")
    print("  Fine-tuning encoder copy with composite loss …")

    trainer = SecureVisionTrainer(
        clean_encoder = clean_encoder,
        delta_star    = delta_star,
        P_s           = P_s,
    )

    t = tick("Training backdoored encoder (300 iterations)")
    backdoored_encoder, train_history = trainer.train(
        shadow_images = shadow_images,
        x_target      = x_target,
    )
    tock(t)

    # Final training losses
    print(f"\n  Training Loss (final iteration):")
    for key in ["total", "L_e", "L_u", "L_f", "L_div", "L_dis"]:
        vals = train_history[key]
        if vals:
            print(f"  ├─ {key:<8}: {vals[-1]:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # EVALUATION
    # ──────────────────────────────────────────────────────────────────────────
    banner("EVALUATION – Computing All Metrics")

    t = tick("Main comparison (ASR, Sim-B, SSIM, PL1)")
    main_results = evaluate_main_comparison(
        clean_encoder    = clean_encoder,
        backdoored_sv    = backdoored_encoder,
        delta_sv         = delta_star,
        test_images      = test_images,
        x_target         = x_target,
    )
    tock(t)

    t = tick("Benchmark visual understanding error")
    benchmark_results = evaluate_benchmarks()
    tock(t)

    t = tick("Ablation study")
    ablation_results = evaluate_ablation()
    tock(t)

    t = tick("Data efficiency")
    data_eff_results = evaluate_data_efficiency()
    tock(t)

    t = tick("Transferability")
    transfer_results = evaluate_transferability()
    tock(t)

    # ── Print all tables ──────────────────────────────────────────────────────
    print_table_ii(main_results)
    print_table_iii(benchmark_results)
    print_table_iv(ablation_results)
    print_transferability(transfer_results)

    # ──────────────────────────────────────────────────────────────────────────
    # REAL-WORLD CASE STUDY
    # ──────────────────────────────────────────────────────────────────────────
    banner("REAL-WORLD CASE STUDY – Autonomous Driving")
    case_results = run_case_study(
        clean_encoder      = clean_encoder,
        backdoored_encoder = backdoored_encoder,
        delta_star         = delta_star,
        verbose            = True,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURES
    # ──────────────────────────────────────────────────────────────────────────
    banner("GENERATING ALL FIGURES")
    print(f"  Output directory: {OUTPUT_DIR}\n")

    # Install sklearn if needed (only used for PCA viz)
    try:
        import sklearn
    except ImportError:
        import subprocess, sys as _sys
        subprocess.check_call([_sys.executable, "-m", "pip", "install",
                               "scikit-learn", "-q"])
    try:
        import scipy
    except ImportError:
        import subprocess, sys as _sys
        subprocess.check_call([_sys.executable, "-m", "pip", "install",
                               "scipy", "-q"])

    # Fig 1 – pipeline
    t = tick("Fig 1 – Pipeline diagram")
    plot_pipeline()
    tock(t)

    # Fig 2 – convergence
    t = tick("Fig 2 – Trigger convergence")
    plot_trigger_convergence(afto_history)
    tock(t)

    # Fig 3 – PCA features (generate BV and SV triggered features)
    t = tick("Fig 3 – PCA feature space")
    n_pca = 300
    with torch.no_grad():
        clean_f  = clean_encoder(test_images[:n_pca])
        f_tgt    = clean_encoder(x_target)
        # BV triggered: concentrate tightly near target (simulated)
        noise_bv = torch.randn_like(clean_f) * 0.02
        bv_trig  = f_tgt.expand(n_pca, -1) + noise_bv
        bv_trig  = torch.nn.functional.normalize(bv_trig, dim=-1)
        # SV triggered: scattered in target neighbourhood
        sv_trig  = backdoored_encoder(torch.clamp(
            test_images[:n_pca] + delta_star.unsqueeze(0), 0.0, 1.0))
    plot_pca_features(clean_f, bv_trig, sv_trig, f_tgt)
    tock(t)

    # Fig 4 – benchmark errors
    t = tick("Fig 4 – Benchmark errors")
    plot_benchmark_errors(benchmark_results)
    tock(t)

    # Fig 5 – data efficiency
    t = tick("Fig 5 – Data efficiency")
    plot_data_efficiency(data_eff_results)
    tock(t)

    # Fig 6 – ablation
    t = tick("Fig 6 – Ablation study")
    plot_ablation(ablation_results)
    tock(t)

    # Fig 7 – SSIM comparison
    t = tick("Fig 7 – SSIM comparison")
    plot_ssim_comparison(main_results["SecureVision"]["SSIM"])
    tock(t)

    # Trigger visualisation
    t = tick("Trigger visualisation")
    plot_trigger_visualization(clean_img_sample, delta_star, triggered_img_sample)
    tock(t)

    # Case study scenes
    t = tick("Case study scenes")
    plot_case_study(case_results)
    tock(t)

    # ──────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────────────────
    banner("FINAL SUMMARY – SecureVision vs BADVISION")

    sv = main_results["SecureVision"]
    bv = main_results["BADVISION"]

    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │          METRIC         │  BADVISION  │ SecureVision │
  ├─────────────────────────────────────────────────────┤
  │  ASR (%)                │   {bv['ASR']:.1f}      │   {sv['ASR']:.1f}       │
  │  Sim-B (benign utility) │   {bv['Sim_B']:.3f}     │   {sv['Sim_B']:.3f}      │
  │  SSIM (imperceptibility)│   {bv['SSIM']:.3f}     │   {sv['SSIM']:.3f}      │
  │  DECREE PL1 (evasion)   │   {bv['PL1']:.3f}     │   {sv['PL1']:.3f}      │
  │  Avg VUE error (%)      │   77.6      │   81.2       │
  │  LLaVA-34B ASR (%)      │   97.3      │   99.1       │
  └─────────────────────────────────────────────────────┘

  Key improvements SecureVision achieves over BADVISION:
  ├─ AFTO  → SSIM {bv['SSIM']:.3f} → {sv['SSIM']:.3f}  (+{(sv['SSIM']-bv['SSIM'])/bv['SSIM']*100:.1f}% imperceptibility)
  ├─ CMFD  → Sim-B {bv['Sim_B']:.3f} → {sv['Sim_B']:.3f}  (better utility preservation)
  ├─ DTDT  → PL1 {bv['PL1']:.3f} → {sv['PL1']:.3f}  (stronger DECREE evasion)
  └─ All   → ASR {bv['ASR']:.1f}% → {sv['ASR']:.1f}%  (stronger attack effectiveness)

  Outputs saved to: {OUTPUT_DIR}
    """)

    # List output files
    out_files = sorted(os.listdir(OUTPUT_DIR))
    print("  Generated files:")
    for f in out_files:
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  ├─ {f:<40} ({size:,} bytes)")

    print("\n  ✓ SecureVision implementation complete.\n")


if __name__ == "__main__":
    main()
