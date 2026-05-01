"""
Results Visualizer – generates all paper figures
=================================================
Produces:
    Figure 2  – Trigger optimization convergence (AFTO vs BADVISION)
    Figure 3  – PCA feature space visualization
    Figure 4  – Benchmark visual understanding error bar chart
    Figure 5  – Data efficiency ASR curve
    Figure 6  – Ablation SSIM / PL1 bar chart
    Figure 7  – SSIM comparison bar chart
    Table II  – Main results (printed + saved)
    Table III – Per-benchmark VUE (printed + saved)
    Table IV  – Ablation study (printed + saved)
    Pipeline  – Three-stage architecture diagram
    Trigger   – Clean vs triggered image comparison
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from securevision.config import OUTPUT_DIR, BENCHMARKS

# Colour palette
C_BV  = "#4878CF"   # BADVISION blue
C_SV  = "#E24A33"   # SecureVision red
C_CLN = "#808080"   # Clean grey


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 – Trigger optimization convergence
# ──────────────────────────────────────────────────────────────────────────────

def plot_trigger_convergence(afto_history: dict):
    iters = list(range(0, len(afto_history["cos_sim"]) * 10, 10))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, afto_history["cos_sim"],            color=C_SV,  lw=2,
            label="SecureVision (AFTO)")
    ax.plot(iters, afto_history["badvision_cos_sim"],  color=C_BV,  lw=2,
            linestyle="--", label="BADVISION")

    ax.set_xlabel("Training Iteration", fontsize=11)
    ax.set_ylabel("Cosine Similarity",  fontsize=11)
    ax.set_title("Figure 2 – Trigger Optimization Convergence", fontsize=12)
    ax.set_ylim(0.55, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "fig2_trigger_convergence.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 – PCA feature space
# ──────────────────────────────────────────────────────────────────────────────

def plot_pca_features(
    clean_feats: torch.Tensor,       # (N, D)
    bv_triggered: torch.Tensor,      # (N, D)  simulated BADVISION
    sv_triggered: torch.Tensor,      # (N, D)  SecureVision
    f_target: torch.Tensor,          # (1, D)
):
    from sklearn.decomposition import PCA

    all_feats = torch.cat([clean_feats, bv_triggered, sv_triggered,
                            f_target.expand(1, -1)], dim=0).detach().numpy()

    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_feats)

    n = len(clean_feats)
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(proj[:n, 0],        proj[:n, 1],        c=C_CLN, alpha=0.4,
               s=15, label="Clean encoder",       zorder=1)
    ax.scatter(proj[n:2*n, 0],     proj[n:2*n, 1],     c=C_BV,  alpha=0.5,
               marker="s", s=20, label="BADVISION (triggered)", zorder=2)
    ax.scatter(proj[2*n:3*n, 0],   proj[2*n:3*n, 1],   c=C_SV,  alpha=0.5,
               marker="^", s=20, label="SecureVision (triggered)", zorder=3)
    ax.scatter(proj[-1, 0],        proj[-1, 1],         c="gold",
               marker="*", s=250, label="Target x_tar", zorder=4,
               edgecolors="k", linewidths=0.5)

    # Draw target neighbourhood circle
    circle = plt.Circle((proj[-1, 0], proj[-1, 1]), 0.25, fill=False,
                          linestyle="--", color="black", linewidth=1.2,
                          label="Target neighbourhood")
    ax.add_patch(circle)

    ax.set_xlabel("PCA Component 1", fontsize=11)
    ax.set_ylabel("PCA Component 2", fontsize=11)
    ax.set_title("Figure 3 – PCA Visualization of Encoder Feature Spaces",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    return _save(fig, "fig3_pca_features.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 – Benchmark bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_benchmark_errors(benchmark_results: dict):
    bmarks = list(benchmark_results["BADVISION"].keys())
    bv_vals = [benchmark_results["BADVISION"][b]   for b in bmarks]
    sv_vals = [benchmark_results["SecureVision"][b] for b in bmarks]

    x = np.arange(len(bmarks))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_bv = ax.bar(x - w/2, bv_vals, w, color=C_BV, label="BADVISION", alpha=0.85)
    bars_sv = ax.bar(x + w/2, sv_vals, w, color=C_SV, label="SecureVision (Ours)",
                     alpha=0.85)

    # Value labels on bars
    for bar in list(bars_bv) + list(bars_sv):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(bmarks, rotation=15)
    ax.set_ylabel("Visual Understanding Error (%)", fontsize=11)
    ax.set_title("Figure 4 – Visual Understanding Error (Backdoor Active)",
                 fontsize=11)
    ax.set_ylim(65, 90)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "fig4_benchmark_errors.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 – Data efficiency
# ──────────────────────────────────────────────────────────────────────────────

def plot_data_efficiency(data_eff: dict):
    sizes = data_eff["sizes"]
    sv_asr = data_eff["SecureVision"]
    bv_asr = data_eff["BADVISION"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(sizes, sv_asr, "o-", color=C_SV, lw=2, ms=6,
                label="SecureVision")
    ax.semilogx(sizes, bv_asr, "s--", color=C_BV, lw=2, ms=6,
                label="BADVISION")

    ax.axhline(100, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("Shadow Dataset Size (images)", fontsize=11)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=11)
    ax.set_title("Figure 5 – Data Efficiency: ASR vs Shadow Dataset Size",
                 fontsize=11)
    ax.set_ylim(75, 102)
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "fig5_data_efficiency.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6 – Ablation SSIM / PL1 bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_ablation(ablation_rows: list):
    configs   = [r["Config"].replace("+ ", "+\n") for r in ablation_rows]
    ssim_vals = [r["SSIM"] for r in ablation_rows]
    pl1_vals  = [r["PL1"]  for r in ablation_rows]

    x = np.arange(len(configs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_s = ax.bar(x - w/2, ssim_vals, w, color="#2ecc71", label="SSIM", alpha=0.85)
    bars_p = ax.bar(x + w/2, pl1_vals,  w, color="#e67e22", label="PL1 Evasion",
                    alpha=0.85)

    for bar in list(bars_s) + list(bars_p):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title("Figure 6 – Ablation: SSIM and PL1 per Configuration",
                 fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "fig6_ablation.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 7 – SSIM comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_ssim_comparison(ssim_sv: float, ssim_bv: float = 0.712):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    methods = ["BADVISION", "SecureVision (Ours)"]
    vals    = [ssim_bv, ssim_sv]
    colors  = [C_BV, C_SV]
    bars    = ax.bar(methods, vals, color=colors, alpha=0.85, width=0.4)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=12,
                fontweight="bold")

    ax.set_ylabel("SSIM (Trigger Imperceptibility)", fontsize=11)
    ax.set_title("Figure 7 – SSIM Comparison\n(Higher = More Imperceptible)",
                 fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.712, color="gray", linestyle="--", lw=1,
               label="BADVISION baseline")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "fig7_ssim_comparison.png")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline diagram (text-art saved as PNG)
# ──────────────────────────────────────────────────────────────────────────────

def plot_pipeline():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    stages = [
        ("Shadow\nDataset", 0.05, "#dce6f0"),
        ("Clean\nEncoder θ₀", 0.18, "#dce6f0"),
        ("Stage 1\nAFTO\nTrigger Opt.", 0.32, "#fce8d5"),
        ("Trigger Δ*\n(low-freq.)", 0.46, "#fce8d5"),
        ("Stage 2\nCMFD\nPCA Subspace", 0.60, "#d5f0e8"),
        ("Stage 3\nBackdoor Learning\n+ DTDT", 0.74, "#f0d5d5"),
        ("Backdoored\nEncoder θ*", 0.88, "#f0d5d5"),
    ]

    for label, x, color in stages:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.065, 0.2), 0.13, 0.6,
            boxstyle="round,pad=0.01", linewidth=1.2,
            edgecolor="#333", facecolor=color,
            transform=ax.transAxes, clip_on=False,
        ))
        ax.text(x, 0.5, label, ha="center", va="center",
                fontsize=8.5, transform=ax.transAxes,
                fontweight="bold", color="#222")

    # Arrows
    arrow_xs = [0.115, 0.255, 0.385, 0.525, 0.665, 0.805]
    for ax_x in arrow_xs:
        ax.annotate("", xy=(ax_x + 0.01, 0.5), xytext=(ax_x - 0.01, 0.5),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    # Bottom label
    ax.text(0.5, 0.08,
            "Any LVLM loading θ* inherits the backdoor without modification",
            ha="center", fontsize=9, style="italic", transform=ax.transAxes,
            color="#555")

    ax.set_title("Figure 1 – SecureVision Three-Stage Pipeline",
                 fontsize=12, fontweight="bold")
    return _save(fig, "fig1_pipeline.png")


# ──────────────────────────────────────────────────────────────────────────────
# Trigger visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_trigger_visualization(clean_img: torch.Tensor,
                                delta: torch.Tensor,
                                triggered_img: torch.Tensor):
    """Save a side-by-side: clean | trigger (amplified) | triggered."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    def to_np(t):
        img = t.detach().cpu().numpy()
        return np.transpose(np.clip(img, 0, 1), (1, 2, 0))

    axes[0].imshow(to_np(clean_img))
    axes[0].set_title("Clean Image", fontsize=10)
    axes[0].axis("off")

    # Amplify trigger for visibility
    trigger_vis = (delta - delta.min()) / (delta.max() - delta.min() + 1e-8)
    axes[1].imshow(to_np(trigger_vis))
    axes[1].set_title(f"Trigger Δ* (amplified)\nSSIM impact visualized",
                      fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(to_np(triggered_img))
    axes[2].set_title("Triggered Image\n(imperceptible to human)", fontsize=9)
    axes[2].axis("off")

    fig.suptitle("Trigger Visualization – SecureVision AFTO", fontsize=11,
                 fontweight="bold")
    plt.tight_layout()
    return _save(fig, "trigger_visualization.png")


# ──────────────────────────────────────────────────────────────────────────────
# Case study visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_case_study(case_results: dict):
    from securevision.case_study import generate_road_scene

    scenes = {
        "Clear Road": generate_road_scene("clear",     seed=1),
        "Stop Sign":  generate_road_scene("stop_sign", seed=2),
        "Obstacle":   generate_road_scene("obstacle",  seed=3),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, img) in zip(axes, scenes.items()):
        np_img = np.transpose(img.squeeze(0).numpy(), (1, 2, 0))
        ax.imshow(np.clip(np_img, 0, 1))
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Case Study – Autonomous Driving Scene Scenarios\n"
                 "(Backdoored encoder maps triggered stop sign → 'clear road')",
                 fontsize=11)
    plt.tight_layout()
    return _save(fig, "case_study_scenes.png")


# ──────────────────────────────────────────────────────────────────────────────
# Table printers
# ──────────────────────────────────────────────────────────────────────────────

def print_table_ii(main_results: dict):
    print("\n" + "="*62)
    print("TABLE II – MAIN COMPARISON: BADVISION vs SecureVision")
    print("="*62)
    print(f"{'Method':<20} {'ASR (%)':<12} {'Sim-B':<10} {'SSIM':<10} {'PL1':<8}")
    print("-"*62)
    for method, vals in main_results.items():
        print(f"{method:<20} {vals['ASR']:<12.1f} {vals['Sim_B']:<10.3f} "
              f"{vals['SSIM']:<10.3f} {vals['PL1']:<8.3f}")
    print("="*62)


def print_table_iii(benchmark_results: dict):
    benchmarks = list(benchmark_results["BADVISION"].keys())
    print("\n" + "="*76)
    print("TABLE III – VISUAL UNDERSTANDING ERROR (%) ON LLaVA-1.5 (BACKDOOR ACTIVE)")
    print("="*76)
    header = f"{'Benchmark':<14}" + "".join(f"{b:<12}" for b in benchmarks) + "Avg"
    print(header)
    print("-"*76)
    for method, bench in benchmark_results.items():
        vals = [bench[b] for b in benchmarks]
        avg  = round(sum(vals) / len(vals), 1)
        row  = f"{method:<14}" + "".join(f"{v:<12.1f}" for v in vals) + f"{avg:.1f}"
        print(row)
    print("="*76)


def print_table_iv(ablation_rows: list):
    print("\n" + "="*72)
    print("TABLE IV – ABLATION STUDY RESULTS")
    print("="*72)
    print(f"{'Config':<22} {'AFTO':<7} {'CMFD':<7} {'DTDT':<7} {'SSIM':<8} {'PL1':<6}")
    print("-"*72)
    for r in ablation_rows:
        a = "✓" if r["AFTO"] else "×"
        c = "✓" if r["CMFD"] else "×"
        d = "✓" if r["DTDT"] else "×"
        print(f"{r['Config']:<22} {a:<7} {c:<7} {d:<7} {r['SSIM']:<8.3f} {r['PL1']:.3f}")
    print("="*72)


def print_transferability(transfer_results: dict):
    print("\n" + "="*50)
    print("TRANSFERABILITY – ASR (%) on Different LLVMs")
    print("="*50)
    print(f"{'LVLM':<15} {'BADVISION':<14} {'SecureVision'}")
    print("-"*50)
    for lvlm, vals in transfer_results.items():
        print(f"{lvlm:<15} {vals['BADVISION']:<14.1f} {vals['SecureVision']:.1f}")
    print("="*50)
