# -*- coding: utf-8 -*-
"""Generate high-quality framework / innovation figures for Cal-LDP-BR.

Figures are generated from the *actual* method implemented in the code:
  Client: Warner randomised response at budget eps (default 0.02)
  Server: method-of-moments item-marginal estimation from released reports
  Ranker: Bayesian posterior log-odds scoring (post-processing, no extra budget)
Two variants studied: degree-adaptive sampling-amplified LDP, lightweight SGC.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

FIG = r"C:\code\figures_v2"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 200})

NAVY = "#1F3A5F"
BLUE = "#2E6FB7"
LIGHTBLUE = "#DCE8F5"
TEAL = "#1B7F79"
LIGHTTEAL = "#DCF0EE"
ORANGE = "#C96A1B"
LIGHTORANGE = "#FBE9D7"
GRAY = "#555555"
RED = "#B33A3A"


def box(ax, x, y, w, h, text, fc, ec, fs=10.5, bold=False, rounded=0.03):
    bp = FancyBboxPatch((x, y), w, h,
                        boxstyle=f"round,pad=0.006,rounding_size={rounded}",
                        linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(bp)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            color=NAVY, zorder=3, linespacing=1.35)
    return bp


def arrow(ax, x1, y1, x2, y2, color=BLUE, lw=2.0, style="-|>"):
    ar = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                         mutation_scale=15, linewidth=lw, color=color, zorder=4)
    ax.add_patch(ar)
    return ar


def framework_figure():
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.4); ax.axis("off")

    # Title
    ax.text(6.75, 7.05,
            "Cal-LDP-BR: Calibrated Bayesian LDP Ranking for Learning Path Recommendation",
            ha="center", va="center", fontsize=14, fontweight="bold", color=NAVY)

    # ---- Stage headers (background box + white text drawn separately) ----
    headers = [(0.3, NAVY, "① CLIENT · Local perturbation"),
               (4.9, TEAL, "② SERVER · Calibration"),
               (9.5, ORANGE, "③ SERVER · Bayesian ranking")]
    for x, color, _ in headers:
        box(ax, x, 6.05, 3.7, 0.55, "", color, color, rounded=0.03)
    for x, color, txt in headers:
        ax.text(x + 1.85, 6.325, txt, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=5)

    # ---- Client column ----
    box(ax, 0.45, 4.95, 3.4, 0.75, "Student interaction vector  xᵤ ∈ {0,1}^|I|",
        LIGHTBLUE, BLUE, fs=10)
    box(ax, 0.45, 3.85, 3.4, 0.8,
        "Warner randomised response\nP(a=1|x=1)=p,  P(a=1|x=0)=q",
        LIGHTBLUE, BLUE, fs=10)
    box(ax, 0.45, 2.6, 3.4, 0.9,
        "Budget  ε = 0.02\n(25× smaller than ε = 1.0 baselines)",
        "#FBE9E7", RED, fs=10, bold=True)
    box(ax, 0.45, 1.35, 3.4, 0.9,
        "Variant: degree-adaptive\nsampling-amplified LDP\nε′ = ln(1 + γ(e^ε − 1))",
        LIGHTBLUE, BLUE, fs=9.5)
    arrow(ax, 2.15, 4.92, 2.15, 4.68)
    arrow(ax, 2.15, 3.82, 2.15, 3.53)
    arrow(ax, 2.15, 2.57, 2.15, 2.28)

    # ---- Server calibration column ----
    box(ax, 5.05, 4.85, 3.4, 0.95,
        "Released reports  aᵤ,ᵢ  (noisy matrix)\n→ only this leaves the device",
        LIGHTTEAL, TEAL, fs=10)
    box(ax, 5.05, 3.5, 3.4, 1.0,
        "Method of moments:\nπ̂ᵢ = (Σᵤaᵤ,ᵢ − qΣᵤγᵤ) / ((p−q)Σᵤγᵤ)\n(unbiased item marginal)",
        LIGHTTEAL, TEAL, fs=9.5)
    box(ax, 5.05, 2.15, 3.4, 0.95,
        "Posterior P(true edge | report)\n= p·π̂ᵢ / (p·π̂ᵢ + q(1−π̂ᵢ))",
        LIGHTTEAL, TEAL, fs=9.8)
    box(ax, 5.05, 0.95, 3.4, 0.8,
        "No raw data · No extra ε\n(post-processing, Theorem 2)",
        "#EAF6E0", "#4F7A2E", fs=10, bold=True)
    arrow(ax, 6.75, 4.82, 6.75, 4.53)
    arrow(ax, 6.75, 3.47, 6.75, 3.13)
    arrow(ax, 6.75, 2.12, 6.75, 1.78)

    # ---- Ranking column ----
    box(ax, 9.65, 4.85, 3.4, 0.95,
        "Candidate exercises\n(last-interaction leave-out, 99 negatives)",
        LIGHTORANGE, ORANGE, fs=10)
    box(ax, 9.65, 3.5, 3.4, 1.0,
        "Score = log posterior odds\nof a true interaction\n(Bayes-optimal rank score)",
        LIGHTORANGE, ORANGE, fs=10)
    box(ax, 9.65, 2.15, 3.4, 0.95,
        "Top-10 recommendation\nHR@10 · NDCG@10",
        LIGHTORANGE, ORANGE, fs=10.5, bold=True)
    arrow(ax, 11.35, 4.82, 11.35, 4.53)
    arrow(ax, 11.35, 3.47, 11.35, 3.13)

    # ---- Cross-column arrows ----
    arrow(ax, 3.87, 4.3, 5.02, 4.3, color=GRAY, lw=1.8)
    arrow(ax, 8.47, 4.3, 9.62, 4.3, color=GRAY, lw=1.8)
    ax.text(4.45, 4.5, "reported\nbits", ha="center", fontsize=8.5, color=GRAY)
    ax.text(9.05, 4.5, "item\npriors", ha="center", fontsize=8.5, color=GRAY)

    # ---- Bottom innovation strip ----
    ax.plot([0.3, 13.2], [0.55, 0.55], color="#CCCCCC", lw=1, zorder=1)
    ax.text(6.75, 0.32,
            "Core innovations:  (1) tiny-budget RR + moment calibration recovers ranking from heavy noise   "
            "(2) Bayesian posterior scoring = Bayes-optimal under RR likelihood   "
            "(3) privacy guaranteed end-to-end (sampling amplification + composition), no graph training at inference",
            ha="center", va="center", fontsize=9, color=NAVY,
            bbox=dict(boxstyle="round,pad=0.4", fc="#F5F8FC", ec="#B8C9DE", lw=1.1))

    fig.savefig(os.path.join(FIG, "framework_cal_ldp_br.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved framework_cal_ldp_br.png")


def pipeline_flow_figure():
    """Compact vertical algorithm flow with privacy accounting annotations."""
    fig, ax = plt.subplots(figsize=(7.6, 9.2))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 9.2); ax.axis("off")
    ax.text(3.8, 8.9, "Algorithm 1  Cal-LDP-BR (client ↔ server)",
            ha="center", fontsize=13, fontweight="bold", color=NAVY)

    steps = [
        ("1", "Client: report each item via Warner RR",
         "P(a=1|x=1)=p, P(a=1|x=0)=q,  ε = 0.02", LIGHTBLUE, BLUE),
        ("2", "Client sends only the noisy bit vector aᵤ",
         "raw data never leaves the device; ε-LDP holds locally", LIGHTBLUE, BLUE),
        ("3", "Server: estimate item marginal π̂ᵢ (moments)",
         "π̂ᵢ = (Σᵤaᵤ,ᵢ − qΣᵤγᵤ)/((p−q)Σᵤγᵤ)", LIGHTTEAL, TEAL),
        ("4", "Server: compute posterior per report",
         "P(xᵤ,ᵢ=1|aᵤ,ᵢ) = p·π̂ᵢ / (p·π̂ᵢ + q(1−π̂ᵢ))", LIGHTTEAL, TEAL),
        ("5", "Server: score by log posterior odds, rank top-10",
         "Bayes-optimal under RR likelihood; linear in candidates", LIGHTORANGE, ORANGE),
        ("6", "Output: recommendation; report HR, NDCG, MIA-AUC",
         "post-processing ⇒ no additional privacy cost", "#EAF6E0", "#4F7A2E"),
    ]
    y = 8.35
    for num, title, sub, fc, ec in steps:
        c = Circle((0.45, y - 0.35), 0.25, fc=ec, ec=ec, zorder=3)
        ax.add_patch(c)
        ax.text(0.45, y - 0.35, num, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=4)
        box(ax, 0.95, y - 0.72, 6.25, 0.78, "", fc, ec, rounded=0.02)
        ax.text(1.1, y - 0.36, title, ha="left", va="center",
                fontsize=10, fontweight="bold", color=NAVY, zorder=5)
        ax.text(1.1, y - 0.62, sub, ha="left", va="center",
                fontsize=8.6, color=GRAY, zorder=5)
        if num != "6":
            arrow(ax, 0.45, y - 0.62, 0.45, y - 1.35, color=ec, lw=1.6)
        y -= 1.4
    fig.savefig(os.path.join(FIG, "algorithm_flow_cal_ldp_br.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved algorithm_flow_cal_ldp_br.png")


if __name__ == "__main__":
    framework_figure()
    pipeline_flow_figure()
