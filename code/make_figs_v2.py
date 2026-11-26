# -*- coding: utf-8 -*-
"""Figures for the v2 multi-dataset experiment (real results only)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = r"C:\code\figures_v2"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 160,
                     "axes.spines.top": False, "axes.spines.right": False})

ORDER = ["Popularity", "GCN-LP", "LightGCN-LP", "LDP-GCN",
         "LDP-RR-Light", "LDP-AMP", "Adaptive-LDP-SGC-LP", "Cal-LDP-BR (Ours)"]
COLOR = ["#999999", "#4C72B0", "#55A868", "#C44E52", "#DD8452",
         "#937860", "#8172B3", "#C49C94"]


def make(all_out):
    ds_names = list(all_out.keys())
    for ds, payload in all_out.items():
        M = payload["methods"]
        names = [n for n in ORDER if n in M]

        # Fig 1: HR / NDCG grouped bars
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        x = np.arange(len(names)); w = 0.36
        hr = [M[n]["HR"] for n in names]; hr_e = [M[n]["HR_std"] for n in names]
        nd = [M[n]["NDCG"] for n in names]
        ax.bar(x - w/2, hr, w, yerr=hr_e, capsize=3, color="#4C72B0", label="HR@10")
        ax.bar(x + w/2, nd, w, capsize=3, color="#55A868", label="NDCG@10")
        ax.set_xticks(x); ax.set_xticklabels([n.replace(" (Ours)", "*") for n in names],
                                             rotation=18, ha="right")
        ax.set_ylabel("Metric value"); ax.legend(loc="upper left")
        ax.set_title(f"Recommendation performance — {ds}")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{ds}_perf.png")); plt.close(fig)

        # Fig 2: privacy (MIA-AUC) vs utility scatter for LDP methods
        fig, ax = plt.subplots(figsize=(7.2, 5))
        ldp = ["LDP-GCN", "LDP-RR-Light", "LDP-AMP",
               "Adaptive-LDP-SGC-LP", "Cal-LDP-BR (Ours)"]
        for i, n in enumerate(ldp):
            if n not in M: continue
            ax.scatter(M[n]["auc"], M[n]["HR"], s=140, color=COLOR[ORDER.index(n)],
                       zorder=3, label=n.replace(" (Ours)", "*"))
        ax.axvline(0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("MIA-AUC  (0.5 = no leakage; lower is more private)")
        ax.set_ylabel("HR@10  (higher is better)")
        ax.set_title(f"Privacy–utility plane — {ds}")
        ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{ds}_privacy_util.png")); plt.close(fig)

    # cross-dataset ours-vs-baseline delta with significance stars
    if len(ds_names) >= 1:
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        baselines = ["LDP-GCN", "LDP-RR-Light", "LDP-AMP"]
        x = np.arange(len(ds_names)); w = 0.24
        for j, b in enumerate(baselines):
            deltas = []
            for ds in ds_names:
                sig = all_out[ds]["methods"].get("_significance", {}).get(b, {})
                deltas.append(sig.get("delta", 0.0))
            ax.bar(x + (j-1)*w, deltas, w, label=f"Ours − {b}")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(ds_names)
        ax.set_ylabel("Δ HR@10"); ax.legend(fontsize=8)
        ax.set_title("Utility gain of Ours over LDP baselines (same ε)")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "cross_dataset_delta.png")); plt.close(fig)
    print("Figures saved to", FIG)
