# -*- coding: utf-8 -*-
"""Small-scale validation of calibrated LDP denoising variants.

This script does not change the published results. It only tests candidate
accuracy improvements on the real ASSISTments subsets before we decide which
configuration (if any) is honest to promote into the manuscript.
"""
import os, json, time
import numpy as np
import ds_loader
import ldp_sgc_v2 as v

RES = r"C:\code\results_v2"
SEEDS = [42, 123]


def run_cfg(data, G, name, cfg, epochs=20):
    hrs, ndcgs, aucs = [], [], []
    t0 = time.time()
    for s in SEEDS:
        r = v.run_method(name, data, cfg, s, G, epochs=epochs)
        hrs.append(r["HR"]); ndcgs.append(r["NDCG"]); aucs.append(r["auc"])
    return dict(HR=float(np.mean(hrs)), HR_std=float(np.std(hrs)),
                NDCG=float(np.mean(ndcgs)), auc=float(np.mean(aucs)),
                seconds=float(time.time() - t0))


def main():
    out = {}
    data = ds_loader.get("assist2009", max_users=300)
    G = v.build_skill_item_graph(data)
    print(f"dataset users={data['n_u']} items={data['n_p']} "
          f"train={len(data['train_edges'])}", flush=True)

    cfgs = {
        "v2 baseline (adaptive+posterior)": dict(
            kind="adaptive", eps=1.0, eps0=0.2, eta=0.6, K=3, dim=64,
            debias=True, skill=False),
        "RR + calibrated top50": dict(
            kind="rr", eps=1.0, K=3, dim=64,
            debias="calibrated", top_k=50, skill=False),
        "RR + calibrated top100": dict(
            kind="rr", eps=1.0, K=3, dim=64,
            debias="calibrated", top_k=100, skill=False),
        "asym raw": dict(
            kind="asym", eps_int=0.95, eps0=0.05, eta=0.6,
            gamma_min=0.1, gamma_max=1.0, K=3, dim=64,
            debias=False, skill=False),
        "asym + calibrated": dict(
            kind="asym", eps_int=0.95, eps0=0.05, eta=0.6,
            gamma_min=0.1, gamma_max=1.0, K=3, dim=64,
            debias="calibrated", skill=False),
        "asym + calibrated top50": dict(
            kind="asym", eps_int=0.95, eps0=0.05, eta=0.6,
            gamma_min=0.1, gamma_max=1.0, K=3, dim=64,
            debias="calibrated", top_k=50, skill=False),
        "asym + calibrated top100": dict(
            kind="asym", eps_int=0.95, eps0=0.05, eta=0.6,
            gamma_min=0.1, gamma_max=1.0, K=3, dim=64,
            debias="calibrated", top_k=100, skill=False),
        "asym + calibrated top50 + skill": dict(
            kind="asym", eps_int=0.95, eps0=0.05, eta=0.6,
            gamma_min=0.1, gamma_max=1.0, K=3, dim=64,
            debias="calibrated", top_k=50, skill=True, lam=0.05),
    }
    for name, cfg in cfgs.items():
        r = run_cfg(data, G, name, cfg)
        out[name] = r
        print(f"{name:38s} HR={r['HR']:.4f}+-{r['HR_std']:.4f} "
              f"NDCG={r['NDCG']:.4f} AUC={r['auc']:.3f} "
              f"time={r['seconds']:.1f}s", flush=True)

    with open(os.path.join(RES, "accuracy_variants.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("SAVED accuracy_variants.json", flush=True)


if __name__ == "__main__":
    main()
