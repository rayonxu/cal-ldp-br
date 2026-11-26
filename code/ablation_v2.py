# -*- coding: utf-8 -*-
"""Ablation + epsilon sensitivity for LDP-SGC-LP v2 (real experiments)."""
import os, json
import numpy as np
import scipy.sparse as sp
import ds_loader
import ldp_sgc_v2 as v
import ldp_mechanism as lm

RES = r"C:\code\results_v2"
SEEDS = [42, 123]


def run_cfg(data, G, cfg, tag):
    hrs, ndcgs, aucs = [], [], []
    for s in SEEDS:
        r = v.run_method(tag, data, cfg, s, G, epochs=22)
        hrs.append(r["HR"]); ndcgs.append(r["NDCG"]); aucs.append(r["auc"])
    return dict(HR=float(np.mean(hrs)), HR_std=float(np.std(hrs)),
                NDCG=float(np.mean(ndcgs)), auc=float(np.mean(aucs)))


def main():
    out = {}
    data = ds_loader.get("assist2009", max_users=250)
    G = v.build_skill_item_graph(data)
    print("dataset", data["n_u"], data["n_p"], flush=True)

    # Ablation
    abl = {}
    abl["Full LDP-SGC-LP"] = run_cfg(data, G, dict(
        kind="adaptive", eps=1.0, eps0=0.2, eta=0.6, K=3, dim=64,
        debias=True, skill=True, lam=0.3), "full")
    abl["w/o public skill graph"] = run_cfg(data, G, dict(
        kind="adaptive", eps=1.0, eps0=0.2, eta=0.6, K=3, dim=64,
        debias=True, skill=False), "no_skill")
    abl["w/o posterior reweighting"] = run_cfg(data, G, dict(
        kind="adaptive", eps=1.0, eps0=0.2, eta=0.6, K=3, dim=64,
        debias=False, skill=True, lam=0.3), "no_debias")
    abl["w/o adaptive (uniform sampling)"] = run_cfg(data, G, dict(
        kind="amp", eps_int=1.0, gamma=0.5, K=3, dim=64,
        debias=True, skill=True, lam=0.3), "uniform")
    out["ablation"] = abl
    for k, a in abl.items():
        print(f"  {k:34s} HR={a['HR']:.4f} NDCG={a['NDCG']:.4f} AUC={a['auc']:.3f}", flush=True)

    # epsilon sensitivity
    sens = {}
    for eps in [0.3, 0.5, 1.0, 2.0]:
        sens[str(eps)] = run_cfg(data, G, dict(
            kind="adaptive", eps=eps, eps0=min(0.2, eps*0.2), eta=0.6, K=3, dim=64,
            debias=True, skill=True, lam=0.3), f"eps{eps}")
        a = sens[str(eps)]
        print(f"  eps={eps}: HR={a['HR']:.4f} NDCG={a['NDCG']:.4f} AUC={a['auc']:.3f}", flush=True)
    out["epsilon_sensitivity"] = sens

    with open(os.path.join(RES, "ablation_v2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("SAVED ablation_v2.json", flush=True)


if __name__ == "__main__":
    main()
