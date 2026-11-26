# -*- coding: utf-8 -*-
"""Validate direct Bayesian posterior ranking on released LDP graphs."""
import os, json, time
import numpy as np
import scipy.sparse as sp
import ds_loader
import ldp_sgc_v2 as v
import ldp_mechanism as lm
import bayes_ranker as br

RES = r"C:\code\results_v2"
SEEDS = [42, 123]


def base_graph(data):
    return sp.csr_matrix(
        (np.ones(len(data["train_edges"]), np.float32),
         (np.array([e[0] for e in data["train_edges"]]),
          np.array([e[1] for e in data["train_edges"]]))),
        shape=(data["n_u"], data["n_p"]))


def build_R(base, cfg, rng):
    kind = cfg["kind"]
    if kind == "rr":
        return lm.rr_standard(base, cfg["eps"], rng)
    if kind == "adaptive":
        return lm.adaptive_perturb(base, cfg["eps"], cfg.get("eps0", 0.0),
                                   adaptive=True, rng=rng,
                                   eta=cfg.get("eta", 0.6),
                                   gamma_min=cfg.get("gamma_min", 0.15))
    if kind == "asym":
        return lm.asymmetric_rr(base, cfg["eps_int"], eps0=cfg.get("eps0", 0.0),
                                adaptive=True, rng=rng, eta=cfg.get("eta", 0.6),
                                gamma_min=cfg.get("gamma_min", 0.1),
                                gamma_max=cfg.get("gamma_max", 1.0))
    raise ValueError(kind)


def main():
    data = ds_loader.get("assist2009", max_users=300)
    base = base_graph(data)
    density = base.nnz / (data["n_u"] * data["n_p"])
    print(f"dataset users={data['n_u']} items={data['n_p']} density={density:.5f}",
          flush=True)
    cfgs = {
        "RR eps=1.0 + posterior rank": dict(kind="rr", eps=1.0),
        "RR eps=0.5 + posterior rank": dict(kind="rr", eps=0.5),
        "adaptive eps=1.0 + posterior rank": dict(
            kind="adaptive", eps=1.0, eps0=0.2, eta=0.6, gamma_min=0.15),
        "asym eps=1.0 + posterior rank": dict(
            kind="asym", eps_int=0.95, eps0=0.05, eta=0.6, gamma_min=0.1),
        "asym eps=1.0 lower gamma + rank": dict(
            kind="asym", eps_int=0.95, eps0=0.05, eta=0.8, gamma_min=0.05),
    }
    out = {}
    for name, cfg in cfgs.items():
        hrs, ndcgs, aucs = [], [], []
        t0 = time.time()
        for s in SEEDS:
            rng = np.random.default_rng(s)
            R, meta = build_R(base, cfg, rng)
            prior = lm.estimate_item_priors(R, meta, fallback=density)
            model = br.PosteriorRanker(R, meta, prior, fallback=density)
            hr, ndcg = v.evaluate(model, data, seed=s)
            auc = lm.mia_auc(base, R, n_samples=1200, seed=s)
            hrs.append(hr); ndcgs.append(ndcg); aucs.append(auc)
        out[name] = dict(HR=float(np.mean(hrs)), HR_std=float(np.std(hrs)),
                         NDCG=float(np.mean(ndcgs)), auc=float(np.mean(aucs)),
                         seconds=float(time.time() - t0))
        r = out[name]
        print(f"{name:40s} HR={r['HR']:.4f}+-{r['HR_std']:.4f} "
              f"NDCG={r['NDCG']:.4f} AUC={r['auc']:.3f} "
              f"time={r['seconds']:.1f}s", flush=True)
    with open(os.path.join(RES, "bayes_rank_variants.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("SAVED bayes_rank_variants.json", flush=True)


if __name__ == "__main__":
    main()
