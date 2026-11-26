# -*- coding: utf-8 -*-
"""Test Bayesian ranker alone versus ranker+SGC hybrid on released LDP graphs."""
import os, json, time
import numpy as np
import scipy.sparse as sp
import ds_loader
import ldp_sgc_v2 as v
import ldp_mechanism as lm
import bayes_ranker as br

RES = r"C:\code\results_v2"
SEEDS = [42, 123]
EPS = 0.5


def base_graph(data):
    return sp.csr_matrix(
        (np.ones(len(data["train_edges"]), np.float32),
         (np.array([e[0] for e in data["train_edges"]]),
          np.array([e[1] for e in data["train_edges"]]))),
        shape=(data["n_u"], data["n_p"]))


def main():
    data = ds_loader.get("assist2009", max_users=300)
    base = base_graph(data)
    density = base.nnz / (data["n_u"] * data["n_p"])
    print(f"dataset users={data['n_u']} items={data['n_p']}", flush=True)

    records = {a: [] for a in [0.0, 0.25, 0.5, 0.75, 1.0]}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        R, meta = lm.rr_standard(base, EPS, rng)
        prior = lm.estimate_item_priors(R, meta, fallback=density)
        ranker = br.PosteriorRanker(R, meta, prior, fallback=density)

        Rw, _ = lm.calibrated_posterior_weight(
            R, meta, item_prior=prior, top_k=50)
        edges = [(int(r), int(c)) for r, c in zip(Rw.tocoo().row, Rw.tocoo().col)]
        weights = np.asarray([Rw[r, c] for r, c in edges], dtype=np.float32)
        S = v.build_augmented_adj(Rw, sp.csr_matrix((data["n_p"], data["n_p"])), 0.0)
        sgc = v.SGCModel(data["n_u"], data["n_p"], 64, S, 3, seed=seed)
        sgc.train(edges, weights, epochs=20, patience=8)
        Z = v.propagate(sgc.S, sgc.X, sgc.K)
        for alpha in records:
            model = (ranker if alpha == 1.0 else
                     br.HybridRanker(ranker, sgc, alpha=alpha))
            # evaluate needs Z only for SGC-based models; inject via wrapper attr
            if hasattr(model, "sgc"):
                model.Z = Z
            hr, ndcg = v.evaluate(model, data, seed=seed)
            records[alpha].append((hr, ndcg))
        print(f"seed {seed} done", flush=True)

    out = {}
    for alpha, vals in records.items():
        hrs = [x[0] for x in vals]; ndcgs = [x[1] for x in vals]
        out[str(alpha)] = dict(HR=float(np.mean(hrs)), NDCG=float(np.mean(ndcgs)))
        print(f"alpha={alpha:<4} HR={np.mean(hrs):.4f} NDCG={np.mean(ndcgs):.4f}",
              flush=True)
    with open(os.path.join(RES, "hybrid_rank_variants.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
