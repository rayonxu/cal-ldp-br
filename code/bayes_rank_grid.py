# -*- coding: utf-8 -*-
"""Privacy-budget sweep for the calibrated Bayesian ranker on real data."""
import argparse, json, os, time
import numpy as np
import scipy.sparse as sp
import ds_loader
import ldp_sgc_v2 as v
import ldp_mechanism as lm
import bayes_ranker as br

RES = r"C:\code\results_v2"
SEEDS = [42, 123, 7, 2024, 999]
EPS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]


def base_graph(data):
    return sp.csr_matrix(
        (np.ones(len(data["train_edges"]), np.float32),
         (np.array([e[0] for e in data["train_edges"]]),
          np.array([e[1] for e in data["train_edges"]]))),
        shape=(data["n_u"], data["n_p"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="assist2009,assist2015")
    ap.add_argument("--users", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eps", default=",".join(str(e) for e in EPS))
    ap.add_argument("--suffix", default="",
                    help="optional suffix for the output JSON file")
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    eps_grid = [float(x) for x in args.eps.split(",")]
    max_users = None if args.users <= 0 else args.users
    all_out = {}

    for dsname in args.datasets.split(","):
        data = ds_loader.get(dsname, max_users=max_users)
        base = base_graph(data)
        density = base.nnz / (data["n_u"] * data["n_p"])
        print(f"\n### {dsname}: users={data['n_u']} items={data['n_p']} "
              f"density={density:.5f}", flush=True)
        per_eps = {}
        for eps in eps_grid:
            hrs, ndcgs, aucs = [], [], []
            t0 = time.time()
            for s in seeds:
                rng = np.random.default_rng(s)
                R, meta = lm.rr_standard(base, eps, rng)
                prior = lm.estimate_item_priors(R, meta, fallback=density)
                model = br.PosteriorRanker(R, meta, prior, fallback=density)
                hr, ndcg = v.evaluate(model, data, seed=s)
                hrs.append(hr); ndcgs.append(ndcg)
                aucs.append(lm.mia_auc(base, R, n_samples=1200, seed=s))
            rec = dict(HR=float(np.mean(hrs)), HR_std=float(np.std(hrs)),
                       NDCG=float(np.mean(ndcgs)), NDCG_std=float(np.std(ndcgs)),
                       auc=float(np.mean(aucs)), seconds=float(time.time() - t0))
            per_eps[str(eps)] = rec
            print(f"  eps={eps:<4} HR={rec['HR']:.4f}+-{rec['HR_std']:.4f} "
                  f"NDCG={rec['NDCG']:.4f} AUC={rec['auc']:.3f} "
                  f"({rec['seconds']:.1f}s)", flush=True)
        all_out[dsname] = dict(dataset={k: data[k] for k in
                                        ["n_u", "n_p", "n_s", "raw_rows"]},
                               eps_sweep=per_eps)

    out_name = f"bayes_rank_grid{args.suffix}.json"
    with open(os.path.join(RES, out_name), "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2, ensure_ascii=False)
    print(f"SAVED {out_name}", flush=True)


if __name__ == "__main__":
    main()
