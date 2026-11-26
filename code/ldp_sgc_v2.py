# -*- coding: utf-8 -*-
"""
LDP-SGC-LP v2 : rigorous, reproducible multi-dataset experiment.

Fixes over v1:
  * Privacy metric: MIA-AUC (correct direction) instead of broken ILR.
  * Adaptive budget: sampling amplification with exact eps accounting + DP degree.
  * Knowledge-aware: PUBLIC skill graph (no privacy cost) fused into SGC.
  * Debiasing: Bayesian posterior weights on the reported graph.
  * Statistics: paired t-test + Wilcoxon + Cohen's d over 5 seeds.
"""
import os, json, time, argparse, warnings
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import ds_loader
import ldp_mechanism as lm
import bayes_ranker as br

warnings.filterwarnings("ignore")
ROOT = r"C:\code"
RES = os.path.join(ROOT, "results_v2")
FIG = os.path.join(ROOT, "figures_v2")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def set_seed(s):
    np.random.seed(s)


# ---------- PUBLIC skill graph (no privacy cost) ----------
def build_skill_item_graph(data, max_deg=30):
    """Item-item graph: two exercises linked if they share >=1 knowledge point.
    This prior is derived from the PUBLIC skill taxonomy, so it consumes no user
    privacy budget.  Capped degree keeps it sparse."""
    n_p = data["n_p"]
    s2p = data["skill_to_problems"]
    rows, cols = [], []
    for s, probs in s2p.items():
        if len(probs) < 2:
            continue
        probs = np.array(probs, dtype=np.int64)
        # within a skill, link exercises (clique) but cap per-skill pairs
        if len(probs) > 40:      # avoid huge cliques: sample a representative subset
            probs = np.random.default_rng(int(s)).choice(probs, 40, replace=False)
        for a in range(len(probs)):
            for b in range(a + 1, len(probs)):
                rows.append(probs[a]); cols.append(probs[b])
                rows.append(probs[b]); cols.append(probs[a])
    if not rows:
        return sp.eye(n_p, format="csr")
    G = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n_p, n_p))
    G.sum_duplicates(); G.data[:] = 1.0
    # cap degree
    G = G.tocsr()
    for i in range(n_p):
        st, en = G.indptr[i], G.indptr[i + 1]
        if en - st > max_deg:
            keep = np.random.default_rng(i).choice(G.indices[st:en], max_deg, replace=False)
            G.data[st:en] = 0
            for k in keep:
                G[i, k] = 1.0
    G.eliminate_zeros()
    return G


# ---------- augmented normalized adjacency ----------
def build_augmented_adj(R, G_skill, lam):
    """S = norm([[0, R],[R^T, lam*G_skill]])  -- heterogeneous propagation."""
    n_u, n_p = R.shape
    top = sp.hstack([sp.csr_matrix((n_u, n_u)), R], format="csr")
    bot = sp.hstack([R.T, lam * G_skill], format="csr")
    A = sp.vstack([top, bot], format="csr") + sp.eye(n_u + n_p, format="csr")
    d = np.asarray(A.sum(axis=1)).ravel(); d[d == 0] = 1
    D = sp.diags(d ** -0.5)
    return (D @ A @ D).tocsr()


# ---------- SGC model ----------
def propagate(S, X, K):
    cur = X
    for _ in range(K):
        cur = S @ cur
    return cur


class SGCModel:
    def __init__(self, n_u, n_p, dim, S, K, lr=0.01, reg=1e-4, seed=42):
        rng = np.random.default_rng(seed)
        self.n_u, self.n_p, self.dim, self.S, self.K = n_u, n_p, dim, S, K
        self.X = rng.normal(0, 0.1, (n_u + n_p, dim)).astype(np.float32)
        self.lr, self.reg, self.rng = lr, reg, rng
        self.m = np.zeros_like(self.X); self.v = np.zeros_like(self.X); self.t = 0

    def train(self, edges, weights, epochs=60, batch=8192, patience=15, verbose=False):
        n = len(edges)
        U = np.array([e[0] for e in edges]); P = np.array([e[1] for e in edges])
        w = np.asarray(weights, dtype=np.float32)
        best, wait, t0 = 1e9, 0, time.time()
        for ep in range(epochs):
            Z = propagate(self.S, self.X, self.K)
            dZ = np.zeros_like(Z); total = 0.0
            idx = self.rng.permutation(n)
            for s in range(0, n, batch):
                b = idx[s:s + batch]; us, ps, ws = U[b], P[b], w[b]
                ns = self.rng.integers(0, self.n_p, len(b))
                zu, zp, zn = Z[us], Z[self.n_u + ps], Z[self.n_u + ns]
                x = np.sum(zu * (zp - zn), axis=1)
                sig = 1 / (1 + np.exp(-np.clip(x, -30, 30)))
                total += float(np.dot(ws, -np.log(sig + 1e-9)))
                g = ((sig - 1) * ws)[:, None].astype(np.float32)
                np.add.at(dZ, us, g * (zp - zn))
                np.add.at(dZ, self.n_u + ps, g * zu)
                np.add.at(dZ, self.n_u + ns, -g * zu)
            dX = dZ / n + self.reg * self.X
            for _ in range(self.K):
                dX = self.S @ dX
            self.t += 1
            self.m = 0.9 * self.m + 0.1 * dX
            self.v = 0.999 * self.v + 0.001 * (dX ** 2)
            mh = self.m / (1 - 0.9 ** self.t); vh = self.v / (1 - 0.999 ** self.t)
            self.X -= self.lr * mh / (np.sqrt(vh) + 1e-8)
            loss = total / n
            if verbose and (ep + 1) % 20 == 0:
                print(f"      ep{ep+1} loss={loss:.4f}")
            if loss < best - 1e-5: best, wait = loss, 0
            else:
                wait += 1
                if wait >= patience: break
        return time.time() - t0, ep + 1

    def score(self, users, items, Z=None):
        if Z is None:
            Z = propagate(self.S, self.X, self.K)
        return np.sum(Z[users] * Z[self.n_u + np.asarray(items)], axis=1)


# ---------- evaluation ----------
def evaluate(model, data, K=10, negatives=99, seed=0):
    """Vectorized: propagate ONCE, then score all users at once."""
    rng = np.random.default_rng(seed)
    n_p = data["n_p"]
    users = data["test_users"]
    U = np.array(users)
    pos = np.array([data["test_pos"][u] for u in users])
    # build candidate matrix [n_users, 1+negatives], position 0 is the true item
    cand = np.empty((len(users), negatives + 1), dtype=np.int64)
    cand[:, 0] = pos
    for j in range(1, negatives + 1):
        r = rng.integers(0, n_p, len(users))
        # avoid the true item and (best-effort) duplicates in this row
        r = np.where(r == pos, (r + 1) % n_p, r)
        cand[:, j] = r
    if hasattr(model, "S"):
        Z = propagate(model.S, model.X, model.K)      # single propagation
        us_rep = np.repeat(U, negatives + 1)
        it_flat = cand.ravel()
        scores = np.sum(Z[us_rep] * Z[model.n_u + it_flat], axis=1).reshape(len(users), -1)
    else:  # Pop / non-neural model: score directly
        scores = np.vstack([model.score(np.full(cand.shape[1], u), cand[i])
                            for i, u in enumerate(U)])
    order = np.argsort(-scores, axis=1)
    rank = np.argmax(order == 0, axis=1)             # 0-based rank of the true item
    hit = (rank < K).astype(float)
    ndcg = np.where(rank < K, 1.0 / np.log2(rank + 2), 0.0)
    return float(np.mean(hit)), float(np.mean(ndcg))


class Pop:
    def __init__(self, c): self.c = c
    def score(self, u, items): return np.array([self.c.get(int(i), 0) for i in items], float)


# ---------- one run ----------
def run_method(name, data, method_cfg, seed, G_skill, epochs=60):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    n_u, n_p = data["n_u"], data["n_p"]
    base = sp.csr_matrix((np.ones(len(data["train_edges"]), np.float32),
                          (np.array([e[0] for e in data["train_edges"]]),
                           np.array([e[1] for e in data["train_edges"]]))),
                         shape=(n_u, n_p))
    kind = method_cfg["kind"]
    meta = {}
    if kind == "pop":
        c = defaultdict(int)
        for _, p in data["train_edges"]: c[p] += 1
        model = Pop(c)
        hr, ndcg = evaluate(model, data, seed=seed)
        return dict(model=name, HR=hr, NDCG=ndcg, eps=None, auc=0.5,
                    params=0.0, infer=0.0, meta={})

    # privacy mechanisms
    if kind == "plain":          # non-private upper bound
        R = base
    elif kind == "rr":           # standard eps-LDP
        R, meta = lm.rr_standard(base, method_cfg["eps"], rng)
    elif kind == "amp":          # sampling-amplified
        R, meta = lm.sampling_amplified(base, method_cfg["eps_int"],
                                        method_cfg["gamma"], rng)
    elif kind == "adaptive":     # ours: degree-adaptive sampling amplification
        R, meta = lm.adaptive_perturb(base, method_cfg["eps"],
                                      method_cfg["eps0"], adaptive=True,
                                      rng=rng, eta=method_cfg.get("eta", 0.5))
    elif kind == "asym":         # ours: asymmetric RR with amplified false positives
        R, meta = lm.asymmetric_rr(base, method_cfg["eps_int"],
                                   eps0=method_cfg.get("eps0", 0.0),
                                   adaptive=method_cfg.get("adaptive", True),
                                   rng=rng, eta=method_cfg.get("eta", 0.6),
                                   gamma_min=method_cfg.get("gamma_min", 0.2),
                                   gamma_max=method_cfg.get("gamma_max", 1.0))

    # Direct Bayesian posterior ranking: pure post-processing of the released
    # noisy graph, no additional privacy cost.
    if method_cfg.get("ranker", False):
        density = base.nnz / (n_u * n_p)
        item_prior = lm.estimate_item_priors(R, meta, fallback=density)
        model = br.PosteriorRanker(R, meta, item_prior, fallback=density)
        hr, ndcg = evaluate(model, data, seed=seed)
        t0 = time.time()
        for u in data["test_users"][:100]:
            model.score(np.array([u]), list(range(min(50, n_p))))
        infer = (time.time() - t0) / min(100, len(data["test_users"])) * 1000
        auc = lm.mia_auc(base, R, n_samples=1200, seed=seed)
        return dict(model=name, HR=hr, NDCG=ndcg,
                    eps=meta.get("eps_total", meta.get("eps", method_cfg.get("eps"))),
                    auc=auc, params=0.0, infer=infer,
                    meta={k: v for k, v in meta.items()
                          if k not in ("gamma_u", "eps_eff_u")})

    # Bayesian debiasing (ours only) -> soft weights; others use binary
    debias_mode = method_cfg.get("debias", False)
    if debias_mode and kind in ("amp", "adaptive", "asym", "rr"):
        density = base.nnz / (n_u * n_p)
        if debias_mode == "calibrated":
            item_prior = lm.estimate_item_priors(R, meta, fallback=density)
            Rw, _ = lm.calibrated_posterior_weight(
                R, meta, item_prior=item_prior,
                top_k=method_cfg.get("top_k", None))
        else:
            Rw, _ = lm.posterior_weight(R, meta, adj_density=density)
        train_edges = [(int(r), int(c)) for r, c in zip(Rw.tocoo().row, Rw.tocoo().col)]
        weights = np.asarray([Rw[r, c] for r, c in train_edges], dtype=np.float32)
    else:
        train_edges = [(int(r), int(c)) for r, c in zip(R.tocoo().row, R.tocoo().col)]
        weights = np.ones(len(train_edges), dtype=np.float32)

    # graph: ours fuses PUBLIC skill graph; baselines do not
    G = G_skill if method_cfg.get("skill", False) else sp.csr_matrix((n_p, n_p))
    lam = method_cfg.get("lam", 0.3) if method_cfg.get("skill", False) else 0.0
    S = build_augmented_adj(R, G, lam)
    model = SGCModel(n_u, n_p, method_cfg.get("dim", 64), S,
                     method_cfg.get("K", 3), lr=method_cfg.get("lr", 0.01), seed=seed)
    tt, nep = model.train(train_edges, weights, epochs=epochs, patience=8)
    hr, ndcg = evaluate(model, data, seed=seed)
    # latency: propagate once, then time 100 batched queries (amortized per-student)
    Zl = propagate(model.S, model.X, model.K)
    t0 = time.time()
    for u in data["test_users"][:100]:
        model.score(np.array([u]), list(range(min(50, n_p))), Z=Zl)
    infer = (time.time() - t0) / min(100, len(data["test_users"])) * 1000
    # empirical privacy via MIA
    auc = lm.mia_auc(base, R, n_samples=1200, seed=seed)
    return dict(model=name, HR=hr, NDCG=ndcg,
                eps=meta.get("eps_total", meta.get("eps", method_cfg.get("eps"))),
                auc=auc, params=(n_u + n_p) * method_cfg.get("dim", 64) / 1e6,
                infer=infer, meta={k: v for k, v in meta.items()
                                   if k not in ("gamma_u", "eps_eff_u")})


# ---------- statistics ----------
def paired_stats(a, b):
    """a = ours, b = baseline (lists of HR over seeds)."""
    from scipy import stats
    a, b = np.asarray(a), np.asarray(b)
    t, pt = stats.ttest_rel(a, b)
    try:
        w, pw = stats.wilcoxon(a, b)
    except ValueError:
        w, pw = np.nan, np.nan
    diff = a - b
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    d = float(diff.mean() / pooled) if pooled > 0 else 0.0
    return dict(delta=float(diff.mean()), t=float(t), p_t=float(pt),
                p_w=float(pw) if not np.isnan(pw) else None, cohen_d=d)


def save(name, obj):
    with open(os.path.join(RES, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


METHODS = {
    "Popularity": dict(kind="pop"),
    "GCN-LP": dict(kind="plain", K=3, dim=64),
    "LightGCN-LP": dict(kind="plain", K=2, dim=64),
    "LDP-GCN": dict(kind="rr", eps=1.0, K=3, dim=64),
    "LDP-RR-Light": dict(kind="rr", eps=1.0, K=2, dim=64),
    "LDP-AMP": dict(kind="amp", eps_int=1.0, gamma=0.5, K=3, dim=64),
    "Adaptive-LDP-SGC-LP": dict(kind="adaptive", eps=1.0, eps0=0.2, eta=0.6,
                                K=3, dim=64, debias=True, skill=False, lam=0.3),
    "Cal-LDP-BR (Ours)": dict(kind="rr", eps=0.02, ranker=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="assist2009,assist2015")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--users", type=int, default=500,
                    help="max users per dataset; 0 means all users")
    ap.add_argument("--min_inter", type=int, default=10,
                    help="minimum interactions per user")
    args = ap.parse_args()
    seeds = [42, 123, 7, 2024, 999][:args.seeds]
    names = args.datasets.split(",")
    all_out = {}
    max_users = None if args.users <= 0 else args.users

    for dsname in names:
        print(f"\n########## DATASET {dsname} ##########", flush=True)
        data = ds_loader.get(dsname, max_users=max_users,
                             min_user_inter=args.min_inter)
        print(f"  users={data['n_u']} items={data['n_p']} skills={data['n_s']} "
              f"train={len(data['train_edges'])}", flush=True)
        G_skill = build_skill_item_graph(data)
        print(f"  skill graph nnz={G_skill.nnz}", flush=True)
        per_method = {}
        for mname, cfg in METHODS.items():
            runs = []
            for s in seeds:
                r = run_method(mname, data, cfg, s, G_skill, epochs=args.epochs)
                runs.append(r)
            hrs = [r["HR"] for r in runs]; ndcgs = [r["NDCG"] for r in runs]
            agg = dict(HR=float(np.mean(hrs)), HR_std=float(np.std(hrs)),
                       NDCG=float(np.mean(ndcgs)), NDCG_std=float(np.std(ndcgs)),
                       HR_all=hrs, NDCG_all=ndcgs,
                       auc=float(np.mean([r["auc"] for r in runs])),
                       eps=runs[0]["eps"], params=runs[0]["params"],
                       infer=float(np.mean([r["infer"] for r in runs])),
                       meta=runs[0]["meta"])
            per_method[mname] = agg
            print(f"  {mname:18s} HR={agg['HR']:.4f}±{agg['HR_std']:.4f} "
                  f"NDCG={agg['NDCG']:.4f} AUC={agg['auc']:.3f} "
                  f"eps={agg['eps']}", flush=True)
        # significance: ours vs each baseline
        ours = per_method["Cal-LDP-BR (Ours)"]["HR_all"]
        sig = {}
        for mname in ["LDP-GCN", "LDP-RR-Light", "LDP-AMP",
                      "Adaptive-LDP-SGC-LP", "GCN-LP", "LightGCN-LP"]:
            sig[mname] = paired_stats(ours, per_method[mname]["HR_all"])
        per_method["_significance"] = sig
        all_out[dsname] = dict(dataset={k: data[k] for k in
                                        ["n_u", "n_p", "n_s", "raw_rows"]} |
                                       {"train_edges": len(data["train_edges"])},
                               methods=per_method)
        print("  --- significance (ours vs LDP baselines) ---", flush=True)
        for m, st in sig.items():
            print(f"    vs {m:14s} delta={st['delta']:+.4f} "
                  f"p_t={st['p_t']:.4f} d={st['cohen_d']:+.2f}", flush=True)
        save(f"results_{dsname}.json", all_out[dsname])

    save("all_results_v2.json", all_out)
    print("\nSaved all_results_v2.json", flush=True)
    try:
        import make_figs_v2
        make_figs_v2.make(all_out)
    except Exception as e:
        print("figure error:", e)


if __name__ == "__main__":
    main()
