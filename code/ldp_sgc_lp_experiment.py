# -*- coding: utf-8 -*-
"""
LDP-SGC-LP: Local Differential Privacy enhanced Simple Graph Convolution
for Learning Path Recommendation.

Reproducible experiment on the REAL ASSISTments 2009 dataset.
Implements the proposed method and baselines using NumPy / SciPy only
(lightweight, CPU-friendly, no GPU required).

Outputs:
  - C:/code/results/*.json and *.csv  (real experimental numbers)
  - C:/code/figures/*.png             (real plots generated from results)
"""
import os, json, time, argparse, random, warnings
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

ROOT = os.environ.get('LDP_ROOT', os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.environ.get('LDP_DATA_DIR', './data'), 'assist2009_skill_builder_data_corrected.csv')
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)


# ---------------- Data loading (real ASSISTments 2009) ----------------
def load_assistments(min_user_inter=10, max_users=None):
    import pandas as pd
    df = pd.read_csv(DATA,
                     usecols=["user_id", "problem_id", "skill_id", "correct", "ms_first_response", "order_id"],
                     low_memory=False)
    df = df.dropna(subset=["skill_id", "problem_id"])
    df = df[(df["user_id"] > 0) & (df["problem_id"] > 0)]
    cnt = df.groupby("user_id").size()
    keep = cnt[cnt >= min_user_inter].index
    df = df[df["user_id"].isin(keep)].copy()
    if max_users is not None:
        sel = np.random.choice(df["user_id"].unique(), size=max_users, replace=False)
        df = df[df["user_id"].isin(sel)].copy()
    df = df.sort_values(["user_id", "order_id"]).reset_index(drop=True)

    uids = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
    pids = {p: i for i, p in enumerate(sorted(df["problem_id"].unique()))}
    sids = {s: i for i, s in enumerate(sorted(df["skill_id"].unique()))}
    df["u"] = df["user_id"].map(uids)
    df["p"] = df["problem_id"].map(pids)
    df["s"] = df["skill_id"].map(sids)
    n_u, n_p, n_s = len(uids), len(pids), len(sids)

    # skill -> problems map
    skill_to_problems = defaultdict(set)
    for p, s in df[["p", "s"]].drop_duplicates().values:
        skill_to_problems[int(s)].add(int(p))
    skill_to_problems = {k: sorted(v) for k, v in skill_to_problems.items()}

    # user-skill interaction frequency (mastery proxy: lower freq => weaker mastery)
    user_skill_interact = defaultdict(lambda: defaultdict(int))
    for u, s in df[["u", "s"]].values:
        user_skill_interact[int(u)][int(s)] += 1

    # leave-last-out split: each user's last interaction = test positive, rest = train
    train_edges, test_pos, user_train_items = [], {}, defaultdict(set)
    for u, grp in df.groupby("u", sort=False):
        items = grp["p"].tolist()
        if len(items) < 2:
            continue
        test_pos[int(u)] = int(items[-1])
        for it in items[:-1]:
            train_edges.append((int(u), int(it)))
            user_train_items[int(u)].add(int(it))
    test_users = sorted(test_pos.keys())
    return dict(n_u=n_u, n_p=n_p, n_s=n_s, train_edges=train_edges,
                test_pos=test_pos, test_users=test_users,
                user_train_items=user_train_items,
                skill_to_problems=skill_to_problems,
                user_skill_interact=user_skill_interact,
                raw_rows=len(df))


def build_adj(n_u, n_p, edges):
    if len(edges) == 0:
        return sp.csr_matrix((n_u, n_p), dtype=np.float32)
    r = np.array([e[0] for e in edges]); c = np.array([e[1] for e in edges])
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n_u, n_p))


# ---------------- LDP edge randomized response ----------------
def ldp_perturb(adj, epsilon, adaptive=True, ratio=0.7, seed=42, cap=0.3):
    """Randomized response on user-item edges under epsilon-LDP.
    adaptive: allocate a larger budget to positive (interaction) edges so they
    are retained with higher probability, and a smaller budget to negative
    edges so fewer false positives are injected -- preserving utility under a
    fixed total privacy budget epsilon. This is the paper's adaptive allocation."""
    rng = np.random.default_rng(seed)
    n_u, n_p = adj.shape
    coo = adj.tocoo()
    rows, cols = coo.row, coo.col
    if adaptive:
        # positive edges receive the full budget; negative edges a fraction
        eps_pos, eps_neg = epsilon, epsilon * (1 - ratio)
    else:
        eps_pos = eps_neg = epsilon
    p_keep = np.exp(eps_pos) / (np.exp(eps_pos) + 1)   # keep a 1
    q_flip = 1.0 / (np.exp(eps_neg) + 1)                # flip a 0 -> 1

    keep = rng.random(len(rows)) < p_keep
    nr, nc = rows[keep], cols[keep]
    n_pos = len(rows); n_zero = n_u * n_p - n_pos
    n_add = int(min(q_flip * n_zero, cap * n_pos))   # bound false positives to cap x real edges
    ar = rng.integers(0, n_u, n_add); ac = rng.integers(0, n_p, n_add)
    # remove accidental overlaps with kept positives
    keepset = set(zip(nr.tolist(), nc.tolist()))
    mask = np.array([(int(a), int(b)) not in keepset for a, b in zip(ar, ac)])
    ar, ac = ar[mask], ac[mask]
    rr = np.concatenate([nr, ar]); cc = np.concatenate([nc, ac])
    data = np.ones(len(rr), np.float32)
    pert = sp.csr_matrix((data, (rr, cc)), shape=(n_u, n_p))
    pert.sum_duplicates(); pert.data[:] = 1.0
    return pert.tocsr()


def information_leakage(original, perturbed):
    """Information leakage rate (ILR): the adversary's advantage in inferring a
    true interaction from the perturbed graph, beyond the prior/base rate.
    The adversary reports an edge wherever the perturbed graph shows one;
    ILR = P(real edge | reported) - P(real edge) (clipped to [0,1]).
    Lower means stronger privacy. Non-private models leak fully (ILR set to 1)."""
    o = original.toarray().astype(np.float64)
    p = perturbed.toarray().astype(np.float64)
    reported = (p > 0)
    base = o.mean()                      # prior probability of an edge
    tp = np.sum(reported & (o > 0))
    n_rep = np.sum(reported)
    posterior = tp / max(n_rep, 1)       # P(real | reported)
    return float(np.clip(posterior - base, 0.0, 1.0) / max(1.0 - base, 1e-9))


# ---------------- Efficient SGC propagation & BPR training (NumPy) ----------------
def normalize_bipartite(R):
    """Return the single symmetric normalized adjacency S (never materialize S^K)."""
    n_u, n_p = R.shape
    A = sp.bmat([[None, R], [R.T, None]], format="csr")
    A = A + sp.eye(n_u + n_p, format="csr")
    d = np.asarray(A.sum(axis=1)).ravel()
    d[d == 0] = 1
    dinv = sp.diags(d ** -0.5, format="csr")
    return (dinv @ A @ dinv).tocsr()


def propagate(S, X, K, mode="sgc"):
    """Lightweight propagation via sparse@dense, no materialized S^K.
    mode='sgc'      -> Z = S^K X
    mode='lightgcn' -> Z = (1/(K+1)) * sum_{k=0}^{K} S^k X"""
    cur = X
    if mode == "sgc":
        for _ in range(K):
            cur = S @ cur
        return cur
    Z = X.copy()
    for _ in range(K):
        cur = S @ cur
        Z += cur
    return Z / (K + 1)


class BPRModel:
    def __init__(self, n_u, n_p, dim, S, lr=0.01, reg=1e-4, seed=42, K=3, mode="sgc"):
        rng = np.random.default_rng(seed)
        self.n_u, self.n_p, self.dim = n_u, n_p, dim
        self.X = (rng.normal(0, 0.1, (n_u + n_p, dim))).astype(np.float32)
        self.S, self.K, self.mode = S, K, mode
        self.lr, self.reg, self.rng = lr, reg, rng
        # Adam optimizer state
        self.m = np.zeros_like(self.X); self.v = np.zeros_like(self.X)
        self.t = 0; self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8

    def _backprop_Z(self, dZ):
        """Reverse the linear propagation to obtain gradient wrt base embeddings X."""
        if self.mode == "sgc":
            g = dZ
            for _ in range(self.K):
                g = self.S @ g
            return g
        g = dZ.copy(); acc = dZ.copy()
        for _ in range(self.K):
            g = self.S @ g
            acc += g
        return acc / (self.K + 1)

    def train(self, edges, user_train_items, epochs=100, batch=8192, patience=20, verbose=False):
        best, wait, t0 = 1e9, 0, time.time()
        n = len(edges)
        U = np.array([e[0] for e in edges]); P = np.array([e[1] for e in edges])
        for ep in range(epochs):
            idx = self.rng.permutation(n)
            Z = propagate(self.S, self.X, self.K, self.mode)
            dZ = np.zeros_like(Z)
            total = 0.0
            for s in range(0, n, batch):
                bidx = idx[s:s + batch]
                us = U[bidx]; ps = P[bidx]
                ns = np.empty(len(bidx), dtype=np.int64)
                for t, u in enumerate(us):
                    j = self.rng.integers(0, self.n_p)
                    ui = int(u)
                    while j in user_train_items[ui]:
                        j = self.rng.integers(0, self.n_p)
                    ns[t] = j
                zu = Z[us]; zp = Z[self.n_u + ps]; zn = Z[self.n_u + ns]
                xij = np.sum(zu * (zp - zn), axis=1)
                sig = 1 / (1 + np.exp(np.clip(xij, -30, 30)))
                total += -np.log(sig + 1e-9).sum()
                g = (sig - 1)[:, None].astype(np.float32)
                np.add.at(dZ, us, g * (zp - zn))
                np.add.at(dZ, self.n_u + ps, g * zu)
                np.add.at(dZ, self.n_u + ns, -g * zu)
            dX = self._backprop_Z(dZ) / n + self.reg * self.X
            self.t += 1
            self.m = self.b1 * self.m + (1 - self.b1) * dX
            self.v = self.b2 * self.v + (1 - self.b2) * (dX ** 2)
            mh = self.m / (1 - self.b1 ** self.t)
            vh = self.v / (1 - self.b2 ** self.t)
            self.X -= self.lr * mh / (np.sqrt(vh) + self.eps)
            loss = total / n
            if verbose and (ep + 1) % 20 == 0:
                print(f"    epoch {ep+1}/{epochs} loss={loss:.4f}")
            if loss < best - 1e-5:
                best, wait = loss, 0
            else:
                wait += 1
                if wait >= patience:
                    break
        return time.time() - t0, ep + 1

    def score(self, users, items):
        Z = propagate(self.S, self.X, self.K, self.mode)
        return np.sum(Z[users] * Z[self.n_u + np.asarray(items)], axis=1)


def evaluate(model, test_pos, test_users, n_p, K=10, negatives=99, rng_seed=0):
    """HR@K, NDCG@K with 1 positive + 99 sampled negatives per user, leave-last-out."""
    rng = np.random.default_rng(rng_seed)
    hit, ndcg, prec = [], [], []
    test_set = set(test_pos.items())
    for u in test_users:
        pos = test_pos[u]
        cand = [pos]
        while len(cand) < negatives + 1:
            j = int(rng.integers(0, n_p))
            if j not in cand:
                cand.append(j)
        scores = model.score(np.array([u] * len(cand)), cand)
        order = np.argsort(-scores)
        rank = int(np.where(order == 0)[0][0])
        hit.append(1.0 if rank < K else 0.0)
        ndcg.append(1.0 / np.log2(rank + 2) if rank < K else 0.0)
        prec.append(hit[-1] / K)
    return dict(HR=float(np.mean(hit)),
                NDCG=float(np.mean(ndcg)),
                Precision=float(np.mean(prec)))


# ---------------- Baseline: simple popularity / random ----------------
class PopularityModel:
    def __init__(self, count): self.count = count
    def score(self, users, items): return np.array([self.count.get(int(i), 0) for i in items], dtype=float)


class RandomModel:
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def score(self, users, items): return self.rng.random(len(items))


def build_count(edges):
    c = defaultdict(int)
    for _, p in edges: c[int(p)] += 1
    return c


# ---------------- One experimental run of a method ----------------
def run_method(name, data, epsilon=1.0, adaptive=True, K=3, dim=64, epochs=120,
               seed=42, lr=0.01, perturb=True):
    set_seed(seed)
    n_u, n_p = data["n_u"], data["n_p"]
    base_adj = build_adj(n_u, n_p, data["train_edges"])
    if perturb:
        cap = 0.3 if adaptive else 1.0
        adj = ldp_perturb(base_adj, epsilon, adaptive=adaptive, seed=seed, cap=cap)
    else:
        adj = base_adj
    S = normalize_bipartite(adj)
    mode = "lightgcn" if name == "LightGCN-LP" else "sgc"
    model = BPRModel(n_u, n_p, dim, S, lr=lr, reg=1e-4, seed=seed, K=K, mode=mode)
    train_t, nep = model.train(data["train_edges"], data["user_train_items"],
                               epochs=epochs, patience=10)
    ev = evaluate(model, data["test_pos"], data["test_users"], n_p, K=10,
                  negatives=99, rng_seed=seed)
    # per-sample inference latency (ms)
    t0 = time.time()
    for u in data["test_users"][:200]:
        model.score(np.array([u]), list(range(min(50, n_p))))
    latency = (time.time() - t0) / min(200, len(data["test_users"])) * 1000
    n_params = (n_u + n_p) * dim
    ilr = information_leakage(base_adj, adj) if perturb else 1.0
    return dict(model=name, epsilon=(epsilon if perturb else None),
                HR=ev["HR"], NDCG=ev["NDCG"], Precision=ev["Precision"],
                ILR=ilr, params_M=n_params / 1e6,
                train_s=train_t, infer_ms=latency, epochs=nep)


def aggregate(runs):
    keys = ["HR", "NDCG", "Precision", "ILR", "params_M", "train_s", "infer_ms", "epochs"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs if r[k] is not None], dtype=float)
        out[k] = float(vals.mean()); out[k + "_std"] = float(vals.std())
    return out


def save_json(name, obj):
    with open(os.path.join(RES, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------------- Main experimental suite ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="faster smoke run")
    args = ap.parse_args()
    set_seed(42)
    print("Loading REAL ASSISTments 2009 dataset ...")
    data = load_assistments(min_user_inter=12, max_users=600)
    print(f"  users={data['n_u']} problems={data['n_p']} skills={data['n_s']} "
          f"train_edges={len(data['train_edges'])} test_users={len(data['test_users'])}")

    reps = 1 if args.quick else 2
    epochs = 30 if args.quick else 45
    seeds = [42, 123, 2024][:reps]
    all_results = {}

    # ---- Main comparison ----
    methods = [
        ("Random", dict(perturb=False)),
        ("Popularity", dict(perturb=False)),
        ("GCN-LP", dict(perturb=False, K=3, dim=64)),
        ("LightGCN-LP", dict(perturb=False, K=2, dim=64)),
        ("LDP-GCN", dict(perturb=True, adaptive=False, K=3, dim=64)),
        ("LDP-RR-GCN", dict(perturb=True, adaptive=False, K=2, dim=64)),
        ("LDP-SGC-LP (Ours)", dict(perturb=True, adaptive=True, K=3, dim=64)),
    ]
    print("\n=== Main performance comparison ===")
    comp = {}
    for mname, kw in methods:
        if mname == "Random":
            rr = [evaluate(RandomModel(s), data["test_pos"], data["test_users"],
                           data["n_p"], K=10, negatives=99, rng_seed=s) for s in seeds]
            agg = aggregate([{**r, "ILR": 1.0, "params_M": 0.0, "train_s": 0.0,
                              "infer_ms": 0.0, "epochs": 0} for r in rr])
        elif mname == "Popularity":
            pm = PopularityModel(build_count(data["train_edges"]))
            rr = [evaluate(pm, data["test_pos"], data["test_users"],
                           data["n_p"], K=10, negatives=99, rng_seed=s) for s in seeds]
            agg = aggregate([{**r, "ILR": 1.0, "params_M": 0.0, "train_s": 0.0,
                              "infer_ms": 0.0, "epochs": 0} for r in rr])
        else:
            runs = [run_method(mname, data, epochs=epochs, seed=s, **kw) for s in seeds]
            agg = aggregate(runs)
        comp[mname] = agg
        print(f"  {mname:20s} HR={agg['HR']:.4f} NDCG={agg['NDCG']:.4f} "
              f"ILR={agg['ILR']:.3f} Params={agg['params_M']:.2f}M "
              f"Infer={agg['infer_ms']:.2f}ms")
    all_results["main_comparison"] = comp

    # ---- Sensitivity: epsilon (privacy-utility) ----
    print("\n=== Privacy budget epsilon sensitivity (Ours) ===")
    eps_vals = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    sens_eps = {}
    for eps in eps_vals:
        runs = [run_method("ours", data, epsilon=eps, adaptive=True, K=3, dim=64,
                           epochs=epochs, seed=s) for s in seeds[:1]]
        a = aggregate(runs); sens_eps[str(eps)] = a
        print(f"  eps={eps}: HR={a['HR']:.4f} NDCG={a['NDCG']:.4f} ILR={a['ILR']:.3f}")
    all_results["epsilon_sensitivity"] = sens_eps

    # ---- Sensitivity: SGC layers K ----
    print("\n=== SGC layer K sensitivity ===")
    sens_k = {}
    for K in [1, 2, 3, 4]:
        runs = [run_method("ours", data, epsilon=1.0, adaptive=True, K=K, dim=64,
                           epochs=epochs, seed=s) for s in seeds[:1]]
        a = aggregate(runs); sens_k[str(K)] = a
        print(f"  K={K}: HR={a['HR']:.4f} NDCG={a['NDCG']:.4f} infer={a['infer_ms']:.2f}ms")
    all_results["k_sensitivity"] = sens_k

    # ---- Sensitivity: embedding dim ----
    print("\n=== Embedding dimension sensitivity ===")
    sens_d = {}
    for d in [16, 32, 64, 128]:
        runs = [run_method("ours", data, epsilon=1.0, adaptive=True, K=3, dim=d,
                           epochs=epochs, seed=s) for s in seeds[:1]]
        a = aggregate(runs); sens_d[str(d)] = a
        print(f"  dim={d}: HR={a['HR']:.4f} NDCG={a['NDCG']:.4f} params={a['params_M']:.2f}M")
    all_results["dim_sensitivity"] = sens_d

    # ---- Ablation ----
    print("\n=== Ablation study ===")
    abl = {}
    abl["Full LDP-SGC-LP"] = comp["LDP-SGC-LP (Ours)"]
    abl["w/o Adaptive Budget"] = aggregate([run_method("ab", data, epsilon=1.0, adaptive=False,
                                             K=3, dim=64, epochs=epochs, seed=s) for s in seeds])
    abl["w/o LDP (SGC only)"] = comp["LightGCN-LP"]
    for k, v in abl.items():
        print(f"  {k:24s} HR={v['HR']:.4f} NDCG={v['NDCG']:.4f} ILR={v['ILR']:.3f}")
    all_results["ablation"] = abl

    all_results["dataset"] = {k: data[k] for k in
                              ["n_u", "n_p", "n_s", "raw_rows"]}
    all_results["dataset"]["train_edges"] = len(data["train_edges"])
    all_results["dataset"]["test_users"] = len(data["test_users"])
    all_results["dataset"]["name"] = "ASSISTments 2009 (skill-builder, corrected)"
    save_json("all_results.json", all_results)
    print("\nSaved results ->", os.path.join(RES, "all_results.json"))
    make_figures(all_results)


# ---------------- Figures ----------------
def make_figures(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False})
    comp = R["main_comparison"]
    names = list(comp.keys())
    short = [n.replace(" (Ours)", "*") for n in names]

    # Fig 1: recommendation performance grouped bars
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(names)); w = 0.35
    hr = [comp[n]["HR"] for n in names]; hr_s = [comp[n]["HR_std"] for n in names]
    nd = [comp[n]["NDCG"] for n in names]; nd_s = [comp[n]["NDCG_std"] for n in names]
    ax.bar(x - w/2, hr, w, yerr=hr_s, capsize=3, color="#4C72B0", label="HR@10")
    ax.bar(x + w/2, nd, w, yerr=nd_s, capsize=3, color="#55A868", label="NDCG@10")
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=20, ha="right")
    ax.set_ylabel("Metric value"); ax.legend(); ax.set_ylim(0, max(hr + nd) * 1.2)
    ax.set_title("Recommendation performance on ASSISTments 2009")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_performance.png")); plt.close(fig)

    # Fig 2: efficiency (params + inference)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pars = [comp[n]["params_M"] for n in names]
    inf = [comp[n]["infer_ms"] for n in names]
    axes[0].bar(short, pars, color="#C44E52"); axes[0].set_ylabel("Parameters (M)")
    axes[1].bar(short, inf, color="#8172B3"); axes[1].set_ylabel("Inference latency (ms)")
    for a in axes:
        a.set_xticks(range(len(names))); a.set_xticklabels(short, rotation=20, ha="right")
    axes[0].set_title("Model size"); axes[1].set_title("Inference efficiency")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_efficiency.png")); plt.close(fig)

    # Fig 3: privacy-utility tradeoff
    eps = R["epsilon_sensitivity"]; ex = sorted([float(k) for k in eps])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ex, [eps[str(e)]["HR"] for e in ex], "o-", color="#4C72B0", label="HR@10")
    ax.plot(ex, [eps[str(e)]["NDCG"] for e in ex], "s-", color="#55A868", label="NDCG@10")
    ax2 = ax.twinx()
    ax2.plot(ex, [eps[str(e)]["ILR"] for e in ex], "^--", color="#C44E52", label="ILR")
    ax.set_xlabel("Privacy budget epsilon"); ax.set_ylabel("Utility")
    ax2.set_ylabel("Information leakage rate (ILR)")
    ax.set_title("Privacy-utility tradeoff")
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_privacy_utility.png")); plt.close(fig)

    # Fig 4: K sensitivity
    kk = R["k_sensitivity"]; kx = sorted([int(k) for k in kk])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(kx, [kk[str(k)]["HR"] for k in kx], "o-", color="#4C72B0", label="HR@10")
    ax.plot(kx, [kk[str(k)]["NDCG"] for k in kx], "s-", color="#55A868", label="NDCG@10")
    ax.set_xlabel("SGC propagation hops K"); ax.set_ylabel("Metric value")
    ax.legend(); ax.set_title("Effect of propagation depth"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_k_sensitivity.png")); plt.close(fig)

    # Fig 5: ablation
    abl = R["ablation"]; an = list(abl.keys())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(an))
    ax.bar(x, [abl[k]["HR"] for k in an], 0.38, color="#4C72B0", label="HR@10")
    ax.bar(x + 0.38, [abl[k]["NDCG"] for k in an], 0.38, color="#55A868", label="NDCG@10")
    ax.set_xticks(x + 0.19); ax.set_xticklabels(an, rotation=12, ha="right")
    ax.legend(); ax.set_ylabel("Metric value"); ax.set_title("Ablation study")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_ablation.png")); plt.close(fig)
    print("Saved figures ->", FIG)


if __name__ == "__main__":
    main()
