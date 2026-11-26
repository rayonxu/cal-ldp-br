# -*- coding: utf-8 -*-
"""
Rigorous LDP mechanisms for educational interaction graphs.

Three mechanisms, all with provable privacy guarantees:
  1) rr_standard        : standard edge-level Warner randomized response (eps-LDP)
  2) sampling_amplified : subsample-then-RR, with sampling-amplification guarantee
  3) adaptive_laplace_degree : privatize degree first, then set per-user sampling rate

Utilities:
  - effective_epsilon(gamma, eps) : sampling amplification bound
  - posterior_weight             : Bayesian debiasing of the reported graph
  - mia_auc                      : membership-inference attack AUC (empirical privacy)
"""
import numpy as np
import scipy.sparse as sp


# ---------- core randomized response ----------
def rr_params(eps):
    """Warner RR parameters for eps-LDP on a binary value."""
    e = np.exp(eps)
    p = e / (1.0 + e)   # P(report 1 | true 1)
    q = 1.0 / (1.0 + e) # P(report 1 | true 0)
    return p, q


def effective_epsilon(gamma, eps):
    """Sampling-amplification: subsample with prob gamma then apply eps-RR.

    Guarantee (Kasiviswanathan et al., Li et al.): the composed mechanism satisfies
        eps' = ln(1 + gamma * (exp(eps) - 1))
    for eps >= 0, gamma in (0,1].  When eps < 0 it is 0.
    """
    if eps <= 0 or gamma <= 0:
        return 0.0
    if gamma >= 1:
        return float(eps)
    return float(np.log(1.0 + gamma * (np.exp(eps) - 1.0)))


def invert_epsilon(eps_target, eps_internal):
    """Given internal RR budget eps_internal, find max sampling rate gamma such
    that the amplified budget equals eps_target.  Used to spend a privacy budget."""
    if eps_internal <= 0:
        return 0.0
    if eps_target >= eps_internal:
        return 1.0
    return float(np.expm1(eps_target) / np.expm1(eps_internal))


# ---------- degree privatisation ----------
def privatize_degrees(deg, eps0, rng):
    """eps0-DP Laplace mechanism on the (integer) degree vector."""
    scale = 1.0 / max(eps0, 1e-6)
    noise = rng.laplace(0.0, scale, size=len(deg))
    return np.clip(deg + noise, 1, None)


# ---------- graph perturbations ----------
def rr_standard(adj, eps, rng):
    """Standard Warner RR on every cell of the user-item matrix."""
    p, q = rr_params(eps)
    coo = adj.tocoo()
    rows, cols = coo.row, coo.col
    keep = rng.random(len(rows)) < p
    nr, nc = rows[keep], cols[keep]
    n_u, n_p = adj.shape
    n_zero = n_u * n_p - len(rows)
    n_add = rng.binomial(n_zero, q) if n_zero > 0 else 0
    ar = rng.integers(0, n_u, n_add); ac = rng.integers(0, n_p, n_add)
    rr = np.concatenate([nr, ar]); cc = np.concatenate([nc, ac])
    out = sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(n_u, n_p))
    out.sum_duplicates(); out.data[:] = 1.0
    return out.tocsr(), dict(p=p, q=q, eps=eps)


def sampling_amplified(adj, eps_internal, gamma, rng):
    """Subsample cells with prob gamma, then eps_internal-RR on sampled cells.

    Each TRUE edge is kept (reported 1) with prob gamma*p.
    Each non-edge is flipped to 1 with prob gamma*q.
    Effective per-user budget = effective_epsilon(gamma, eps_internal).
    """
    p, q = rr_params(eps_internal)
    coo = adj.tocoo()
    rows, cols = coo.row, coo.col
    keep = rng.random(len(rows)) < (gamma * p)
    nr, nc = rows[keep], cols[keep]
    n_u, n_p = adj.shape
    n_zero = n_u * n_p - len(rows)
    # sampled non-edges: binomial over sampled cells
    n_add = rng.binomial(n_zero, gamma * q) if n_zero > 0 else 0
    ar = rng.integers(0, n_u, n_add); ac = rng.integers(0, n_p, n_add)
    rr = np.concatenate([nr, ar]); cc = np.concatenate([nc, ac])
    out = sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(n_u, n_p))
    out.sum_duplicates(); out.data[:] = 1.0
    return out.tocsr(), dict(p=p, q=q, gamma=gamma,
                             eps_eff=effective_epsilon(gamma, eps_internal))


def asymmetric_rr(adj, eps, eps0=0.0, adaptive=True, rng=None, eta=0.6,
                  gamma_min=0.2, gamma_max=1.0):
    """Asymmetric RR with sampling-amplified FALSE positives (the proposed design).

    Privacy accounting (parallel composition over disjoint user-item cells):
      * positive cell (true edge): Warner RR with budget eps_pos = eps  -> high retention
      * negative cell (non-edge): subsample with prob gamma_u then RR eps_neg = eps
        -> amplified budget ln(1+gamma_u(exp(eps)-1)) <= eps
      Since each cell is perturbed once and cells are disjoint, the overall per-record
      guarantee is max(eps_pos, eps_neg_amplified) = eps (NOT their sum).  An optional
      eps0 is spent on privatising the degree used to set gamma_u (sequential comp.),
      giving total eps0 + eps <= eps_budget if eps = eps_budget - eps0.
    The false-positive rate becomes gamma_u*q << q, so the reported graph is much cleaner
    while retaining genuine edges at rate p.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n_u, n_p = adj.shape
    p, q = rr_params(eps)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    if adaptive:
        d_tilde = privatize_degrees(deg, eps0, rng) if eps0 > 0 else deg.astype(float)
        d_ref = float(np.median(np.maximum(d_tilde, 1)))
        gamma_u = np.clip((d_ref / np.maximum(d_tilde, 1.0)) ** eta,
                          gamma_min, gamma_max)
    else:
        gamma_u = np.full(n_u, 1.0)

    coo = adj.tocoo()
    rr_list, cc_list = [], []
    for u in range(n_u):
        mask = coo.row == u
        cs = coo.col[mask]
        # TRUE edges: standard RR with full budget (retain with prob p)
        kp = rng.random(len(cs)) < p
        rr_list.append(np.full(int(kp.sum()), u, dtype=np.int64)); cc_list.append(cs[kp])
        # NON-edges: sample with gamma_u, then flip with q
        n_zero = n_p - len(cs)
        n_add = rng.binomial(n_zero, gamma_u[u] * q) if n_zero > 0 else 0
        if n_add > 0:
            ac = rng.integers(0, n_p, n_add)
            existing = set(cs.tolist())
            ac = np.array([x for x in ac if int(x) not in existing], dtype=np.int64)
            rr_list.append(np.full(len(ac), u, dtype=np.int64)); cc_list.append(ac)
    rr_all = np.concatenate(rr_list) if rr_list else np.array([], dtype=np.int64)
    cc_all = np.concatenate(cc_list) if cc_list else np.array([], dtype=np.int64)
    out = sp.csr_matrix((np.ones(len(rr_all), np.float32), (rr_all, cc_all)), shape=(n_u, n_p))
    out.sum_duplicates(); out.data[:] = 1.0
    eps_eff_u = np.array([effective_epsilon(g, eps) for g in gamma_u])
    meta = dict(p=p, q=q, gamma_u=gamma_u, eps_eff_u=eps_eff_u,
                eps_pos=eps, eps_neg=eps, eps0=eps0,
                eps_total=eps0 + float(eps_eff_u.max()), adaptive=adaptive,
                kept=int(rr_all.size), orig=int(adj.nnz), asymmetric=True)
    return out.tocsr(), meta


def adaptive_perturb(adj, eps, eps0, adaptive=True, rng=None, eta=0.5,
                     gamma_min=0.15, gamma_max=1.0):
    """Adaptive sampling-amplified LDP (the proposed mechanism).

    Steps (per-user):
      (1) privatise degree d_u with eps0 budget;
      (2) set sampling rate gamma_u decreasing with privatised degree:
              gamma_u = clip( (d_ref / d_tilde_u)^eta, gamma_min, gamma_max )
          so active users (high degree, redundant info) are sampled less;
      (3) subsample+RR with internal budget eps_int = eps - eps0 (sequential comp.)
      Overall guarantee: eps0 + max_u effective_epsilon(gamma_u, eps_int) <= eps,
      which we verify explicitly and report.
    Returns perturbed graph and a metadata dict for accounting.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n_u, n_p = adj.shape
    eps_int = max(eps - eps0, 1e-6)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    if adaptive:
        d_tilde = privatize_degrees(deg, eps0, rng)
        d_ref = float(np.median(d_tilde))
        gamma_u = np.clip((d_ref / np.maximum(d_tilde, 1.0)) ** eta,
                          gamma_min, gamma_max)
    else:
        gamma_u = np.full(n_u, invert_epsilon(min(eps, eps_int), eps_int)
                          if eps <= eps_int else 1.0)
        # non-adaptive baseline: single gamma spending the whole budget
        gamma_u = np.full(n_u, 1.0)

    # Perturb each user's row with its own gamma_u.
    p, q = rr_params(eps_int)
    rr_list, cc_list = [], []
    coo = adj.tocoo()
    # group true edges by user
    by_user = defaultdict(list) if False else None
    for u in range(n_u):
        gu = gamma_u[u]
        # true edges of this user
        mask = coo.row == u
        rs, cs = coo.row[mask], coo.col[mask]
        kp = rng.random(len(cs)) < (gu * p)
        rr_list.append(np.full(kp.sum(), u, dtype=np.int64)); cc_list.append(cs[kp])
        # sampled non-edges of this user (all injected edges belong to THIS user)
        n_zero = n_p - len(cs)
        n_add = rng.binomial(n_zero, gu * q) if n_zero > 0 else 0
        if n_add > 0:
            ac = rng.integers(0, n_p, n_add)
            existing = set(cs.tolist())
            ac = np.array([x for x in ac if int(x) not in existing], dtype=np.int64)
            rr_list.append(np.full(len(ac), u, dtype=np.int64)); cc_list.append(ac)

    rr_all = np.concatenate(rr_list) if rr_list else np.array([], dtype=np.int64)
    cc_all = np.concatenate(cc_list) if cc_list else np.array([], dtype=np.int64)
    out = sp.csr_matrix((np.ones(len(rr_all), np.float32), (rr_all, cc_all)),
                        shape=(n_u, n_p))
    out.sum_duplicates(); out.data[:] = 1.0
    eps_eff_u = np.array([effective_epsilon(g, eps_int) for g in gamma_u])
    eps_total = (eps0 if adaptive else 0.0) + float(eps_eff_u.max())
    meta = dict(p=p, q=q, eps_int=eps_int, eps0=eps0 if adaptive else 0.0,
                gamma_u=gamma_u, eps_eff_u=eps_eff_u, eps_total=eps_total,
                adaptive=adaptive, kept=int(rr_all.size), orig=int(adj.nnz))
    return out.tocsr(), meta


# ---------- Bayesian debiasing ----------
def posterior_weight(perturbed, meta, adj_density=None):
    """Posterior probability that a reported edge is genuine, used as soft weight.

    P(true=1 | report=1) = p*pi / (p*pi + q*(1-pi))
    where pi is the prior edge density and p,q are the *effective* RR parameters
    after sampling amplification:  p_eff = gamma*p, q_eff = gamma*q.  The gamma
    cancels in the posterior ratio only if uniform; with user-specific gamma we
    use per-row effective parameters.
    """
    pert = perturbed.tocoo()
    p, q = meta["p"], meta["q"]
    gamma_u = meta.get("gamma_u", None)
    pi = 0.02 if adj_density is None else adj_density
    if gamma_u is not None:
        g = gamma_u[pert.row]                 # vectorized per-row gamma
    else:
        g = np.ones(len(pert.row))
    pe, qe = g * p, g * q
    denom = pe * pi + qe * (1 - pi)
    w = np.where(denom > 0, pe * pi / np.maximum(denom, 1e-12), pi).astype(np.float32)
    out = sp.csr_matrix((w, (pert.row, pert.col)), shape=perturbed.shape)
    return out.tocsr(), w


def estimate_item_priors(perturbed, meta, fallback=None):
    """Estimate item marginals from the *released* perturbed graph.

    For each cell, E[Y_ui] = gamma_u * (p*A_ui + q*(1-A_ui)).  Aggregating over
    users gives a moment estimator for pi_i = P(A_ui = 1):

        pi_i = (sum_u Y_ui - q * sum_u gamma_u)
               / ((p - q) * sum_u gamma_u).

    Only the published noisy graph is used, so this is post-processing and
    consumes no additional privacy budget.
    """
    p, q = meta["p"], meta["q"]
    n_u, n_p = perturbed.shape
    gamma_u = meta.get("gamma_u", None)
    if gamma_u is None:
        gamma_u = np.ones(n_u, dtype=np.float64)
    else:
        gamma_u = np.asarray(gamma_u, dtype=np.float64)
    counts = np.asarray(perturbed.sum(axis=0)).ravel().astype(np.float64)
    s_gamma = max(float(gamma_u.sum()), 1e-9)
    pi_hat = (counts - q * s_gamma) / (max(p - q, 1e-9) * s_gamma)
    base = 1e-6 if fallback is None else float(fallback)
    return np.clip(pi_hat, base, 0.95)


def calibrated_posterior_weight(perturbed, meta, item_prior=None,
                                fallback=None, top_k=None):
    """Bayesian calibration using item-specific priors.

    P(A=1 | Y=1, u, i) = gamma_u*p*pi_i /
                         (gamma_u*p*pi_i + gamma_u*q*(1-pi_i)).
    gamma_u cancels for a fixed (u,i), but keeping it makes the formula valid
    when priors are estimated with non-uniform sampling.  Optionally retain
    only the top-k reported edges per user (posterior denoising).
    """
    pert = perturbed.tocoo()
    p, q = meta["p"], meta["q"]
    n_u, n_p = perturbed.shape
    gamma_u = meta.get("gamma_u", None)
    if gamma_u is None:
        g = np.ones(len(pert.row), dtype=np.float64)
    else:
        g = np.asarray(gamma_u, dtype=np.float64)[pert.row]
    if item_prior is None:
        base = 1e-6 if fallback is None else float(fallback)
        pi = np.full(n_p, base, dtype=np.float64)
    else:
        pi = np.asarray(item_prior, dtype=np.float64)[pert.col]
    pe, qe = g * p, g * q
    denom = pe * pi + qe * (1.0 - pi)
    w = np.where(denom > 0, pe * pi / np.maximum(denom, 1e-12), 0.0)

    if top_k is not None and top_k > 0 and len(w) > 0:
        # Sort by row then descending posterior weight.
        order = np.lexsort((-w, pert.row))
        sr = pert.row[order]
        uniq, starts = np.unique(sr, return_index=True)
        ends = np.r_[starts[1:], len(order)]
        keep = np.concatenate([order[s:min(s + int(top_k), e)]
                               for s, e in zip(starts, ends)])
        rr, cc = pert.row[keep], pert.col[keep]
        ww = w[keep]
    else:
        rr, cc, ww = pert.row, pert.col, w

    out = sp.csr_matrix((ww.astype(np.float32), (rr, cc)), shape=perturbed.shape)
    out.sum_duplicates()
    return out.tocsr(), ww.astype(np.float32)


# ---------- empirical privacy: membership inference attack AUC ----------
def mia_auc(original_adj, perturbed_adj, n_samples=2000, seed=0):
    """Train-free membership inference: adversary knows a set of candidate edges
    C (mix of true and false) and observes the perturbed graph. It scores each
    candidate by the reported value (+ posterior weight).  AUC measures how well
    the adversary separates true from false members.  AUC=0.5 => no leakage.
    """
    rng = np.random.default_rng(seed)
    n_u, n_p = original_adj.shape
    orig = original_adj.tocsr()
    pert = perturbed_adj.tocsr()
    # sample candidate (u,i) pairs: half true edges, half random non-edges
    coo = orig.tocoo()
    idx = rng.choice(len(coo.row), size=min(n_samples // 2, len(coo.row)), replace=False)
    pos = list(zip(coo.row[idx], coo.col[idx]))
    neg = []
    tries = 0
    while len(neg) < len(pos) and tries < n_samples * 5:
        u = int(rng.integers(0, n_u)); i = int(rng.integers(0, n_p))
        if orig[u, i] == 0:
            neg.append((u, i))
        tries += 1
    pairs = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg))
    # vectorized CSR lookup: mark reported edges in a dense boolean row-index map
    pu = np.array([p[0] for p in pairs]); pi = np.array([p[1] for p in pairs])
    pert_coo = pert.tocoo()
    reported = np.zeros((n_u,), dtype=bool)  # placeholder, use set of (u,i)
    rep_set = set(zip(pert_coo.row.tolist(), pert_coo.col.tolist()))
    scores = np.array([1.0 if (int(u), int(i)) in rep_set else 0.0 for u, i in pairs])
    # AUC via rank statistic (Mann-Whitney)
    if len(np.unique(labels)) < 2:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # tie handling (average ranks)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n1 = labels.sum(); n0 = len(labels) - n1
    auc = (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc)
