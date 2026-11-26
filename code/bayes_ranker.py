# -*- coding: utf-8 -*-
"""Bayesian posterior ranker for released LDP interaction graphs.

The ranker is a pure post-processing step: it uses only the locally perturbed
graph published by the clients and public item marginals estimated from that
graph. It ranks a candidate item by the log posterior odds that the underlying
interaction equals one.
"""
import numpy as np
import scipy.sparse as sp


class PosteriorRanker:
    def __init__(self, R, meta, item_prior, fallback=1e-4):
        self.R = R.tocsr()
        self.meta = meta
        self.prior = np.asarray(item_prior, dtype=np.float64)
        self.prior = np.clip(self.prior, fallback, 0.95)
        n_u, n_p = R.shape
        gamma = meta.get("gamma_u", None)
        self.gamma = (np.ones(n_u, dtype=np.float64) if gamma is None
                      else np.asarray(gamma, dtype=np.float64))
        self.p = float(meta["p"])
        self.q = float(meta["q"])
        self.asymmetric = bool(meta.get("asymmetric", False))
        self.eps0 = float(meta.get("eps0", 0.0))

    def score(self, users, items):
        u = int(np.asarray(users).ravel()[0])
        items = np.asarray(items, dtype=np.int64)
        y = self.R.getrow(u).toarray().ravel()[items].astype(bool)
        pi = self.prior[items]
        g = self.gamma[u]
        odds = np.empty_like(pi)

        if self.asymmetric:
            # True edge reported with p; false edge reported with gamma*q.
            pos_rate = self.p
            neg_rate = np.clip(g * self.q, 1e-9, 1 - 1e-9)
        else:
            # Sampling amplification applies to both true and false cells;
            # gamma cancels in the positive likelihood ratio.
            pos_rate = np.clip(g * self.p, 1e-9, 1 - 1e-9)
            neg_rate = np.clip(g * self.q, 1e-9, 1 - 1e-9)

        pos = np.log(pos_rate * pi) - np.log(neg_rate * (1 - pi) + 1e-300)
        neg = np.log((1 - pos_rate) * pi + 1e-300) - \
              np.log((1 - neg_rate) * (1 - pi) + 1e-300)
        odds[:] = np.where(y, pos, neg)
        return odds


class HybridRanker:
    """Convex combination of posterior ranking and an SGC model score."""

    def __init__(self, ranker, sgc_model, alpha=0.5):
        self.ranker = ranker
        self.sgc = sgc_model
        self.alpha = float(alpha)

    def score(self, users, items, Z=None):
        r = self.ranker.score(users, items)
        if Z is None:
            Z = getattr(self, "Z", None)
        s = np.asarray(self.sgc.score(users, items, Z=Z), dtype=np.float64)
        # Robust scale matching on the candidate batch.
        rs = r.std() if r.std() > 1e-9 else 1.0
        ss = s.std() if s.std() > 1e-9 else 1.0
        return self.alpha * r / rs + (1 - self.alpha) * s / ss
