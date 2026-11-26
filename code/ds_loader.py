# -*- coding: utf-8 -*-
"""
Unified loader for REAL public educational interaction datasets.
Each loader returns a dict with the SAME schema so experiments are dataset-agnostic:
  n_u, n_p, n_s, train_edges[(u,p)], test_pos{u:p}, test_users,
  user_train_items{u:set}, skill_to_problems{s:[p]}, raw_rows, name
"""
import os
from collections import defaultdict
import numpy as np

DATA_DIR = os.environ.get('LDP_DATA_DIR', './data')


def _sample_users(df, user_col, max_users, seed=42):
    # max_users <= 0 means "use all users" (CLI convention).
    if max_users is None or max_users <= 0:
        return df
    ids = np.unique(df[user_col].values)
    if len(ids) <= max_users:
        return df
    rng = np.random.default_rng(seed)
    sel = rng.choice(ids, size=max_users, replace=False)
    return df[df[user_col].isin(sel)].copy()


def _finalize(df, u_col, p_col, s_col, order_col, name, min_user_inter):
    """Common tail: recode ids, leave-last-out split, build skill map."""
    df = df.dropna(subset=[p_col, s_col]).copy()
    df = df[(df[u_col] > 0)]
    cnt = df.groupby(u_col).size()
    keep = cnt[cnt >= min_user_inter].index
    df = df[df[u_col].isin(keep)].copy()
    df = df.sort_values([u_col, order_col], kind="mergesort").reset_index(drop=True)

    uvals = np.sort(df[u_col].unique()); pvals = np.sort(df[p_col].unique())
    svals = np.sort(df[s_col].unique())
    umap = {int(x): i for i, x in enumerate(uvals)}
    pmap = {int(x): i for i, x in enumerate(pvals)}
    smap = {int(x): i for i, x in enumerate(svals)}
    df["u"] = df[u_col].map(umap).astype(int)
    df["p"] = df[p_col].map(pmap).astype(int)
    df["s"] = df[s_col].map(smap).astype(int)
    n_u, n_p, n_s = len(umap), len(pmap), len(smap)

    skill_to_problems = defaultdict(set)
    for p, s in df[["p", "s"]].drop_duplicates().values:
        skill_to_problems[int(s)].add(int(p))
    skill_to_problems = {k: sorted(v) for k, v in skill_to_problems.items()}

    train_edges, test_pos, user_train_items = [], {}, defaultdict(set)
    for u, grp in df.groupby("u", sort=False):
        items = grp["p"].tolist()
        if len(items) < 2:
            continue
        target = int(items[-1])
        test_pos[int(u)] = target
        # Strict leave-last-out: the held-out item must not remain in this
        # user's training history through an earlier repeated attempt.
        for it in items[:-1]:
            if int(it) == target:
                continue
            train_edges.append((int(u), int(it)))
            user_train_items[int(u)].add(int(it))
    test_users = sorted(test_pos.keys())
    return dict(n_u=n_u, n_p=n_p, n_s=n_s, train_edges=train_edges,
                test_pos=test_pos, test_users=test_users,
                user_train_items=user_train_items, skill_to_problems=skill_to_problems,
                raw_rows=len(df), name=name)


def load_assist2009(min_user_inter=10, max_users=700, seed=42):
    import pandas as pd
    path = os.path.join(DATA_DIR, "assist2009_skill_builder_data_corrected.csv")
    df = pd.read_csv(path, usecols=["user_id", "problem_id", "skill_id", "order_id"],
                     low_memory=False)
    # A multi-skill row may list multiple space-separated skill ids; take the first.
    df["skill_id"] = df["skill_id"].astype(str).str.split().str[0]
    df["skill_id"] = pd.to_numeric(df["skill_id"], errors="coerce")
    df = df[(df["problem_id"] > 0)]
    df = _sample_users(df, "user_id", max_users, seed)
    return _finalize(df, "user_id", "problem_id", "skill_id", "order_id",
                     "ASSISTments 2009", min_user_inter)


def load_assist2015(min_user_inter=10, max_users=700, seed=42):
    import pandas as pd
    path = os.path.join(DATA_DIR, "assist2015.csv")
    df = pd.read_csv(path, low_memory=False)
    # 2015 has user_id, log_id (time), sequence_id (== skill builder, 100 of them).
    # Treat sequence_id as the exercise/item and as its own skill (1:1 in this release).
    df = df[["user_id", "log_id", "sequence_id"]].copy()
    df = df[(df["user_id"] > 0) & (df["sequence_id"] > 0)]
    df = _sample_users(df, "user_id", max_users, seed)
    d = _finalize(df, "user_id", "sequence_id", "sequence_id", "log_id",
                  "ASSISTments 2015", min_user_inter)
    # each skill maps to exactly its single item
    d["skill_to_problems"] = {s: [p] for s, p in
                              df[["sequence_id", "sequence_id"]].drop_duplicates().values}
    return d


def load_assist2012(min_user_inter=10, max_users=700, seed=42):
    """Third dataset: ASSISTments 2012 (tab-separated, preprocessed release)."""
    import pandas as pd
    path = os.path.join(DATA_DIR, "assist2012.csv")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # Columns: user_id, item_id, timestamp, correct, skill_id
    df = df[["user_id", "item_id", "timestamp", "skill_id"]].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = _sample_users(df, "user_id", max_users, seed)
    return _finalize(df, "user_id", "item_id", "skill_id", "timestamp",
                     "ASSISTments 2012", min_user_inter)


def load_ednet(min_user_inter=10, max_users=0, seed=42):
    """Load the locally prepared EdNet sample produced by prepare_ednet.py."""
    import pandas as pd
    path = os.path.join(DATA_DIR, "ednet_sampled.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run prepare_ednet.py first, or put your "
            "EdNet sample at this path with columns "
            "user_id,item_id,skill_id,timestamp.")
    df = pd.read_csv(path, low_memory=False)
    for c in ["user_id", "item_id", "skill_id", "timestamp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "skill_id", "timestamp"])
    df = df[(df["user_id"] > 0) & (df["item_id"] > 0)]
    df = _sample_users(df, "user_id", max_users, seed)
    return _finalize(df, "user_id", "item_id", "skill_id", "timestamp",
                     "EdNet", min_user_inter)


LOADERS = {
    "assist2009": load_assist2009,
    "assist2015": load_assist2015,
    "assist2012": load_assist2012,
    "ednet": load_ednet,
}


def get(name, **kw):
    return LOADERS[name](**kw)
