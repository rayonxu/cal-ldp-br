# -*- coding: utf-8 -*-
"""Prepare a reproducible local sample from the official EdNet release.

EdNet is distributed as one CSV file per user.  This script samples user files
first (never the whole interaction stream), keeps question-response events,
maps string IDs to contiguous integers, and writes the common schema consumed
by ds_loader.load_ednet:

    user_id, item_id, skill_id, timestamp

Example:
  python C:\\code\\prepare_ednet.py --kt 1 \
      --ednet_dir "D:\\datasets\\ednet_kt1" --users 5000
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT = os.path.join(os.environ.get('LDP_DATA_DIR', './data'),
                           'ednet_sampled.csv')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kt", type=int, choices=[1, 2, 3, 4], default=1)
    ap.add_argument("--ednet_dir", required=True,
                    help="Folder containing extracted per-user EdNet CSV files")
    ap.add_argument("--users", type=int, default=5000,
                    help="Number of user files to sample; 0 means all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_user_inter", type=int, default=5)
    ap.add_argument("--max_rows_per_user", type=int, default=0,
                    help="Optional chronological cap per user; 0 means no cap")
    ap.add_argument("--out", default=DEFAULT_OUT)
    return ap.parse_args()


def read_one(path, kt):
    if kt == 1:
        df = pd.read_csv(path, low_memory=False)
        needed = {"timestamp", "question_id"}
        if not needed.issubset(df.columns):
            return None
        df = df.dropna(subset=["question_id"])
        df = df[df["question_id"].astype(str).str.startswith("q")]
        df["item_raw"] = df["question_id"].astype(str)
        # Two official KT1 releases exist: the older one has 'bundle_id'
        # (a public item-group / skill tag); the newer one replaces it with
        # 'solving_id' and has no taxonomy.  Use bundle_id when available,
        # otherwise fall back to the item itself as a trivial skill id.  The
        # main Cal-LDP-BR method does not consume skill_id, so this only
        # affects the (already negative) skill-graph ablation.
        if "bundle_id" in df.columns:
            df["skill_raw"] = df["bundle_id"].astype(str)
        else:
            df["skill_raw"] = df["item_raw"]
        df = df[["timestamp", "item_raw", "skill_raw"]].copy()
    else:
        df = pd.read_csv(path, low_memory=False)
        needed = {"timestamp", "action_type", "item_id"}
        if not needed.issubset(df.columns):
            return None
        df = df[df["action_type"].astype(str) == "respond"]
        df = df[["timestamp", "item_id"]].dropna().copy()
        df = df[df["item_id"].astype(str).str.startswith("q")]
        df["item_raw"] = df["item_id"].astype(str)
        # KT2-4 do not expose a public skill taxonomy; use the item itself as
        # the (trivial) skill so the loader schema remains uniform.
        df["skill_raw"] = df["item_raw"]
    df = df[["timestamp", "item_raw", "skill_raw"]]
    return df if len(df) else None


def main():
    args = parse_args()
    root = Path(args.ednet_dir)
    if not root.is_dir():
        raise SystemExit(f"EdNet directory does not exist: {root}")

    files = sorted(root.rglob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found under {root}")
    rng = np.random.default_rng(args.seed)
    if args.users > 0 and len(files) > args.users:
        chosen = rng.choice(len(files), size=args.users, replace=False)
        files = [files[int(i)] for i in chosen]
    print(f"Found {len(files)} EdNet user files to process", flush=True)

    parts = []
    for n, path in enumerate(files, 1):
        try:
            df = read_one(path, args.kt)
        except Exception as exc:
            print(f"  skip {path.name}: {exc}", flush=True)
            continue
        if df is None:
            continue
        df = df.sort_values("timestamp", kind="mergesort")
        if args.max_rows_per_user > 0:
            df = df.head(args.max_rows_per_user)
        df["user_id"] = path.stem
        parts.append(df)
        if n % 1000 == 0:
            print(f"  processed {n}/{len(files)} files", flush=True)

    if not parts:
        raise SystemExit("No usable interaction rows were found")
    all_df = pd.concat(parts, ignore_index=True)
    all_df["user_id"] = pd.factorize(all_df["user_id"], sort=True)[0] + 1
    all_df["item_id"] = pd.factorize(all_df["item_raw"], sort=True)[0] + 1
    all_df["skill_id"] = pd.factorize(all_df["skill_raw"], sort=True)[0] + 1
    cnt = all_df.groupby("user_id").size()
    keep = cnt[cnt >= args.min_user_inter].index
    all_df = all_df[all_df["user_id"].isin(keep)]
    out = all_df[["user_id", "item_id", "skill_id", "timestamp"]].copy()
    out = out.sort_values(["user_id", "timestamp"], kind="mergesort")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} rows, {out.user_id.nunique()} users, "
          f"{out.item_id.nunique()} items to {args.out}", flush=True)


if __name__ == "__main__":
    main()
