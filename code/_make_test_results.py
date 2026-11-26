# -*- coding: utf-8 -*-
"""Temporary: build a dummy results JSON in the v2 schema to test the DOCX builder."""
import json, os
RES = r"C:\code\results_v2"
os.makedirs(RES, exist_ok=True)
names = ["Popularity", "GCN-LP", "LightGCN-LP", "LDP-GCN",
         "LDP-RR-Light", "LDP-AMP", "LDP-SGC-LP (Ours)"]
hr = [0.22, 0.33, 0.42, 0.08, 0.08, 0.09, 0.10]
auc = [0.5, 1.0, 1.0, 0.78, 0.78, 0.65, 0.52]
ALL = {}
for ds, nu, np_, ns in [("assist2009", 305, 10863, 114), ("assist2015", 300, 100, 100)]:
    methods = {}
    for n, h, a in zip(names, hr, auc):
        methods[n] = dict(HR=h, HR_std=0.01, NDCG=h/2, NDCG_std=0.005,
                          HR_all=[h]*5, NDCG_all=[h/2]*5, auc=a,
                          eps=(None if "LDP" not in n else 1.0),
                          params=0.92, infer=20.0, meta={})
    methods["_significance"] = {
        b: dict(delta=0.02, t=3.5, p_t=0.01, p_w=0.02, cohen_d=1.2)
        for b in ["LDP-GCN", "LDP-RR-Light", "LDP-AMP", "GCN-LP", "LightGCN-LP"]}
    ALL[ds] = dict(dataset=dict(n_u=nu, n_p=np_, n_s=ns, raw_rows=33000,
                                train_edges=33138), methods=methods)
with open(os.path.join(RES, "all_results_v2.json"), "w", encoding="utf-8") as f:
    json.dump(ALL, f, ensure_ascii=False, indent=2)
print("dummy written")
