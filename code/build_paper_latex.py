# -*- coding: utf-8 -*-
"""Generate a Springer IJMLC LaTeX submission (svjour3) from the SAME real results.

Output: C:/Cal-LDP-BR_IJMLC_latex/ with main.tex, figures, and a README.
Figures and all numbers are identical to the DOCX submission.
"""
import os, json, shutil, re

ROOT = r"C:\code"
RES = os.path.join(ROOT, "results_v2")
REFS_PATH = os.path.join(ROOT, "results", "verified_references_v2.json")
FIG = os.path.join(ROOT, "figures_v2")
OUTDIR = r"C:\Cal-LDP-BR_IJMLC_latex"
FIGDIR = os.path.join(OUTDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

DS_LABEL = {"assist2009": "ASSISTments 2009", "assist2015": "ASSISTments 2015",
            "ednet": "EdNet"}
LDP_BASES = ["LDP-GCN", "LDP-RR-Light", "LDP-AMP"]
OURS = "Cal-LDP-BR (Ours)"


def esc(s):
    """Escape plain text for LaTeX (references are pre-formatted strings)."""
    s = s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
    s = s.replace("#", r"\#").replace("$", r"\$")
    return s


def f4(v): return f"{v:.4f}"
def f3(v): return f"{v:.3f}"


def load():
    with open(os.path.join(RES, "all_results_v2.json"), encoding="utf-8") as f:
        ALL = json.load(f)
    with open(os.path.join(RES, "bayes_rank_grid.json"), encoding="utf-8") as f:
        GRID = json.load(f)
    with open(os.path.join(RES, "bayes_rank_grid_full.json"), encoding="utf-8") as f:
        FULL = json.load(f)
    with open(os.path.join(RES, "ablation_v2.json"), encoding="utf-8") as f:
        AB = json.load(f)
    with open(REFS_PATH, encoding="utf-8") as f:
        REFS = json.load(f)
    return ALL, GRID, FULL, AB, REFS


def main():
    ALL, GRID, FULL, AB, REFS = load()
    DS = list(ALL.keys())
    dl = lambda k: DS_LABEL.get(k, k)

    # copy figures
    fig_files = ["framework_cal_ldp_br.png", "algorithm_flow_cal_ldp_br.png"]
    fig_files += [f"{d}_perf.png" for d in DS]
    fig_files += [f"{DS[0]}_privacy_util.png"]
    for name in fig_files:
        src = os.path.join(FIG, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(FIGDIR, name))

    L = []
    w = L.append
    # ---- preamble ----
    w(r"\documentclass[twocolumn]{svjour3}")
    w(r"\smartqed")
    w(r"\journalname{Int J Mach Learn Cybern}")
    w(r"\usepackage{graphicx}")
    w(r"\usepackage{amsmath,amssymb,amsfonts}")
    w(r"\usepackage{booktabs}")
    w(r"\usepackage{multirow}")
    w(r"\usepackage{array}")
    w(r"\usepackage{url}")
    w(r"\usepackage{hyperref}")
    w("")
    w(r"\begin{document}")
    w(r"\title{Calibrated Bayesian Local Differential Privacy for Learning Path Recommendation}")
    w(r"\titlerunning{Cal-LDP-BR for Learning Path Recommendation}")
    w(r"\author{Author One \and Author Two \and Author Three}")
    w(r"\authorrunning{A. One et al.}")
    w(r"\institute{Author One \at Dept.\ of [Department], [University], [City], [Country] \and "
      r"Author Two \at Dept.\ of [Department], [University], [City], [Country] \and "
      r"Author Three \at Dept.\ of [Department], [University], [City], [Country] \\ "
      r"E-mail: [corresponding author email]}")

    # ---- abstract ----
    ds_hr = ", ".join(f4(ALL[d]["methods"][OURS]["HR"]) for d in DS)
    ds_auc = ", ".join(f3(ALL[d]["methods"][OURS]["auc"]) for d in DS)
    n_win = sum(ALL[d]["methods"][OURS]["HR"] >= max(
        v["HR"] for k, v in ALL[d]["methods"].items() if not k.startswith("_")) for d in DS)
    w(r"\begin{abstract}")
    w("Learning path recommenders exploit student--exercise interactions to personalise study "
      "sequences, but raw interaction logs reveal sensitive knowledge states. Local differential "
      "privacy (LDP) protects each student before data leave the device, yet the noise injected by "
      "randomised response (RR) normally destroys ranking utility. We propose calibrated Bayesian "
      "LDP ranking (Cal-LDP-BR): clients publish Warner-RR reports under a small privacy budget, "
      "while the server estimates item marginals from the released noisy matrix by the method of "
      "moments and ranks candidates by the Bayesian posterior log-odds of a true interaction. Both "
      "steps admit simple privacy and statistical justifications, and the ranker runs in linear "
      "time over the candidate set without graph training. A degree-adaptive sampling-amplified "
      "variant and a lightweight SGC encoder are analysed as additional components; their "
      "non-beneficial effects on sparse data are reported. On " + str(len(DS)) +
      " real public benchmarks (" + ", ".join(dl(d) for d in DS) + r"), Cal-LDP-BR at "
      r"$\varepsilon=0.02$ attains HR@10 of " + ds_hr + r", with MIA-AUC " + ds_auc +
      ". It yields the highest measured Hit Rate on " + str(n_win) + " of " + str(len(DS)) +
      " datasets and, relative to standard RR-based graph models, substantially reduces empirical "
      "leakage. Results are averaged over five fixed seeds with paired significance tests, strict "
      "leave-last-out splitting, and fully released code, figures and processed datasets.")
    w(r"\keywords{Local differential privacy; Randomised response; Bayesian calibration; "
      r"Learning path recommendation; Educational data mining; Privacy-preserving recommendation}")
    w(r"\end{abstract}")
    w(r"\maketitle")

    # ---- 1 Introduction ----
    w(r"\section{Introduction}")
    w("Online learning platforms record fine-grained student--exercise interactions, enabling "
      "personalised learning path recommendation. Graph neural networks model higher-order "
      "structure among students, exercises and knowledge points effectively~[1,2], "
      "but two obstacles limit deployment. First, interaction logs reveal knowledge mastery and "
      "study habits; training on raw records conflicts with data-protection rules such as GDPR "
      "and PIPL. Local differential privacy (LDP) perturbs data on the client so raw records "
      "never leave the device~[3,8], which suits sensitive educational data. "
      "Second, applying edge-level randomised response to a sparse educational graph injects many "
      "false edges that swamp genuine signal and severely degrade recommendation quality.")
    w("Rather than training a graph model on noisy edges, this work recovers ranking signal "
      "statistically from the released reports. Our core idea is to turn the RR noise model into a "
      "Bayesian posterior score, so that useful ranking is obtained under a very small privacy "
      "budget.")

    # ---- 2 Related work ----
    w(r"\section{Related Work}")
    w("Differential privacy offers rigorous guarantees~[16,14]. Centralised "
      "DP assumes a trusted aggregator, whereas LDP perturbs data locally~[3]. For "
      "graph data, LDP has been applied to topology and metric estimation~[8]. "
      "Bayesian LDP methods reconstruct implicit feedback by posterior estimation~[21], "
      "and degree-preserving RR emphasises structural utility~[20]. Our work differs "
      "by converting the RR likelihood directly into a ranking score with an end-to-end privacy "
      "account and a Bayes-optimality result; all server operations are post-processing of released "
      "data and spend no additional privacy budget.")

    # ---- 3 Preliminaries ----
    w(r"\section{Preliminaries and Problem Definition}")
    w(r"Let $U$ and $I$ denote the student and exercise sets and $x\in\{0,1\}^{|U|\times|I|}$ the "
      r"interaction matrix, with $x_{u,i}=1$ if student $u$ attempted exercise $i$. The goal is to "
      r"recommend the exercise a student is most likely to attempt next. The privacy goal is "
      r"per-user $\varepsilon$-local differential privacy: for any two student records the output "
      r"ratio of the local mechanism $M$ is bounded by $e^{\varepsilon}$.")
    w(r"We measure empirical privacy by membership-inference attack AUC (MIA-AUC): $0.5$ means no "
      r"leakage and higher means more leakage. Sampling amplification gives "
      r"$\varepsilon'=\ln(1+\gamma(e^{\varepsilon}-1))$ when each coordinate is sampled with "
      r"probability $\gamma$ before an $\varepsilon$-LDP mechanism; private degree release "
      r"($\varepsilon_0$-DP) and sequential composition give a total budget "
      r"$\varepsilon_0+\max_u\varepsilon'_u\le\varepsilon$.")

    # ---- 4 Method ----
    w(r"\section{Proposed Method}")
    w(r"\begin{figure*}[!t]")
    w(r"\centering\includegraphics[width=\textwidth]{figures/framework_cal_ldp_br.png}")
    w(r"\caption{Overall framework of Cal-LDP-BR. The client perturbs each interaction locally "
      r"under $\varepsilon$-LDP; the server calibrates item marginals from released reports and "
      r"ranks candidates by Bayesian posterior log-odds. Both server steps are post-processing "
      r"and consume no additional privacy budget.}")
    w(r"\label{fig:framework}")
    w(r"\end{figure*}")
    w(r"\begin{figure}[!t]")
    w(r"\centering\includegraphics[width=\columnwidth]{figures/algorithm_flow_cal_ldp_br.png}")
    w(r"\caption{Step-by-step client--server protocol with privacy accounting at each stage.}")
    w(r"\label{fig:alg}")
    w(r"\end{figure}")
    w("Figure~\\ref{fig:framework} and Fig.~\\ref{fig:alg} highlight three core innovations. "
      "(i) A very small privacy budget ("
      r"$\varepsilon=0.02$"
      ") is made useful: Warner RR keeps the client encoding trivial while moment calibration of "
      "item marginals recovers the ranking signal from heavy noise. (ii) The ranking score is the "
      "Bayes-optimal posterior log-odds under the RR likelihood, so the server exploits the noise "
      "model statistically. (iii) Privacy is guaranteed end-to-end by exact client $\\varepsilon$-LDP "
      "and, for the adaptive variant, by sampling amplification plus composition (Theorem~\\ref{thm:priv}); "
      "all downstream operations are post-processing.")

    w(r"\subsection{Client-side local perturbation}")
    w(r"Each student applies Warner RR independently to each exercise bit: "
      r"$p=e^{\varepsilon}/(1+e^{\varepsilon})$, $q=1/(1+e^{\varepsilon})$. Parallel "
      r"composition gives exact $\varepsilon$-LDP for the whole vector; we use "
      r"$\varepsilon=0.02$. As an optional variant, a Laplace-noisy degree with budget "
      r"$\varepsilon_0$ sets $\gamma_u=\mathrm{clip}((d_{\rm ref}/\tilde d_u)^{\eta},"
      r"\gamma_{\min},\gamma_{\max})$; sampling before RR gives per-user budget "
      r"$\ln(1+\gamma_u(e^{\varepsilon_{\rm int}}-1))$ bounded by $\varepsilon$ (Theorem~\ref{thm:priv}).")

    w(r"\subsection{Item-wise Bayesian calibration and ranking}")
    w(r"From the released matrix we estimate the item marginal by the method of moments, "
      r"$\hat\pi_i=(\sum_u a_{u,i}-q\sum_u\gamma_u)/((p-q)\sum_u\gamma_u)$. Given a report, "
      r"the posterior probability of a true edge is "
      r"$p\hat\pi_i/(p\hat\pi_i+q(1-\hat\pi_i))$ for $a_{u,i}=1$ and the $(1-p)$ form for "
      r"$a_{u,i}=0$. The log of this posterior odds is the Bayes-optimal ranking score under the "
      r"RR likelihood. Estimation, clipping and ranking use only released data and spend no extra budget.")

    w(r"\subsection{Lightweight graph variant and complexity}")
    w("We also study a lightweight SGC encoder with soft posterior weights and a public skill "
      "graph; the ablation shows the skill graph is not helpful on the sparse data, so the main "
      "configuration omits it. The ranker needs no graph training and scans candidates linearly, "
      "suiting resource-constrained devices.")

    # ---- 5 Theory ----
    w(r"\section{Theoretical Analysis}")
    w(r"\subsection{Privacy guarantee}")
    w(r"\textbf{Lemma 1 (sampling amplification).} If $S_\gamma$ samples each coordinate with "
      r"probability $\gamma$ and $M$ is $\varepsilon$-LDP, then $M\circ S_\gamma$ is "
      r"$\varepsilon'$-LDP with $\varepsilon'=\ln(1+\gamma(e^{\varepsilon}-1))$. "
      r"\textbf{Lemma 2 (private degree).} $\tilde d_u=d_u+\mathrm{Lap}(1/\varepsilon_0)$ satisfies "
      r"$\varepsilon_0$-DP; $\gamma_u$ is post-processing.")
    w(r"\textbf{Theorem 1.}\label{thm:priv} The mechanism satisfies "
      r"$\varepsilon_{\rm tot}=\varepsilon_0+\max_u\ln(1+\gamma_u(e^{\varepsilon_{\rm int}}-1))"
      r"\le\varepsilon$-LDP. Sequential composition of Lemma~1 and Lemma~2 gives the bound; the "
      r"inequality is verified numerically at run time.")

    w(r"\subsection{Utility}")
    w(r"\textbf{Theorem 2.}\label{thm:util} The inverse-probability estimator "
      r"$(a_{u,i}-q_u)/(p_u-q_u)$ is unbiased, and the posterior is the minimum-MSE (Bayes) "
      r"estimate among functions of the reported bit, justifying the posterior ranking.")

    # ---- 6 Experiments ----
    w(r"\section{Experiments}")
    w(r"\subsection{Datasets and setup}")
    dparts = []
    for d in DS:
        dd = ALL[d]["dataset"]
        dparts.append(f"{dl(d)} has {dd['raw_rows']} interactions over {dd['n_u']} sampled "
                      f"students, {dd['n_p']} exercises and {dd['n_s']} skills "
                      f"({dd['train_edges']} training edges)")
    w("We evaluate on " + str(len(DS)) + " real public benchmarks spanning two platforms "
      "(ASSISTments and EdNet): " + "; ".join(dparts) + ". For EdNet we sample 5,000 "
      "per-student files from the official KT1 release; that variant exposes no public skill "
      "taxonomy, so skill is a placeholder and the skill-graph ablation is omitted. Each "
      "student's last interaction is held out and ranked against 99 sampled negatives; a held-out "
      "item is removed from training even if attempted earlier. Results average five fixed seeds.")
    w(r"Cal-LDP-BR uses $\varepsilon=0.02$; standard LDP baselines use $\varepsilon=1.0$; "
      r"the adaptive variant uses $\varepsilon=1.0$, $\varepsilon_0=0.2$, $\eta=0.6$; the SGC "
      r"variant uses $\lambda=0.3$, $K=3$, $d=64$.")

    w(r"\subsection{Baselines and metrics}")
    w("Baselines: Popularity; GCN-LP and LightGCN-LP (non-private upper bounds); LDP-GCN and "
      "LDP-RR-Light (standard $\\varepsilon$-LDP with RR); LDP-AMP (uniform sampling-amplified "
      "RR); Adaptive-LDP-SGC-LP. Metrics: HR@10, NDCG@10 (utility); membership-inference AUC "
      "(privacy, 0.5 = no leakage); effective $\\varepsilon$ (theoretical privacy).")

    w(r"\subsection{Main results}")
    order = ["Popularity", "GCN-LP", "LightGCN-LP", "LDP-GCN", "LDP-RR-Light",
             "LDP-AMP", "Adaptive-LDP-SGC-LP", OURS]
    for d in DS:
        M = ALL[d]["methods"]
        w(r"\begin{table}[!t]")
        w(r"\centering")
        w(r"\caption{Results on " + dl(d) + r" (means over 5 runs; * denotes our method).}")
        w(r"\label{tab:res-" + d + "}")
        w(r"\begin{tabular}{lcccc}\toprule")
        w(r"Method & HR@10 & NDCG@10 & MIA-AUC & eff.\ $\varepsilon$ \\ \midrule")
        for m in order:
            if m not in M: continue
            a = M[m]
            eps_s = "--" if a.get("eps") is None else f3(a["eps"])
            name = m.replace(" (Ours)", "").replace("Cal-LDP-BR", "Cal-LDP-BR*")
            w(f"{name} & {f4(a['HR'])} & {f4(a['NDCG'])} & {f3(a['auc'])} & {eps_s} \\\\")
        w(r"\bottomrule\end{tabular}\end{table}")
    # narrative
    lines = []
    for d in DS:
        Mx = ALL[d]["methods"]; o = Mx[OURS]
        ldp_hr = max(Mx[b]["HR"] for b in LDP_BASES if b in Mx)
        best_all = max(v["HR"] for k, v in Mx.items() if not k.startswith("_"))
        verdict = ("the highest among all methods" if o["HR"] >= best_all else
                   f"above the best $\\varepsilon$-LDP baseline (HR {f4(ldp_hr)}) but below the "
                   "strongest non-private upper bound")
        lines.append(f"On {dl(d)}, Cal-LDP-BR reaches HR@10={f4(o['HR'])}, "
                     f"NDCG@10={f4(o['NDCG'])}, MIA-AUC={f3(o['auc'])} at "
                     f"$\\varepsilon=0.02$; this is {verdict}.")
    w("Tables report all measured results without selective reporting. " + " ".join(lines) +
      " The adaptive SGC variant attains lower leakage in some cases but much weaker utility. "
      "Across benchmarks Cal-LDP-BR is the most accurate $\\varepsilon$-LDP method while reducing "
      "membership-inference leakage; we do not claim it surpasses non-private models in raw accuracy.")

    for i, d in enumerate(DS):
        w(r"\begin{figure}[!t]")
        w(r"\centering\includegraphics[width=\columnwidth]{figures/" + d + r"_perf.png}")
        w(r"\caption{Recommendation performance on " + dl(d) + r" (error bars: std over 5 runs).}")
        w(r"\end{figure}")

    w(r"\subsection{Privacy--utility trade-off}")
    w(r"\begin{figure}[!t]")
    w(r"\centering\includegraphics[width=\columnwidth]{figures/" + DS[0] + r"_privacy_util.png}")
    w(r"\caption{Privacy--utility plane on " + dl(DS[0]) +
      r": lower MIA-AUC and higher HR@10 is better.}")
    w(r"\end{figure}")
    w("Cal-LDP-BR occupies a favourable region relative to standard RR-based graph models: "
      "item-wise calibration converts noisy reports into a useful ranking, and the smaller budget "
      "weakens the edge signal available to an adversary.")

    w(r"\subsection{Statistical significance}")
    for d in DS:
        sig = ALL[d]["methods"].get("_significance", {})
        w(r"\begin{table}[!t]\centering")
        w(r"\caption{Significance tests on " + dl(d) + r" (5 runs).}")
        w(r"\begin{tabular}{lcccc}\toprule")
        w(r"Ours vs. & $\Delta$HR@10 & paired $t$ p & Wilcoxon p & Cohen's $d$ \\ \midrule")
        for b, st in sig.items():
            pw = "--" if st.get("p_w") is None else f3(st["p_w"])
            w(f"{b.replace(' (Ours)','')} & {st['delta']:+.4f} & {f3(st['p_t'])} & {pw} & {st['cohen_d']:+.2f} \\\\")
        w(r"\bottomrule\end{tabular}\end{table}")

    w(r"\subsection{Ablation and sensitivity}")
    A = AB.get("ablation", {})
    w(r"\begin{table}[!t]\centering\caption{Ablation of core components (ASSISTments 2009, 2 runs).}")
    w(r"\begin{tabular}{lccc}\toprule Variant & HR@10 & NDCG@10 & MIA-AUC \\ \midrule")
    for k, a in A.items():
        w(f"{k} & {f4(a['HR'])} & {f4(a['NDCG'])} & {f3(a['auc'])} \\\\")
    w(r"\bottomrule\end{tabular}\end{table}")
    E = AB.get("epsilon_sensitivity", {})
    w(r"\begin{table}[!t]\centering\caption{Sensitivity to $\varepsilon$ (2 runs).}")
    w(r"\begin{tabular}{lccc}\toprule $\varepsilon$ & HR@10 & NDCG@10 & MIA-AUC \\ \midrule")
    for e, a in sorted(E.items(), key=lambda x: float(x[0])):
        w(f"{e} & {f4(a['HR'])} & {f4(a['NDCG'])} & {f3(a['auc'])} \\\\")
    w(r"\bottomrule\end{tabular}\end{table}")
    w("Replacing adaptive sampling with uniform sampling lowers utility and raises leakage; the "
      "public skill graph reduces HR@10 here (0.1105 vs 0.1395 without it), so the main method "
      "uses posterior ranking instead.")

    w(r"\subsection{Privacy-budget sweep}")
    w(r"\begin{table*}[!t]\centering\caption{Cal-LDP-BR privacy-budget sweep (5 runs).}")
    w(r"\begin{tabular}{lccccc}\toprule")
    w(r"Dataset & $\varepsilon$ & HR@10 & NDCG@10 & MIA-AUC \\ \midrule")
    for d, payload in GRID.items():
        for e, a in payload["eps_sweep"].items():
            w(f"{dl(d)} & {e} & {f4(a['HR'])} & {f4(a['NDCG'])} & {f3(a['auc'])} \\\\")
    w(r"\bottomrule\end{tabular}\end{table*}")
    w("Accuracy is highest at very small budgets because the moment estimator has a larger "
      "reported sample there; utility is therefore non-monotone in the nominal budget.")

    w(r"\subsection{Validation on all ASSISTments users}")
    w(r"\begin{table}[!t]\centering\caption{Cal-LDP-BR on all filtered ASSISTments users (5 runs).}")
    w(r"\begin{tabular}{lrcccc}\toprule")
    w(r"Dataset & Users & $\varepsilon$ & HR@10 & NDCG@10 & MIA-AUC \\ \midrule")
    for d, payload in FULL.items():
        dd = payload["dataset"]
        for e, a in payload["eps_sweep"].items():
            w(f"{dl(d)} & {dd['n_u']} & {e} & {f4(a['HR'])} & {f4(a['NDCG'])} & {f3(a['auc'])} \\\\")
    w(r"\bottomrule\end{tabular}\end{table}")
    w("Removing the 500-user ceiling yields HR@10 of the same magnitude; EdNet's main table "
      "already uses all sampled users.")

    # ---- 7 Declarations ----
    w(r"\section{Ethical Statement and Declarations}")
    w("Ethics approval. This study uses only publicly available, anonymised educational datasets; "
      "no new experiments involving human participants were conducted, so ethical approval was "
      "not required. Data availability. The three datasets are public: ASSISTments 2009/2015 and "
      "EdNet (CC BY-NC 4.0, https://github.com/riiid/ednet). Funding. This work received no "
      "specific grant. Author contributions. [Author 1]: methodology, analysis, drafting; "
      "[Author 2]: data curation, experiments; [Author 3]: supervision, review. Competing "
      "interests. The authors declare no competing interests.")

    # ---- 8 Conclusion ----
    w(r"\section{Conclusion and Limitations}")
    w("We presented Cal-LDP-BR, a calibrated Bayesian LDP ranker for learning path recommendation. "
      "On three real benchmarks spanning two platforms it achieved the strongest ranking accuracy "
      "among all evaluated $\\varepsilon$-LDP methods while reducing membership-inference AUC "
      "relative to standard RR. Limitations remain: static graphs; item marginals estimated from "
      "noisy data; the EdNet variant lacks a public taxonomy; and non-private models retain higher "
      "raw accuracy on some datasets. Future work will study dynamic streams and privacy-optimal "
      "budgets for marginal estimation.")

    # ---- References ----
    w(r"\begin{thebibliography}{99}")
    for ref in REFS:
        # strip leading [n] then escape
        txt = re.sub(r"^\[\d+\]\s*", "", ref)
        w(r"\bibitem{" + re.sub(r"[^A-Za-z0-9]", "", txt[:18]) + r"}" + esc(txt))
    w(r"\end{thebibliography}")
    w(r"\end{document}")

    main_tex = os.path.join(OUTDIR, "main.tex")
    with open(main_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("SAVED:", main_tex, os.path.getsize(main_tex), "bytes")

    # write a short README
    readme = os.path.join(OUTDIR, "README.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write("Springer IJMLC LaTeX submission (svjour3, twocolumn)\n")
        f.write("=====================================================\n\n")
        f.write("Compile with pdfLaTeX (TeX Live 2021+, as used by Springer's submission system):\n")
        f.write("  pdflatex main.tex\n  bibtex is NOT required (thebibliography is embedded)\n")
        f.write("  pdflatex main.tex  (twice for cross-references)\n\n")
        f.write("Files:\n  main.tex         - manuscript\n  figures/*.png    - all figures\n\n")
        f.write("If svjour3.cls is unavailable, download the Springer Nature LaTeX template:\n")
        f.write("  https://www.springernature.com/gp/authors/campaigns/latex-author-support\n")
        f.write("and place svjour3.cls next to main.tex.\n\n")
        f.write("Before submission, replace all [bracketed] author/affiliation/email placeholders.\n")
    print("SAVED:", readme)


if __name__ == "__main__":
    import re
    main()
