# Cal-LDP-BR — Reproduction artifacts

Reference implementation and reproduction artifacts for the manuscript
**"Calibrated Bayesian Local Differential Privacy for Lightweight Learning Path Recommendation"**
submitted to the *International Journal of Machine Learning and Cybernetics* (IJMLC).

## Repository contents

- `code/` — reference implementation and experiment scripts
  - `bayes_ranker.py`, `ldp_mechanism.py`, `ds_loader.py` — core method (client perturbation, item-wise calibration, Bayesian posterior ranking)
  - `ldp_sgc_v2.py`, `ldp_sgc_lp_experiment.py`, `bayes_rank_grid.py`, `ablation_v2.py`, `hybrid_rank_experiment.py` — baselines and ablations
  - `prepare_ednet.py` — fixed 5,000-user EdNet KT1 sample preparation
  - `DATASET_DOWNLOAD_AND_RUN.txt` (also `DATASETDOWNLOADANDRUN.txt`) — step-by-step dataset download and full-data commands
  - `AUDIT_AND_REPRODUCIBILITY.txt` — audit and reproducibility report
  - `results/`, `results_v2/`, `results_test/` — raw result JSON and run logs for every table/figure
  - `figures/`, `figures_v2/` — generated figures
  - `theory_derivation.md` — derivation notes
- `LDP-SGC-LP_LaTeX/` — Springer LaTeX source of the manuscript (`main.tex`, `refs.bib`, `figures/`, class/style files)

## Data

The three benchmark datasets are public and are **not** redistributed here:

| Dataset | Source | License |
|---|---|---|
| ASSISTments 2009 (skill-builder, corrected) | [assistmentsdata.org](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) | see source |
| ASSISTments 2015 | [assistmentsdata.org](https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data) | see source |
| EdNet KT1 | [github.com/riiid/ednet](https://github.com/riiid/ednet) | CC BY-NC 4.0 |

Download links, expected filenames and the fixed sample specification are in
`code/DATASET_DOWNLOAD_AND_RUN.txt`. Put the datasets in a directory of your choice and
point the scripts at it with the `LDP_DATA_DIR` environment variable (default `./data`):

```bash
export LDP_DATA_DIR=/path/to/data      # Windows PowerShell: $env:LDP_DATA_DIR = "C:\path\to\data"
```

Student IDs are re-coded in all released results; no raw personal data are included.

## Reproducing the results

From `code/`:

```bash
python bayes_rank_grid.py        # main ranker: HR@10 / NDCG@10 / MIA-AUC across seeds
python ldp_sgc_lp_experiment.py  # LDP graph and SGC baselines
python ablation_v2.py            # adaptive SGC ablation (Section 6.9)
python prepare_ednet.py          # regenerate the fixed EdNet KT1 sample
```

All experiments use five fixed random seeds. Reported numbers in the manuscript come
directly from the JSON files under `results/` and `results_v2/`; run logs record seeds
and file hashes. Section B.2 of the manuscript describes the intended `make data`,
`make run`, `make eval`, `make figures` stage sequence; the scripts above are the
reference commands that implement those stages.

## Compiling the manuscript

```bash
cd LDP-SGC-LP_LaTeX
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Requires a TeX distribution with `sn-jnl`, `booktabs`, `algorithm/algpseudocode`,
`amsmath` and `natbib` (the Springer class and BST are included in the directory).

## License

Code released under the MIT license. Dataset licenses follow their respective sources.
