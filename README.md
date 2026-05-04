
# PAIRE

Benchmark accompanying PAIRE: A Testbed for Measuring Dataset Diversity and Model Fairness Under Limited Sensitive Information.

This repository provides the experimental pipeline used to evaluate quantification methods across three axes:

1. **Diversity Estimation** -- how well quantifiers estimate class prevalences under distribution shift.
2. **Fairness Assessment** -- whether quantifiers can reliably estimate group-level fairness metrics (demographic disparity, equalized opportunity) without access to individual sensitive attributes.
3. **Adversarial Vulnerability** -- adversarial (differencing) attack that attempts to reconstruct individual sensitive attributes from point estimates.

Two datasets are supported throughout: **UCI Adult** (tabular, binary sensitive attribute) and **TREC Fair Ranking** (text, multi-class geographic regions).

## Two execution modes

The repository supports two complementary ways of running the pipeline:

1. **Benchmark mode** -- flexible scripts under `examples/` that you run stage-by-stage (tune, train, evaluate). Use this when iterating on hyperparameters, swapping in new quantifiers, or running ad-hoc experiments. CLI flags drive the behaviour and outputs go to `data/`, `models/`, and `reports/`.

2. **Reproducibility mode** -- paper-based scripts under `reproduce/` that are configured via YAML files in `configs/`. Use this when you want to reproduce the numbers reported in the paper. The scripts read their hyperparameters and protocol settings from the YAML, and outputs go to `models/paper/<dataset>/` and `reports/paper/<dataset>/` so they don't collide with benchmark-mode artefacts.

The two modes share the same underlying source code in `src/` (the reproducibility drivers call the same `run(...)` functions exposed by the example scripts).

## Repository layout

```
paire/
├── configs/                       # Frozen YAML configs for the reproducibility drivers
│   └── adult_paper.yaml
├── data/                          # Input data (.indices, CSVs, JSONL, fitted preprocessors)
│   ├── adult/
│   └── trec/
├── models/                        # Trained quantifier artefacts
│   ├── adult/                     # Benchmark-based outputs
│   ├── trec/
│   └── paper/                     # Reproducibility-based outputs
│       ├── adult/
│       └── trec/
├── reports/                       # Evaluation outputs (estimation, fairness, adversarial)
│   ├── adult/
│   ├── trec/
│   └── paper/
│       ├── adult/
│       └── trec/
├── src/
│   ├── data.py                    # Dataset loading (UCI Adult)
│   ├── models.py                  # Preprocessing, hyperparameter tuning, model training
│   ├── evaluation.py              # Estimation accuracy, fairness evaluation, TREC corpus building
│   ├── adversarial.py             # Differencing attack implementation
│   ├── plots.py                   # Paper figures
│   └── utils.py                   # Shared helpers
├── examples/                      # Stage-by-stage scripts (benchmark)
│   ├── load_data.py
│   ├── tune_hyperparameters.py
│   ├── train_models.py
│   ├── evaluate_estimation_accuracy.py
│   ├── evaluate_fairness.py
│   └── evaluate_adversarial.py
└── reproduce/                     # Paper-based entry points (reproducibility)
    └── run_adult.py
```

##  Requirements
- Python 3.10+
- [QuaPy](https://github.com/HLT-ISTI/QuaPy)
- scikit-learn, pandas, numpy, scipy, tqdm
- dill, joblib
- PyYAML (reproducibility-mode config loading)
- Whoosh (TREC BM25 indexing only)
- matplotlib, seaborn (plotting only)

Install everything from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Data preparation

**UCI Adult**
1. Download the indices files from https://zenodo.org/records/14283870.
2. Place the `.indices` files into `data/adult/` (they define three fixed splits D1, D2, D3). To fetch the full dataset from OpenML and materialise the CSVs, run:

```bash
python -m examples.load_data --dataset adult
```

**TREC Fair Ranking**
1. Download `trec_benchmark.tar.gz` from https://zenodo.org/records/19476764 and unpack with `tar -xvzf trec_benchmark.tar.gz`.
2. Place `trec_train.jsonl` and the per-query `trec_test_query_*.jsonl` files into `data/trec/`. These are derived from a modified version of the [TREC 2022 Fair Ranking Track](https://fair-trec.github.io/) corpus and are not fetched automatically.

## Reproducibility mode

The fastest way to reproduce the paper's Adult results is:

```bash
python -m reproduce.run_adult
```

This single command:

1. checks that the Adult `.indices` files are present and (re)builds `adult_D1.csv`, `adult_D2.csv`, `adult_D3.csv` if needed;
2. trains the quantifiers listed in `configs/adult_paper.yaml` with frozen hyperparameters used in the paper;
3. runs the estimation quality, fairness, and differencing attack stages with the protocol settings from the YAML;
4. writes models to `models/paper/adult/` and reports to `reports/paper/adult/`.

### CLI options

```
python -m reproduce.run_adult \
    [--config configs/adult_paper.yaml] \
    [--rebuild-data]                      # force regeneration of split CSVs
    [--retrain]                           # force retraining even if model files exist
    [--skip-training]                     # skip the training stage
    [--skip-estimation]                   # skip the estimation-quality stage
    [--skip-fairness]                     # skip the fairness stage
    [--skip-adversarial]                  # skip the differencing-attack stage
    [--adversarial-workers N]             # parallel workers for the differencing attack (default: cpu - 1)
```

## Benchmark mode

All example scripts are run from the repository root and accept `--dataset {adult,trec}` where applicable.

### 1. Tune hyperparameters

```bash
python -m examples.tune_hyperparameters --dataset adult
```

Runs a grid search over classifier regularisation (and KDE bandwidth) using QuaPy's `GridSearchQ`. Saves the best parameters to `models/params_<dataset>.json`.

### 2. Train quantifiers

```bash
python -m examples.train_models --dataset adult
```

Trains each quantifier (CC, PCC, PACC, EMQ, KDEyML) with the tuned parameters in `models/params_<dataset>.json` and persists both the fitted models and preprocessing artefacts.

### 3. Evaluate estimation accuracy

```bash
python -m examples.evaluate_estimation_accuracy --dataset adult
```

Evaluates each trained quantifier under artificial prevalence shift (APP protocol for Adult, NPP for TREC) and reports mean absolute error (MAE) and mean relative absolute error (MRAE).

### 4. Evaluate fairness estimation

```bash
# Adult
python -m examples.evaluate_fairness --dataset adult

# TREC -- first build corpus artefacts (one-time, slow)
python -m examples.evaluate_fairness --dataset trec --build-corpus --index-dir /path/to/index

# TREC -- then evaluate (fairness quantifiers are trained on demand if missing)
python -m examples.evaluate_fairness --dataset trec --index-dir /path/to/index
```

For Adult, this estimates demographic disparity and equalised opportunity gaps using quantifiers fitted on D2 subgroups, evaluated across controlled prevalence shifts on D3.

For TREC, this measures how accurately quantifiers estimate the regional diversity (KL divergence from a target distribution) of BM25-ranked result lists.

### 5. Evaluate adversarial privacy

```bash
python -m examples.evaluate_adversarial --dataset adult \
    --n-attack-instances 500 \
    --background-sizes 1,10,100 \
    --vote-budgets 1,10,100
```

Runs the differencing attack: for each target individual, the attacker constructs query sets with and without the target, obtains quantifier estimates, and infers the sensitive attribute from the difference. Reports macro-F1 across background sizes and vote budgets.

### Programmatic API

Each example script also exposes a `run(...)` function so the benchmark stages can be composed from Python (this is what `reproduce/run_adult.py` uses internally). For example:

```python
from pathlib import Path
from examples.train_models import run as train_run
from examples.evaluate_estimation_accuracy import run as estimation_run

train_run(
    dataset="adult",
    data_dir=Path("data/adult"),
    models_dir=Path("models/adult"),
    parameters={"CC": {"classifier__C": 10.0, ...}, ...},
    quantifiers=["CC", "PCC"],
    random_seed=0,
)

estimation_run(
    dataset="adult",
    data_dir=Path("data/adult"),
    models_dir=Path("models/adult"),
    reports_dir=Path("reports/adult"),
    sample_size=500,
    repeats=10,
    quantifiers=["CC", "PCC"],
)
```

## Quantifiers

The following QuaPy quantifiers are used by default:

| ID     | Method |
|--------|--------|
| CC     | Classify and Count |
| PCC    | Probabilistic Classify and Count |
| PACC   | Probabilistic Adjusted Classify and Count |
| EMQ    | Expectation-Maximisation for Quantification |
| KDEyML | Kernel Density Estimation (Maximum Likelihood) |

In fairness evaluation outputs, CC and PCC are displayed as TE (Threshold Estimator) and WE (Weighted Estimator) respectively to match the paper's notation.

## Citation

*To be added upon publication.*
