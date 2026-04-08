
# PAIRE

Benchmark accompanying PAIRE: A Testbed for Measuring Dataset Diversity and Model Fairness Under Limited Sensitive Information.

This repository provides the experimental pipeline used to evaluate quantification methods across three axes:

1. **Diversity Estimation** -- how well quantifiers estimate class prevalences under distribution shift.
2. **Fairness Assessment** -- whether quantifiers can reliably estimate group-level fairness metrics (demographic disparity, equalized opportunity) without access to individual sensitive attributes.
3. **Adversarial Vulnerability** -- adversarial (differencing) attack that attempts to reconstruct individual sensitive attributes from point estimates.

Two datasets are supported throughout: **UCI Adult** (tabular, binary sensitive attribute) and **TREC Fair Ranking** (text, multi-class geographic regions).

## Repository layout

```
paire/
├── data/                # Data files; populated by loading, tuning, and training example scripts
├── models/              # Trained quantifiers; populated by loading, tuning, and training example scripts
├── reports/             # Evaluation outputs; populated by loading, tuning, and training example scripts
├── src/
│   ├── data.py          # Dataset loading (UCI Adult)
│   ├── models.py        # Preprocessing, hyperparameter tuning, and model training
│   ├── evaluation.py    # Estimation accuracy, fairness evaluation, TREC corpus building
│   ├── adversarial.py   # Differencing attack implementation
│   ├── plots.py         # Paper figures
│   └── utils.py         # Shared helpers
└── examples/
    ├── load_data.py
    ├── tune_hyperparameters.py
    ├── train_models.py
    ├── evaluate_estimation_accuracy.py
    ├── evaluate_fairness.py
    └── evaluate_adversarial.py
```

##  Requirements
- Python 3.10+
- [QuaPy](https://github.com/HLT-ISTI/QuaPy)
- scikit-learn, pandas, numpy, scipy, tqdm
- dill, joblib
- Whoosh (TREC BM25 indexing only)
- matplotlib, seaborn (plotting only)

## Data preparation
**UCI Adult**
1. Download indices files from https://zenodo.org/records/14283870
2. Add `.indices` files to `data/`, which define three fixed splits (D1, D2, D3). To fetch the full dataset from OpenML and materialise the CSVs, execute the following line from the repository root:

```bash
python -m examples.load_data.py
```

**TREC Fair Ranking**
1. Download `trec_benchmark.tar.gz` from https://zenodo.org/records/19476764 and unpack with `tar -xvzf trec_benchmark.tar.gz`
2. Add `trec_train.jsonl` and the per-query `trec_test_query_*.jsonl` files to `data/`. These are derived from a modified version of the [TREC 2022 Fair Ranking Track](https://fair-trec.github.io/) corpus and are not fetched automatically.

## Pipeline

All example scripts are run from the repository root. Each script accepts `--dataset {adult,trec}` where applicable.

### 1. Tune hyperparameters

```bash
python -m examples.tune_hyperparameters.py --dataset adult
```

Runs a grid search over classifier regularisation (and KDE bandwidth where applicable) using QuaPy's `GridSearchQ`. Saves the best parameters to `models/params_adult.json`.

### 2. Train quantifiers

```bash
python -m examples.train_models.py --dataset adult
```

Trains each quantifier (CC, PCC, PACC, EMQ, KDEyML) with the tuned parameters and persists both the fitted models and preprocessing artefacts.

### 3. Evaluate estimation accuracy

```bash
python -m examples.evaluate_estimation_accuracy.py --dataset adult
```

Evaluates each trained quantifier under artificial prevalence shift (APP protocol for Adult, NPP for TREC) and reports mean absolute error (MAE) and mean relative absolute error (MRAE).

### 4. Evaluate fairness estimation

```bash
# Adult (quantifiers trained during evaluation)
python -m examples.evaluate_fairness.py --dataset adult

# TREC -- first build corpus artefacts (one-time, slow)
python -m examples.evaluate_fairness.py --dataset trec --build-corpus --index-dir /path/to/index

# TREC -- then evaluate (fairness quantifiers are trained on demand if missing)
python -m examples.evaluate_fairness.py --dataset trec --index-dir /path/to/index
```

For Adult, this estimates demographic disparity and equalised opportunity gaps using quantifiers fitted on D2 subgroups, evaluated across controlled prevalence shifts on D3.

For TREC, this measures how accurately quantifiers estimate the regional diversity (KL divergence from a target distribution) of BM25-ranked result lists.

### 5. Evaluate adversarial privacy

```bash
python -m examples.evaluate_adversarial.py --dataset adult \
    --n-attack-instances 500 \
    --background-sizes 1,10,100 \
    --vote-budgets 1,10,100
```

Runs the differencing attack: for each target individual, the attacker constructs query sets with and without the target, obtains quantifier estimates, and infers the sensitive attribute from the difference. Reports macro-F1 across background sizes and vote budgets.

## Quantifiers

The following QuaPy quantifiers are used by default:

| ID | Method |
|----|--------|
| CC | Classify and Count |
| PCC | Probabilistic Classify and Count |
| PACC | Adjusted Classify and Count |
| EMQ | Expectation-Maximisation for Quantification |
| KDEyML | Kernel Density Estimation (ML variant) |

In fairness evaluation outputs, CC and PCC are displayed as  TE (Threshold Estimator) and WE (Weighted Estimator) respectively to match the paper's notation.

## Citation

*To be added upon publication.*