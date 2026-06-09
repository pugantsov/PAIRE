"""
Paper-faithful reproducibility entry point for the TREC experiments.

- Reads a frozen YAML config
- Checks that the required TREC input files are present
- Trains the quantifiers listed in the config
- Runs the estimation-quality evaluation over the per-query test files
- Runs the fairness (ranked-list diversity) evaluation: builds the BM25
  corpus/index, ranks documents per query, and measures the mean ranked
  fairness error (MRFE) of each quantifier
- Runs the adversarial-vulnerability evaluation: the differencing attack
  that recovers sensitive (region) labels via majority-voted quantifier
  queries, scored by macro-F1
- Outputs land under the paths declared in the config (by default
  models/paper/trec and reports/paper/trec).

Usage:

    python -m reproduce.run_trec [--config configs/trec_paper.yaml]

Args:

        --config: Path to the YAML config (default: configs/trec_paper.yaml).
        --retrain: Retrain models even if all expected model files exist.
        --skip-training: Skip the training stage.
        --skip-estimation: Skip the estimation-quality stage.
        --skip-fairness: Skip the fairness stage.
        --skip-adversarial: Skip the adversarial (differencing-attack) stage.
        --estimation-workers: Number of worker processes for per-query
            estimation.
        --adversarial-workers: Number of worker processes for the differencing
            attack.
        --rebuild-fairness-corpus: Rebuild the fairness corpus artefacts
            (vectorizer, queries, docs, BM25 index, ranked lists) even if
            they already exist.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import dill as pickle
import joblib
import pandas as pd
import yaml
from whoosh.index import exists_in

from examples.train_models import run as train_run
from examples.evaluate_estimation_accuracy import run as estimation_run
from examples.evaluate_adversarial import run as adversarial_run
from src.evaluation import (
    TRECFairnessCorpusBuilder,
    TRECFairnessEvaluator,
)

REQUIRED_TOP_LEVEL_KEYS = (
    "dataset",
    "random_seed",
    "paths",
    "quantifiers",
    "training",
    "estimation",
)
REQUIRED_PATH_KEYS = ("data_dir", "models_dir", "reports_dir")
REQUIRED_ESTIMATION_KEYS = ("sample_size", "repeats")
REQUIRED_ADVERSARIAL_KEYS = (
    "n_attack_instances",
    "background_sizes",
    "vote_budgets",
)
TREC_TRAIN_FILE = "trec_train.jsonl"
TREC_TEST_QUERY_GLOB = "trec_test_query_*.jsonl"
MODEL_SUFFIX = "trec"
DEFAULT_ADVERSARIAL_N_RUNS = 5

DEFAULT_INDEX_DIR = "data/trec/bm25/index"
FAIR_VECTORIZER_FILE = "trec_fair_vectorizer.joblib"
FAIR_QUERIES_FILE = "trec_fair_queries.csv"
FAIR_DOCS_FILE = "trec_fair_docs.csv"
FAIR_RANKED_LISTS_FILE = "trec_fair_ranked_lists.pkl"
FAIR_LABEL_ENCODER_FILE = "label_encoder_trec_train.joblib"
FAIR_MODEL_PREFIX = "trec_fair"
FAIR_EXPERIMENT_NAME = "BM25"
DEFAULT_FAIRNESS_CUTOFFS = [50, 100, 500, 1000]
DEFAULT_FAIRNESS_TOP_N_TERMS = 100
DEFAULT_FAIRNESS_RANK_LIM = 10000
DEFAULT_FAIRNESS_SPLIT_TEST_SIZE = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the TREC experiments from the paper using a "
            "frozen YAML config."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/trec_paper.yaml"),
        help="Path to the YAML config (default: configs/trec_paper.yaml).",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain models even if all expected model files exist.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the training stage.",
    )
    parser.add_argument(
        "--skip-estimation",
        action="store_true",
        help="Skip the estimation-quality stage.",
    )
    parser.add_argument(
        "--skip-fairness",
        action="store_true",
        help="Skip the fairness stage.",
    )
    parser.add_argument(
        "--skip-adversarial",
        action="store_true",
        help="Skip the adversarial (differencing-attack) stage.",
    )
    parser.add_argument(
        "--estimation-workers",
        type=int,
        default=max(mp.cpu_count() // 2, 1),
        help="Number of worker processes for per-query estimation.",
    )
    parser.add_argument(
        "--adversarial-workers",
        type=int,
        default=max(mp.cpu_count() - 1, 1),
        help="Number of worker processes for the differencing attack.",
    )
    parser.add_argument(
        "--rebuild-fairness-corpus",
        action="store_true",
        help=(
            "Rebuild the fairness corpus artefacts (vectorizer, queries, "
            "docs, BM25 index, ranked lists) even if they already exist."
        ),
    )
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(
            f"Config at {config_path} did not parse to a mapping."
        )
    return config


def validate_config(config: dict[str, Any]) -> None:
    missing_top = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in config]
    if missing_top:
        raise ValueError(
            f"Config is missing required top-level keys: {missing_top}"
        )

    if config["dataset"] != "trec":
        raise ValueError(
            f"This script only supports dataset='trec' "
            f"(got {config['dataset']!r})."
        )

    paths = config["paths"]
    if not isinstance(paths, dict):
        raise ValueError("Config 'paths' must be a mapping.")
    missing_paths = [k for k in REQUIRED_PATH_KEYS if k not in paths]
    if missing_paths:
        raise ValueError(
            f"Config 'paths' is missing required keys: {missing_paths}"
        )

    training = config["training"]
    if not isinstance(training, dict) or "parameters" not in training:
        raise ValueError(
            "Config 'training' must be a mapping containing 'parameters'."
        )

    parameters = training["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("Config 'training.parameters' must be a mapping.")

    quantifiers = config["quantifiers"]
    if not isinstance(quantifiers, list) or not quantifiers:
        raise ValueError("Config 'quantifiers' must be a non-empty list.")

    missing_params = [q for q in quantifiers if q not in parameters]
    if missing_params:
        raise ValueError(
            "Every quantifier listed in 'quantifiers' must have an entry "
            f"in 'training.parameters'. Missing: {missing_params}"
        )

    estimation = config["estimation"]
    if not isinstance(estimation, dict):
        raise ValueError("Config 'estimation' must be a mapping.")
    missing_est = [k for k in REQUIRED_ESTIMATION_KEYS if k not in estimation]
    if missing_est:
        raise ValueError(
            f"Config 'estimation' is missing required keys: {missing_est}"
        )

    adversarial = config.get("adversarial")
    if adversarial is not None:
        if not isinstance(adversarial, dict):
            raise ValueError("Config 'adversarial' must be a mapping.")
        missing_adv = [
            k for k in REQUIRED_ADVERSARIAL_KEYS if k not in adversarial
        ]
        if missing_adv:
            raise ValueError(
                f"Config 'adversarial' is missing required keys: {missing_adv}"
            )
        for list_key in ("background_sizes", "vote_budgets"):
            if (
                not isinstance(adversarial[list_key], list)
                or not adversarial[list_key]
            ):
                raise ValueError(
                    f"Config 'adversarial.{list_key}' must be a non-empty list."
                )


def resolve_paths(
    project_root: Path,
    config: dict[str, Any],
) -> dict[str, Path]:
    paths_cfg = config["paths"]
    resolved = {
        key: (project_root / paths_cfg[key]).resolve()
        for key in REQUIRED_PATH_KEYS
    }
    resolved["models_dir"].mkdir(parents=True, exist_ok=True)
    resolved["reports_dir"].mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_trec_data(data_dir: Path) -> tuple[Path, list[Path]]:
    """
    Make sure the required TREC input files are present in `data_dir`.

    Returns the training-file path and the sorted list of per-query test
    files. Raises FileNotFoundError if anything required is missing; the
    TREC files are provided externally and are not (re)built here.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"TREC data directory {data_dir} does not exist. "
            f"Place the unpacked TREC files there first."
        )

    train_path = data_dir / TREC_TRAIN_FILE
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing TREC training file {train_path}.")

    query_paths = sorted(data_dir.glob(TREC_TEST_QUERY_GLOB))
    if not query_paths:
        raise FileNotFoundError(
            f"No TREC test query files matching '{TREC_TEST_QUERY_GLOB}' "
            f"found in {data_dir}."
        )

    print(
        f"  Found TREC training file {train_path.name} and "
        f"{len(query_paths)} test query files in {data_dir}."
    )
    return train_path, query_paths


def expected_model_paths(
    models_dir: Path,
    quantifiers: list[str],
) -> list[Path]:
    return [models_dir / f"{qid}_{MODEL_SUFFIX}.pkl" for qid in quantifiers]


def models_exist(models_dir: Path, quantifiers: list[str]) -> bool:
    return all(
        p.exists() for p in expected_model_paths(models_dir, quantifiers)
    )


def run_training_if_needed(
    data_dir: Path,
    models_dir: Path,
    quantifiers: list[str],
    parameters: dict[str, dict[str, Any]],
    random_seed: int,
    retrain: bool,
) -> None:
    expected = expected_model_paths(models_dir, quantifiers)
    have_all = models_exist(models_dir, quantifiers)

    if have_all and not retrain:
        print(
            f"  All expected model artefacts already exist in {models_dir}; "
            f"skipping training (use --retrain to force)."
        )
        return

    if retrain:
        print("  --retrain set: training all quantifiers.")
    else:
        missing = [p.name for p in expected if not p.exists()]
        print(
            f"  Missing model artefacts {missing}; training all quantifiers."
        )

    train_run(
        dataset="trec",
        data_dir=data_dir,
        models_dir=models_dir,
        parameters=parameters,
        quantifiers=quantifiers,
        random_seed=random_seed,
    )
    print(f"  Models written to {models_dir}.")


def run_estimation(
    data_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    quantifiers: list[str],
    estimation_cfg: dict[str, Any],
    n_workers: int,
) -> None:
    protocol = estimation_cfg.get("protocol", "NPP")
    sample_size = estimation_cfg["sample_size"]
    repeats = estimation_cfg["repeats"]

    if protocol != "NPP":
        raise ValueError(
            f"TREC estimation uses protocol='NPP' " f"(got {protocol!r})."
        )

    print(
        f"  Estimation-quality stage: protocol={protocol}, "
        f"sample_size={sample_size}, repeats={repeats}, "
        f"n_workers={n_workers}."
    )

    estimation_run(
        dataset="trec",
        data_dir=data_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        sample_size=sample_size,
        repeats=repeats,
        quantifiers=quantifiers,
        n_workers=n_workers,
    )
    print(f"  Estimation reports written to {reports_dir}.")


def resolve_index_dir(project_root: Path, config: dict[str, Any]) -> Path:
    index_rel = config["paths"].get("index_dir", DEFAULT_INDEX_DIR)
    return (project_root / index_rel).resolve()


def build_fairness_corpus(
    data_dir: Path,
    index_dir: Path,
    fairness_cfg: dict[str, Any],
    random_seed: int,
    rebuild: bool,
) -> None:
    """
    Build (or reuse) the fairness corpus artefacts: the fair-train TF-IDF
    vectorizer, the per-query term-based query table, the combined docs table,
    and the Whoosh BM25 index.
    """
    split_cfg = fairness_cfg.get("split", {})
    test_size = float(
        split_cfg.get("test_size", DEFAULT_FAIRNESS_SPLIT_TEST_SIZE)
    )
    split_random_state = int(split_cfg.get("random_state", random_seed))
    top_n_terms = int(
        fairness_cfg.get("query_top_n_terms", DEFAULT_FAIRNESS_TOP_N_TERMS)
    )

    vectorizer_path = data_dir / FAIR_VECTORIZER_FILE
    queries_path = data_dir / FAIR_QUERIES_FILE
    docs_path = data_dir / FAIR_DOCS_FILE

    artefacts_present = (
        vectorizer_path.is_file()
        and queries_path.is_file()
        and docs_path.is_file()
        and index_dir.exists()
        and exists_in(str(index_dir))
    )
    if artefacts_present and not rebuild:
        print(
            f"  Fairness corpus artefacts already present in {data_dir} "
            f"and {index_dir}; reusing (use --rebuild-fairness-corpus to "
            f"force)."
        )
        return

    if rebuild:
        print("  --rebuild-fairness-corpus set: rebuilding corpus artefacts.")
    else:
        print("  Building fairness corpus artefacts.")

    builder = TRECFairnessCorpusBuilder(
        top_n_terms=top_n_terms,
        random_state=split_random_state,
    )

    train_df = pd.read_json(data_dir / TREC_TRAIN_FILE, lines=True)
    fair_train, nonrel_corpus = builder.build_train_nonrelevant_split(
        train_df,
        test_size=test_size,
    )

    vectorizer = builder.vectorize(fair_train["text"].values)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"    Wrote fair-train vectorizer to {vectorizer_path}.")

    query_paths = sorted(data_dir.glob(TREC_TEST_QUERY_GLOB))
    queries_df, rel_docs = builder.build_query_table(
        query_paths=query_paths,
        vectorizer=vectorizer,
    )
    docs_df = builder.build_docs_table(rel_docs, nonrel_corpus)

    builder.save_dataframe(queries_df, queries_path)
    builder.save_dataframe(docs_df, docs_path)
    print(
        f"    Wrote {len(queries_df)} queries and {len(docs_df)} docs to "
        f"{data_dir}."
    )

    builder.build_bm25_index(docs_df, index_dir, overwrite=True)
    print(f"    Built BM25 index in {index_dir}.")


def rank_or_load_ranked_lists(
    data_dir: Path,
    index_dir: Path,
    queries_df: pd.DataFrame,
    rank_lim: int,
    rebuild: bool,
) -> dict[str, list[tuple]]:
    """
    Rank the BM25 corpus for every fairness query, caching the result so
    repeated runs reuse the same ranked lists.
    """
    ranked_path = data_dir / FAIR_RANKED_LISTS_FILE

    if ranked_path.is_file() and not rebuild:
        print(f"  Reusing cached ranked lists from {ranked_path}.")
        with ranked_path.open("rb") as f:
            return pickle.load(f)

    ranked_lists = TRECFairnessCorpusBuilder.rank_bm25_index(
        queries_df,
        index_dir,
        limit=rank_lim,
    )
    with ranked_path.open("wb") as f:
        pickle.dump(ranked_lists, f)
    print(f"  Ranked lists written to {ranked_path}.")
    return ranked_lists


def run_fairness(
    data_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    index_dir: Path,
    quantifiers: list[str],
    fairness_cfg: dict[str, Any],
    random_seed: int,
    rebuild_corpus: bool,
) -> None:
    cutoffs = list(fairness_cfg.get("cutoffs", DEFAULT_FAIRNESS_CUTOFFS))
    rank_lim = int(fairness_cfg.get("rank_lim", DEFAULT_FAIRNESS_RANK_LIM))

    label_encoder_path = data_dir / FAIR_LABEL_ENCODER_FILE
    if not label_encoder_path.is_file():
        raise FileNotFoundError(
            f"Missing {label_encoder_path}. Run the training stage first so "
            f"the TREC label encoder is available."
        )

    print(
        f"  Fairness stage: quantifiers={quantifiers}, cutoffs={cutoffs}, "
        f"rank_lim={rank_lim}, index_dir={index_dir}."
    )

    build_fairness_corpus(
        data_dir=data_dir,
        index_dir=index_dir,
        fairness_cfg=fairness_cfg,
        random_seed=random_seed,
        rebuild=rebuild_corpus,
    )

    if rebuild_corpus:
        stale_models = sorted(models_dir.glob(f"{FAIR_MODEL_PREFIX}_*.pkl"))
        for stale in stale_models:
            stale.unlink()
        if stale_models:
            print(
                f"  Removed {len(stale_models)} stale fairness model(s) so "
                f"they are retrained against the rebuilt vectorizer."
            )

    queries_df = pd.read_csv(data_dir / FAIR_QUERIES_FILE)
    ranked_lists = rank_or_load_ranked_lists(
        data_dir=data_dir,
        index_dir=index_dir,
        queries_df=queries_df,
        rank_lim=rank_lim,
        rebuild=rebuild_corpus,
    )

    docs_df = pd.read_csv(data_dir / FAIR_DOCS_FILE)
    vectorizer = joblib.load(data_dir / FAIR_VECTORIZER_FILE)
    label_encoder = joblib.load(label_encoder_path)

    evaluator = TRECFairnessEvaluator(
        quantifiers=quantifiers,
        cutoffs=cutoffs,
    )
    results = evaluator.evaluate_ranked_lists(
        ranked_lists=ranked_lists,
        docs=docs_df,
        vectorizer=vectorizer,
        models_dir=models_dir,
        data_dir=data_dir,
        labels=label_encoder.classes_,
        experiment_name=FAIR_EXPERIMENT_NAME,
        model_prefix=FAIR_MODEL_PREFIX,
    )

    results_path = reports_dir / "trec_fairness_results.csv"
    evaluator.save_report(results, results_path)
    print(f"  Per-cutoff fairness results written to {results_path}.")

    rkl = evaluator.aggregate_rkl(results)
    rkl_path = reports_dir / "trec_fairness_rkl.csv"
    evaluator.save_report(rkl, rkl_path)
    print(f"  Aggregated rKL/MRFE report written to {rkl_path}.")

    summary = evaluator.generate_summary_table(rkl)
    print(summary)


def run_adversarial(
    data_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    quantifiers: list[str],
    adversarial_cfg: dict[str, Any],
    random_seed: int,
    n_workers: int,
) -> None:
    n_attack_instances = int(adversarial_cfg["n_attack_instances"])
    background_sizes = list(adversarial_cfg["background_sizes"])
    vote_budgets = list(adversarial_cfg["vote_budgets"])
    n_runs = int(adversarial_cfg.get("n_runs", DEFAULT_ADVERSARIAL_N_RUNS))

    print(
        f"  Adversarial stage: n_attack_instances={n_attack_instances}, "
        f"n_runs={n_runs}, background_sizes={background_sizes}, "
        f"vote_budgets={vote_budgets}, n_workers={n_workers}."
    )

    _, output_path = adversarial_run(
        dataset="trec",
        data_dir=data_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        quantifiers=quantifiers,
        n_attack_instances=n_attack_instances,
        n_runs=n_runs,
        background_sizes=background_sizes,
        vote_budgets=vote_budgets,
        base_seed=random_seed,
        n_workers=n_workers,
    )
    print(f"  Adversarial report written to {output_path}.")


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    print(f"[reproduce/run_trec] Project root: {project_root}")
    print(f"[reproduce/run_trec] Config:       {config_path}")

    config = load_yaml_config(config_path)
    validate_config(config)

    paths = resolve_paths(project_root, config)
    data_dir = paths["data_dir"]
    models_dir = paths["models_dir"]
    reports_dir = paths["reports_dir"]
    index_dir = resolve_index_dir(project_root, config)

    quantifiers: list[str] = list(config["quantifiers"])
    parameters: dict[str, dict[str, Any]] = config["training"]["parameters"]
    random_seed: int = int(config["random_seed"])

    print(f"[reproduce/run_trec] data_dir:     {data_dir}")
    print(f"[reproduce/run_trec] models_dir:   {models_dir}")
    print(f"[reproduce/run_trec] reports_dir:  {reports_dir}")
    print(f"[reproduce/run_trec] quantifiers:  {quantifiers}")
    print(f"[reproduce/run_trec] random_seed:  {random_seed}")

    print("\n[1/5] TREC data preparation")
    ensure_trec_data(data_dir=data_dir)

    print("\n[2/5] Training")
    if args.skip_training:
        print("  --skip-training set; skipping.")
    else:
        run_training_if_needed(
            data_dir=data_dir,
            models_dir=models_dir,
            quantifiers=quantifiers,
            parameters=parameters,
            random_seed=random_seed,
            retrain=args.retrain,
        )

    print("\n[3/5] Estimation quality")
    if args.skip_estimation:
        print("  --skip-estimation set; skipping.")
    else:
        run_estimation(
            data_dir=data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            quantifiers=quantifiers,
            estimation_cfg=config["estimation"],
            n_workers=args.estimation_workers,
        )

    print("\n[4/5] Fairness / ranked-list diversity")
    if args.skip_fairness:
        print("  --skip-fairness set; skipping.")
    else:
        fairness_cfg = config.get("fairness")
        if not isinstance(fairness_cfg, dict):
            raise ValueError(
                "Config 'fairness' must be a mapping to run the fairness "
                "stage (or pass --skip-fairness)."
            )
        run_fairness(
            data_dir=data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            index_dir=index_dir,
            quantifiers=quantifiers,
            fairness_cfg=fairness_cfg,
            random_seed=random_seed,
            rebuild_corpus=args.rebuild_fairness_corpus,
        )

    print("\n[5/5] Adversarial vulnerability")
    if args.skip_adversarial:
        print("  --skip-adversarial set; skipping.")
    else:
        adversarial_cfg = config.get("adversarial")
        if not isinstance(adversarial_cfg, dict):
            raise ValueError(
                "Config 'adversarial' must be a mapping to run the "
                "adversarial stage (or pass --skip-adversarial)."
            )
        run_adversarial(
            data_dir=data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            quantifiers=quantifiers,
            adversarial_cfg=adversarial_cfg,
            random_seed=random_seed,
            n_workers=args.adversarial_workers,
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
