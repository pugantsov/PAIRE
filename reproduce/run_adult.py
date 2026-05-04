"""
Paper-faithful reproducibility entry point for the Adult experiments.

- Reads a frozen YAML config
- Prepares the Adult split CSVs if needed
- Trains the quantifiers listed in the config
- Runs the estimation-quality, fairness, and adversarial evaluations
- Outputs land under the paths declared in the config (by default models/paper/adult and reports/paper/adult).

Usage:

    python -m reproduce.run_adult [--config configs/adult_paper.yaml]

Args:

        --config: Path to the YAML config (default: configs/adult_paper.yaml).
        --rebuild-data: Rebuild Adult split CSVs even if they already exist.
        --retrain: Retrain models even if all expected model files exist.
        --skip-training: Skip the training stage.
        --skip-estimation: Skip the estimation-quality stage.
        --skip-fairness: Skip the fairness stage.
        --skip-adversarial: Skip the adversarial (differencing-attack) stage.
        --adversarial-workers: Number of worker processes for the differencing attack.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import yaml

from examples.train_models import run as train_run
from examples.evaluate_estimation_accuracy import run as estimation_run
from examples.evaluate_fairness import run_adult_eval
from examples.evaluate_adversarial import run as adversarial_run
from src.data import AdultDatasetLoader

REQUIRED_TOP_LEVEL_KEYS = (
    "dataset",
    "random_seed",
    "paths",
    "quantifiers",
    "training",
    "estimation",
    "fairness",
    "adversarial",
)
REQUIRED_PATH_KEYS = ("data_dir", "models_dir", "reports_dir")
REQUIRED_ADVERSARIAL_KEYS = (
    "n_attack_instances",
    "background_sizes",
    "vote_budgets",
)
ADULT_SPLIT_CSV_NAMES = ("adult_D1.csv", "adult_D2.csv", "adult_D3.csv")
MODEL_SUFFIX = "adult"
DEFAULT_ADVERSARIAL_N_RUNS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the Adult experiments from the paper using a "
            "frozen YAML config."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/adult_paper.yaml"),
        help="Path to the YAML config (default: configs/adult_paper.yaml).",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Rebuild Adult split CSVs even if they already exist.",
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
        "--adversarial-workers",
        type=int,
        default=max(mp.cpu_count() - 1, 1),
        help="Number of worker processes for the differencing attack.",
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

    if config["dataset"] != "adult":
        raise ValueError(
            f"This script only supports dataset='adult' "
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

    adversarial = config["adversarial"]
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


def ensure_adult_data(
    data_dir: Path,
    rebuild: bool,
) -> bool:
    """
    Make sure the Adult split CSVs exist in `data_dir`.

    Returns True if the CSVs were (re)built in this call, False if the
    existing ones were reused.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Adult data directory {data_dir} does not exist."
        )

    csv_paths = [data_dir / name for name in ADULT_SPLIT_CSV_NAMES]
    existing = [p for p in csv_paths if p.exists()]
    missing = [p for p in csv_paths if not p.exists()]

    if not rebuild and not missing:
        print(
            f"  Adult split CSVs already present in {data_dir} "
            f"(found {len(existing)}/{len(csv_paths)})."
        )
        return False

    if rebuild:
        print(f"  Regenerating Adult split CSVs in {data_dir}.")
    else:
        print(
            f"  Missing Adult split CSVs ({[p.name for p in missing]}); "
            f"rebuilding from .indices files in {data_dir}."
        )

    loader = AdultDatasetLoader(data_dir=data_dir)
    loader.write_split_csvs()
    print(f"  Wrote Adult split CSVs to {data_dir}.")
    return True


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
        dataset="adult",
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
) -> None:
    protocol = estimation_cfg.get("protocol", "APP")
    sample_size = estimation_cfg["sample_size"]
    repeats = estimation_cfg["repeats"]

    if protocol != "APP":
        raise ValueError(
            f"Adult estimation reproduces protocol='APP' "
            f"(got {protocol!r})."
        )

    print(
        f"  Estimation-quality stage: protocol={protocol}, "
        f"sample_size={sample_size}, repeats={repeats}."
    )

    estimation_run(
        dataset="adult",
        data_dir=data_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        sample_size=sample_size,
        repeats=repeats,
        quantifiers=quantifiers,
    )
    print(f"  Estimation reports written to {reports_dir}.")


def run_fairness(
    data_dir: Path,
    reports_dir: Path,
    quantifiers: list[str],
    fairness_cfg: dict[str, Any],
    random_seed: int,
) -> None:
    n_prevalences = fairness_cfg["n_prevalences"]
    max_prev = fairness_cfg["max_prev"]
    sample_size = fairness_cfg["sample_size"]
    repeats = fairness_cfg["repeats"]

    print(
        f"  Fairness stage: n_prevalences={n_prevalences}, "
        f"max_prev={max_prev}, sample_size={sample_size}, repeats={repeats}."
    )

    output_path = run_adult_eval(
        data_dir=data_dir,
        reports_dir=reports_dir,
        dataset_id="adult",
        n_prevalences=n_prevalences,
        max_prev=max_prev,
        sample_size=sample_size,
        repeats=repeats,
        quantifiers=quantifiers,
        random_state=random_seed,
    )
    print(f"  Fairness report written to {output_path}.")


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
        dataset="adult",
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

    print(f"[reproduce/run_adult] Project root: {project_root}")
    print(f"[reproduce/run_adult] Config:       {config_path}")

    config = load_yaml_config(config_path)
    validate_config(config)

    paths = resolve_paths(project_root, config)
    data_dir = paths["data_dir"]
    models_dir = paths["models_dir"]
    reports_dir = paths["reports_dir"]

    quantifiers: list[str] = list(config["quantifiers"])
    parameters: dict[str, dict[str, Any]] = config["training"]["parameters"]
    random_seed: int = int(config["random_seed"])

    print(f"[reproduce/run_adult] data_dir:     {data_dir}")
    print(f"[reproduce/run_adult] models_dir:   {models_dir}")
    print(f"[reproduce/run_adult] reports_dir:  {reports_dir}")
    print(f"[reproduce/run_adult] quantifiers:  {quantifiers}")
    print(f"[reproduce/run_adult] random_seed:  {random_seed}")

    print("\n[1/5] Adult data preparation")
    ensure_adult_data(
        data_dir=data_dir,
        rebuild=args.rebuild_data,
    )

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
        )

    print("\n[4/5] Fairness")
    if args.skip_fairness:
        print("  --skip-fairness set; skipping.")
    else:
        run_fairness(
            data_dir=data_dir,
            reports_dir=reports_dir,
            quantifiers=quantifiers,
            fairness_cfg=config["fairness"],
            random_seed=random_seed,
        )

    print("\n[5/5] Adversarial (differencing attack)")
    if args.skip_adversarial:
        print("  --skip-adversarial set; skipping.")
    else:
        run_adversarial(
            data_dir=data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            quantifiers=quantifiers,
            adversarial_cfg=config["adversarial"],
            random_seed=random_seed,
            n_workers=args.adversarial_workers,
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
