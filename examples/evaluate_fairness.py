from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*'where' used without 'out'.*")

import dill as pickle
import joblib
import numpy as np
import pandas as pd

from src.evaluation import (
    AdultFairnessEvaluator,
    TRECFairnessCorpusBuilder,
    TRECFairnessEvaluator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fairness estimation using quantification."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["adult", "trec"],
        required=True,
        help="Dataset to evaluate.",
    )

    # Adult args
    parser.add_argument("--dataset-id", type=str, default="adult")
    parser.add_argument("--classifier", type=str, default="lr", choices=["lr"])
    parser.add_argument("--n-prevalences", type=int, default=11)
    parser.add_argument("--max-prev", type=float, default=0.1)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=10)

    # TREC args
    parser.add_argument(
        "--build-corpus",
        action="store_true",
        help="Build TREC fairness corpus artefacts before evaluation.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Directory for the Whoosh BM25 index.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="BM25",
        help="Label attached to the TREC fairness experiment.",
    )
    return parser.parse_args()


def run_adult(args: argparse.Namespace, dirs: dict[str, Path]) -> None:
    data1 = pd.read_csv(dirs["data"] / f"{args.dataset_id}_D1.csv")
    data2 = pd.read_csv(dirs["data"] / f"{args.dataset_id}_D2.csv")
    data3 = pd.read_csv(dirs["data"] / f"{args.dataset_id}_D3.csv")

    evaluator = AdultFairnessEvaluator()

    report = evaluator.evaluate(
        data1=data1,
        data2=data2,
        data3=data3,
        classifier_name=args.classifier,
        n_prevalences=args.n_prevalences,
        max_prev=args.max_prev,
        sample_size=args.sample_size,
        repeats=args.repeats,
    )

    output_path = dirs["reports"] / f"{args.dataset_id}_fairness.pkl"
    evaluator.save_report(report, output_path)
    print(report.head())


def run_trec(args: argparse.Namespace, dirs: dict[str, Path]) -> None:
    if args.build_corpus:
        if args.index_dir is None:
            raise ValueError(
                "--index-dir is required when --build-corpus is used."
            )
        builder = TRECFairnessCorpusBuilder()
        train_df = pd.read_json(dirs["data"] / "trec_train.jsonl", lines=True)
        fair_train, nonrel_corpus = builder.build_train_nonrelevant_split(
            train_df
        )

        vectorizer = builder.vectorize(fair_train["text"].values)
        joblib.dump(vectorizer, dirs["data"] / "trec_fair_vectorizer.joblib")

        query_paths = sorted(dirs["data"].glob("trec_test_query_*.jsonl"))
        queries_df, rel_docs = builder.build_query_table(
            query_paths=query_paths,
            vectorizer=vectorizer,
        )
        docs_df = builder.build_docs_table(rel_docs, nonrel_corpus)

        builder.save_dataframe(
            queries_df, dirs["data"] / "trec_fair_queries.csv"
        )
        builder.save_dataframe(docs_df, dirs["data"] / "trec_fair_docs.csv")

        builder.build_bm25_index(docs_df, args.index_dir, overwrite=False)

        print(f"Saved TREC fairness queries to {"trec_fair_queries.csv"}")
        print(f"Saved TREC fairness docs to {"trec_fair_docs.csv"}")
        print(f"Saved TREC fairness index to {args.index_dir}")
    else:
        queries_df = pd.read_csv(dirs["data"] / "trec_fair_queries.csv")

        if not Path(dirs["data"] / "trec_fair_ranked_lists.pkl").exists():
            ranked_lists = TRECFairnessCorpusBuilder.rank_bm25_index(
                queries_df, args.index_dir, limit=10000
            )
        else:
            with open(dirs["data"] / "trec_fair_ranked_lists.pkl", "rb") as f:
                ranked_lists = pickle.load(f)

        docs_df = pd.read_csv(dirs["data"] / "trec_fair_docs.csv")
        vectorizer = joblib.load(dirs["data"] / "trec_fair_vectorizer.joblib")
        label_encoder = joblib.load(
            dirs["data"] / "label_encoder_trec_train.joblib"
        )

        evaluator = TRECFairnessEvaluator(quantifiers=["CC", "PCC"])
        results = evaluator.evaluate_ranked_lists(
            ranked_lists=ranked_lists,
            docs=docs_df,
            vectorizer=vectorizer,
            models_dir=dirs["models"],
            data_dir=dirs["data"],
            labels=label_encoder.classes_,
            experiment_name=args.experiment_name,
        )

        results_output = dirs["reports"] / "trec_fairness_results.csv"
        evaluator.save_report(results, results_output)

        rkl = evaluator.aggregate_rkl(results)
        rkl_output = dirs["reports"] / "trec_fairness_rkl.csv"
        evaluator.save_report(rkl, rkl_output)

        summary = evaluator.generate_summary_table(rkl)
        print(summary)


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dirs = {
        folder: project_root / folder
        for folder in ["data", "models", "reports"]
    }

    if args.dataset == "adult":
        run_adult(args, dirs)
    else:
        run_trec(args, dirs)


if __name__ == "__main__":
    main()
