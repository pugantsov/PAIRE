from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing fairness data and/or processed files.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Directory where per-model evaluation reports will be saved.",
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
        "--models-dir",
        type=Path,
        default=None,
        help="Directory containing TREC fairness quantifier models.",
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


def run_adult(args: argparse.Namespace) -> None:
    data1 = pd.read_csv(args.data_dir / f"{args.dataset_id}_D1.csv")
    data2 = pd.read_csv(args.data_dir / f"{args.dataset_id}_D2.csv")
    data3 = pd.read_csv(args.data_dir / f"{args.dataset_id}_D3.csv")

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

    output_path = args.reports_dir / f"{args.dataset_id}_fairness.pkl"
    evaluator.save_report(report, output_path)
    print(report.head())


def run_trec(args: argparse.Namespace) -> None:
    if args.build_trec_corpus:
        if args.index_dir is None:
            raise ValueError(
                "--index-dir is required when --build-trec-corpus is used."
            )

        builder = TRECFairnessCorpusBuilder()
        train_df = pd.read_json(args.data_dir / "trec_train.json", lines=True)
        fair_train, nonrel_corpus = builder.build_train_nonrelevant_split(
            train_df
        )

        vectorizer = builder.vectorize(fair_train["text"].values)
        joblib.dump(vectorizer, args.data_dir / "trec_fair_vectorizer.joblib")

        query_paths = sorted(args.data_dir.glob("trec_test_query_*.jsonl"))
        queries_df, rel_docs = builder.build_query_table(
            query_paths=query_paths,
            vectorizer=vectorizer,
        )
        docs_df = builder.build_docs_table(rel_docs, nonrel_corpus)

        builder.save_dataframe(queries_df, "trec_fair_queries.csv")
        builder.save_dataframe(docs_df, "trec_fair_docs.csv")

        ranked_lists = builder.build_bm25_ranked_lists(
            docs=docs_df,
            queries=queries_df,
            index_dir=args.index_dir,
        )
        ranked_lists_path = args.data_dir / "trec_fair_ranked_lists.pkl"
        builder.save_pickle(ranked_lists, ranked_lists_path)

        print(f"Saved TREC fairness queries to {"trec_fair_queries.csv"}")
        print(f"Saved TREC fairness docs to {"trec_fair_docs.csv"}")
        print(f"Saved TREC fairness ranked lists to {ranked_lists_path}")
    else:
        if not Path(args.data_dir / "trec_fair_ranked_lists.pkl").exists():
            raise FileNotFoundError(
                "TREC fairness ranked lists file not found. Run with --build-trec-corpus to build it."
            )

        with open(args.data_dir / "trec_fair_ranked_lists.pkl", "rb") as f:
            ranked_lists = pickle.load(f)

        docs_df = pd.read_csv(args.data_dir / "trec_fair_docs.csv")
        vectorizer = joblib.load(args.data_dir / "trec_fair_vectorizer.joblib")
        labels = np.load(
            args.data_dir / "trec_fair_labels.npy", allow_pickle=True
        )

        evaluator = TRECFairnessEvaluator()
        results = evaluator.evaluate_ranked_lists(
            ranked_lists=ranked_lists,
            docs=docs_df,
            vectorizer=vectorizer,
            models_dir=args.models_dir,
            labels=labels,
            experiment_name=args.experiment_name,
            model_pattern="trec_fair_*.pkl",
        )

        results_output = args.reports_dir / "trec_fairness_results.csv"
        evaluator.save_report(results, results_output)

        rkl = evaluator.aggregate_rkl(results)
        rkl_output = args.reports_dir / "trec_fairness_rkl.csv"
        evaluator.save_report(rkl, rkl_output)

        summary = evaluator.generate_summary_table(rkl)
        print(summary)


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "adult":
        run_adult(args)
    else:
        run_trec(args)


if __name__ == "__main__":
    main()
