import argparse
import json
import pathlib

from .config import Qube2Config

from .benchmark import (
    benchmark_cv,
    benchmark_holdout,
    run_all_drugs_benchmark,
)

from .data import (
    load_dataset,
    load_edge_list,
    print_dataset_summary,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QUBE 2.0 graph descriptor benchmark",
        epilog=(
            "Examples:\n"
            "  python qube2_graph_descriptor.py --data dataset.csv --target label\n"
            "  python qube2_graph_descriptor.py --data ccle_gdsc_merged.csv --drug AZD7762 --cv 5\n"
            "  python qube2_graph_descriptor.py --data ccle_gdsc_merged.csv --all-drugs --cv 5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="CSV/XLSX dataset path")
    parser.add_argument(
        "--target",
        default=None,
        help="Target label column. Optional when --drug is supplied.",
    )
    parser.add_argument(
        "--drug",
        default=None,
        help=(
            "Drug name for CCLE/GDSC merged data. Filters DRUG_NAME, drops missing "
            "LN_IC50, and creates label = LN_IC50 below median."
        ),
    )
    parser.add_argument(
        "--all-drugs",
        action="store_true",
        help="Benchmark every DRUG_NAME with at least 500 samples and save multi_drug_results.csv.",
    )
    parser.add_argument(
        "--max-drugs",
        type=int,
        default=None,
        help="Benchmark only the top N drugs ranked by sample count.",
    )
    parser.add_argument("--edge-list", default=None, help="Optional pathway/interactions CSV/XLSX")
    parser.add_argument("--source", default="source", help="Edge-list source gene column")
    parser.add_argument("--target-gene", default="target", help="Edge-list target gene column")
    parser.add_argument("--weight", default=None, help="Optional edge-list weight column")
    parser.add_argument("--n-descriptors", type=int, default=256)
    parser.add_argument("--graph-k", type=int, default=8)
    parser.add_argument("--spectral-dims", type=int, default=8)
    parser.add_argument("--quantum-channels", type=int, default=96)
    parser.add_argument("--rf-trees", type=int, default=600)
    parser.add_argument("--selector-trees", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--cv", type=int, default=0, help="Use stratified K-fold CV instead of holdout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-labels",
        action="store_true",
        help="Shuffle only training labels inside each fold as a leakage sanity check.",
    )
    parser.add_argument(
        "--random-features",
        action="store_true",
        help="Add Gaussian random features + RF control with the same bank width as QUBE2.",
    )
    parser.add_argument(
        "--validation-seeds",
        action="store_true",
        help="Repeat CV with seeds 42, 123, 456, 789, and 2026.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Qube2Config(
        n_descriptors=args.n_descriptors,
        graph_k=args.graph_k,
        spectral_dims=args.spectral_dims,
        quantum_channels=args.quantum_channels,
        rf_trees=args.rf_trees,
        selector_trees=args.selector_trees,
        batch_size=args.batch_size,
        random_state=args.seed,
    )
    edge_list = load_edge_list(args.edge_list)

    if args.all_drugs:
        result = run_all_drugs_benchmark(
            args=args,
            config=config,
            edge_list=edge_list,
        )
        text = json.dumps(result, indent=2)
        print(text)
        if args.output:
            pathlib.Path(args.output).write_text(text, encoding="utf-8")
        return

    X, y, feature_names, data_summary = load_dataset(
        args.data,
        target=args.target,
        drug=args.drug,
    )

    print_dataset_summary(data_summary)

    if args.cv and args.cv > 1:
        seeds = [42, 123, 456, 789, 2026] if args.validation_seeds else [args.seed]
        result = benchmark_cv(
            X=X,
            y=y,
            feature_names=feature_names,
            config=config,
            edge_list=edge_list,
            source_col=args.source,
            target_col=args.target_gene,
            weight_col=args.weight,
            folds=args.cv,
            shuffle_labels=args.shuffle_labels,
            random_features=args.random_features,
            seeds=seeds,
        )
    else:
        result = benchmark_holdout(
            X=X,
            y=y,
            feature_names=feature_names,
            config=config,
            edge_list=edge_list,
            source_col=args.source,
            target_col=args.target_gene,
            weight_col=args.weight,
            test_size=args.test_size,
            shuffle_labels=args.shuffle_labels,
            random_features=args.random_features,
        )

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        pathlib.Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
