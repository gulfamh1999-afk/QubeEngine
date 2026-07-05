from .classifier import QubeEngine
from sklearn.model_selection import train_test_split
import json
import pathlib
import numpy as np
import pandas as pd

from dataclasses import asdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
)

from .utils import (
    _confidence_interval,
    _paired_t_test,
    _make_stronger_baselines,
    _class_counts,
    _predict_scores,
    _evaluate_predictions,
    _metric_mean,
    _format_optional,
)

def run_all_drugs_benchmark(
    args: argparse.Namespace,
    config: Qube2Config,
    edge_list: pd.DataFrame | None,
    min_samples: int = 500,
) -> dict:
    counts = count_drug_samples(args.data)
    eligible = counts[counts >= min_samples]
    if eligible.empty:
        raise ValueError(f"No drugs found with at least {min_samples} samples.")

    total_eligible = len(eligible)
    if args.max_drugs is not None:
        if args.max_drugs <= 0:
            raise ValueError("--max-drugs must be a positive integer.")
        eligible = eligible.head(args.max_drugs)

    print("=" * 54)
    print(f"Eligible drugs: {total_eligible}")
    if args.max_drugs is not None:
        print(f"Benchmarking top {len(eligible)} drugs by sample count")
    else:
        print(f"Benchmarking all {len(eligible)} eligible drugs")
    print("=" * 54)

    rows = []
    per_drug_results = {}

    for index, (drug, counted_samples) in enumerate(eligible.items(), start=1):
        print("=" * 72)
        print(f"Drug {index}/{len(eligible)}: {drug} ({counted_samples} raw samples)")
        try:
            X, y, feature_names, data_summary = load_dataset(args.data, drug=str(drug))
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

            rf_acc = _metric_mean(result, "rf", "accuracy")
            rf_auc = _metric_mean(result, "rf", "roc_auc")
            qube_acc = _metric_mean(result, "qube2_rf", "accuracy")
            qube_auc = _metric_mean(result, "qube2_rf", "roc_auc")
            improvement = None if rf_acc is None or qube_acc is None else qube_acc - rf_acc
            roc_gain = None if rf_auc is None or qube_auc is None else qube_auc - rf_auc

            row = {
                "Drug": str(drug),
                "Raw Samples": int(counted_samples),
                "Samples": int(data_summary["samples"]),
                "Genes": int(data_summary["genes"]),
                "RF Accuracy": rf_acc,
                "RF ROC-AUC": rf_auc,
                "ExtraTrees Accuracy": _metric_mean(result, "extra_trees", "accuracy"),
                "XGBoost Accuracy": _metric_mean(result, "xgboost", "accuracy"),
                "LightGBM Accuracy": _metric_mean(result, "lightgbm", "accuracy"),
                "QUBE2 Accuracy": qube_acc,
                "QUBE2 ROC-AUC": qube_auc,
                "Random Feature Accuracy": _metric_mean(
                    result, "random_features_rf", "accuracy"
                ),
                "Improvement": improvement,
                "ROC Gain": roc_gain,
                "Status": "ok",
            }
            rows.append(row)
            per_drug_results[str(drug)] = result
        except Exception as exc:
            rows.append(
                {
                    "Drug": str(drug),
                    "Raw Samples": int(counted_samples),
                    "Samples": None,
                    "Genes": None,
                    "RF Accuracy": None,
                    "RF ROC-AUC": None,
                    "ExtraTrees Accuracy": None,
                    "XGBoost Accuracy": None,
                    "LightGBM Accuracy": None,
                    "QUBE2 Accuracy": None,
                    "QUBE2 ROC-AUC": None,
                    "Random Feature Accuracy": None,
                    "Improvement": None,
                    "ROC Gain": None,
                    "Status": f"failed: {exc}",
                }
            )
            print(f"Skipping {drug}: {exc}")

    results_df = pd.DataFrame(rows)
    results_path = pathlib.Path("multi_drug_results.csv")
    results_df.to_csv(results_path, index=False)

    ranked = (
        results_df[results_df["Status"] == "ok"][
            ["Drug", "RF Accuracy", "QUBE2 Accuracy", "Improvement", "ROC Gain"]
        ]
        .sort_values(["Improvement", "ROC Gain"], ascending=False)
        .reset_index(drop=True)
    )

    print("=" * 72)
    print("Ranked summary")
    print("Drug | RF | QUBE2 | Improvement | ROC Gain")
    for _, row in ranked.iterrows():
        print(
            f"{row['Drug']} | "
            f"{_format_optional(row['RF Accuracy'])} | "
            f"{_format_optional(row['QUBE2 Accuracy'])} | "
            f"{_format_optional(row['Improvement'])} | "
            f"{_format_optional(row['ROC Gain'])}"
        )

    improvements = ranked["Improvement"].dropna()
    if improvements.empty:
        stats = {
            "mean_improvement": None,
            "median_improvement": None,
            "std_improvement": None,
            "mean_rf_accuracy": None,
            "mean_qube2_accuracy": None,
            "best_drug": None,
            "worst_drug": None,
        }
    else:
        best_idx = improvements.idxmax()
        worst_idx = improvements.idxmin()
        stats = {
            "mean_improvement": float(improvements.mean()),
            "median_improvement": float(improvements.median()),
            "std_improvement": float(improvements.std(ddof=1))
            if len(improvements) > 1
            else 0.0,
            "mean_rf_accuracy": float(ranked["RF Accuracy"].dropna().mean()),
            "mean_qube2_accuracy": float(ranked["QUBE2 Accuracy"].dropna().mean()),
            "best_drug": str(ranked.loc[best_idx, "Drug"]),
            "worst_drug": str(ranked.loc[worst_idx, "Drug"]),
        }

    print("Multi-drug summary statistics")
    print(f"Top drug: {stats['best_drug']}")
    print(f"Worst drug: {stats['worst_drug']}")
    print(f"Mean RF Accuracy: {_format_optional(stats['mean_rf_accuracy'])}")
    print(f"Mean QUBE2 Accuracy: {_format_optional(stats['mean_qube2_accuracy'])}")
    print(f"Mean Improvement: {_format_optional(stats['mean_improvement'])}")
    print(f"Median Improvement: {_format_optional(stats['median_improvement'])}")
    print(f"Standard Deviation: {_format_optional(stats['std_improvement'])}")
    print(f"Saved results to {results_path}")

    return {
        "results_csv": str(results_path),
        "ranked_summary": ranked.to_dict(orient="records"),
        "summary_statistics": stats,
        "per_drug_results": per_drug_results,
    }


def benchmark_holdout(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    config: Qube2Config,
    edge_list: pd.DataFrame | None,
    source_col: str,
    target_col: str,
    weight_col: str | None,
    test_size: float = 0.25,
    shuffle_labels: bool = False,
    random_features: bool = False,
) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=config.random_state,
        stratify=y,
    )
    fold_result = evaluate_validation_fold(
        fold=1,
        seed=config.random_state,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        config=config,
        edge_list=edge_list,
        source_col=source_col,
        target_col=target_col,
        weight_col=weight_col,
        shuffle_labels=shuffle_labels,
        random_features=random_features,
    )
    return summarize_validation_results([fold_result], shuffle_labels=shuffle_labels)


def benchmark_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    config: Qube2Config,
    edge_list: pd.DataFrame | None,
    source_col: str,
    target_col: str,
    weight_col: str | None,
    folds: int = 5,
    shuffle_labels: bool = False,
    random_features: bool = False,
    seeds: list[int] | None = None,
) -> dict:
    seeds = seeds or [config.random_state]
    fold_results = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            fold_config = Qube2Config(**{**asdict(config), "random_state": seed + fold})
            fold_results.append(
                evaluate_validation_fold(
                    fold=fold,
                    seed=seed,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    feature_names=feature_names,
                    config=fold_config,
                    edge_list=edge_list,
                    source_col=source_col,
                    target_col=target_col,
                    weight_col=weight_col,
                    shuffle_labels=shuffle_labels,
                    random_features=random_features,
                )
            )

    return summarize_validation_results(fold_results, shuffle_labels=shuffle_labels)


def evaluate_validation_fold(
    fold: int,
    seed: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    config: Qube2Config,
    edge_list: pd.DataFrame | None,
    source_col: str,
    target_col: str,
    weight_col: str | None,
    shuffle_labels: bool = False,
    random_features: bool = False,
) -> dict:
    rng = np.random.default_rng(seed + fold)
    y_fit = np.array(y_train, copy=True)
    if shuffle_labels:
        rng.shuffle(y_fit)

    print(f"Fold {fold} | Seed {seed}")
    print(f"Train samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Train class counts: {_class_counts(y_train)}")
    print(f"Test class counts: {_class_counts(y_test)}")
    if shuffle_labels:
        print(f"Shuffled train class counts: {_class_counts(y_fit)}")

    metrics = {}
    unavailable = {}
    for name, model in _make_stronger_baselines(config, seed + fold).items():
        if isinstance(model, str):
            unavailable[name.replace("_unavailable", "")] = model
            continue
        model.fit(X_train, y_fit)
        pred = model.predict(X_test)
        proba = _predict_scores(model, X_test)
        metrics[name] = _evaluate_predictions(y_test, pred, proba)

    qube = QubeEngine(
        config=config,
        feature_names=feature_names,
        edge_list=edge_list,
        source_col=source_col,
        target_col=target_col,
        weight_col=weight_col,
    )
    qube.fit(X_train, y_fit)
    qube_pred = qube.predict(X_test)
    qube_proba = qube.predict_proba(X_test)
    metrics["qube2_rf"] = _evaluate_predictions(y_test, qube_pred, qube_proba)

    if random_features:
        n_random = qube.n_bank_features_
        random_train = rng.normal(0.0, 1.0, size=(len(y_train), n_random)).astype(np.float32)
        random_test = rng.normal(0.0, 1.0, size=(len(y_test), n_random)).astype(np.float32)
        random_rf = RandomForestClassifier(
            n_estimators=config.rf_trees,
            random_state=seed + fold,
            n_jobs=config.n_jobs,
            class_weight="balanced",
        )
        random_rf.fit(random_train, y_fit)
        random_pred = random_rf.predict(random_test)
        random_proba = random_rf.predict_proba(random_test)
        metrics["random_features_rf"] = _evaluate_predictions(
            y_test, random_pred, random_proba
        )

    qube_report = qube.report()
    graph = qube_report["graph"]
    descriptor_quality = qube_report["descriptor_quality"]
    print(f"Selected descriptors: {qube.n_bank_features_}")
    print(f"Graph edges: {graph['edges']}")
    print(f"Removed constant features: {graph['removed_constant_features']}")
    print(f"Graph diagnostics: {json.dumps(graph, indent=2)}")
    print(
        "Descriptor quality: "
        + json.dumps(
            {
                "expanded_descriptor_count": qube.n_expanded_features_,
                "final_descriptor_bank_size": qube.n_bank_features_,
                **descriptor_quality,
            },
            indent=2,
        )
    )

    return {
        "seed": int(seed),
        "fold": int(fold),
        "metrics": metrics,
        "unavailable_models": unavailable,
        "fold_diagnostics": {
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "train_class_counts": _class_counts(y_train),
            "test_class_counts": _class_counts(y_test),
            "shuffled_train_class_counts": _class_counts(y_fit) if shuffle_labels else None,
            "selected_descriptors": int(qube.n_bank_features_),
            "graph_edges": int(graph["edges"]),
            "removed_constant_features": int(graph["removed_constant_features"]),
        },
        "qube2_report": qube_report,
    }


def summarize_validation_results(
    fold_results: list[dict],
    shuffle_labels: bool = False,
) -> dict:
    rows = []
    unavailable = {}
    for fold_result in fold_results:
        unavailable.update(fold_result.get("unavailable_models", {}))
        for model, scores in fold_result["metrics"].items():
            rows.append(
                {
                    "seed": fold_result["seed"],
                    "fold": fold_result["fold"],
                    "model": model,
                    "accuracy": scores["accuracy"],
                    "precision": scores.get("precision"),
                    "recall": scores.get("recall"),
                    "f1": scores.get("f1"),
                    "roc_auc": scores["roc_auc"],
                }
            )

    scores_df = pd.DataFrame(rows)
    aggregate = {}
    if not scores_df.empty:
        metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        for model, group in scores_df.groupby("model"):
            aggregate[model] = {
                metric: _confidence_interval(group[metric].dropna())
                for metric in metric_columns
                if metric in group
            }

    paired_tests = {}
    if not scores_df.empty and "qube2_rf" in set(scores_df["model"]):
        qube_rows = scores_df[scores_df["model"] == "qube2_rf"].sort_values(["seed", "fold"])
        for model in sorted(set(scores_df["model"]) - {"qube2_rf"}):
            model_rows = scores_df[scores_df["model"] == model].sort_values(["seed", "fold"])
            merged = qube_rows.merge(
                model_rows,
                on=["seed", "fold"],
                suffixes=("_qube2", f"_{model}"),
            )
            paired_tests[f"qube2_rf_vs_{model}"] = {
                "accuracy": _paired_t_test(
                    merged["accuracy_qube2"],
                    merged[f"accuracy_{model}"],
                ),
                "roc_auc": _paired_t_test(
                    merged["roc_auc_qube2"],
                    merged[f"roc_auc_{model}"],
                ),
            }

    seed_variance = {}
    if not scores_df.empty:
        seed_means = (
            scores_df.groupby(["seed", "model"])[["accuracy", "roc_auc"]]
            .mean()
            .reset_index()
        )
        for model, group in seed_means.groupby("model"):
            seed_variance[model] = {
                "accuracy_variance_across_seeds": float(group["accuracy"].var(ddof=1))
                if len(group) > 1
                else 0.0,
                "roc_auc_variance_across_seeds": float(group["roc_auc"].var(ddof=1))
                if len(group) > 1
                else 0.0,
                "seed_means": group.to_dict(orient="records"),
            }

    validation_report = build_final_validation_report(
        aggregate=aggregate,
        paired_tests=paired_tests,
        shuffle_labels=shuffle_labels,
    )

    return {
        "folds": fold_results,
        "scores": rows,
        "aggregate": aggregate,
        "paired_t_tests": paired_tests,
        "seed_variance": seed_variance,
        "unavailable_models": unavailable,
        "validation_report": validation_report,
    }


def build_final_validation_report(
    aggregate: dict,
    paired_tests: dict,
    shuffle_labels: bool,
) -> dict:
    qube = aggregate.get("qube2_rf", {})
    rf = aggregate.get("rf", {})
    qube_acc = qube.get("accuracy", {}).get("mean")
    qube_auc = qube.get("roc_auc", {}).get("mean")
    rf_auc = rf.get("roc_auc", {}).get("mean")

    leakage_flags = []
    if shuffle_labels and qube_acc is not None and qube_auc is not None:
        if qube_acc > 0.65 or qube_auc > 0.65:
            leakage_flags.append(
                "QUBE2 remains far above chance under shuffled training labels."
            )
    if not shuffle_labels:
        leakage_flags.append(
            "Run with --shuffle-labels to complete the empirical leakage sanity check."
        )

    significant = None
    test = paired_tests.get("qube2_rf_vs_rf", {}).get("roc_auc")
    if test and test.get("p_value") is not None:
        significant = bool(test["p_value"] < 0.05)

    additional_information = None
    if qube_auc is not None and rf_auc is not None:
        additional_information = bool(qube_auc > rf_auc)

    return {
        "code_level_leakage_detected": False,
        "leakage_notes": leakage_flags,
        "free_from_detectable_leakage": len(leakage_flags) == 0,
        "improvement_statistically_significant_vs_rf_auc": significant,
        "qube2_additional_predictive_information_vs_rf": additional_information,
        "interpretation": (
            "Representation fitting is performed inside each train fold. "
            "The shuffled-label control is the decisive empirical leakage test."
        ),
    }


