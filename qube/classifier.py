import gc
import os
import pathlib
import json

import numpy as np
import pandas as pd

from dataclasses import asdict
from typing import Iterable

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from .config import Qube2Config
from .engine import GraphSpatialQuantumTransformer
from .selector import DescriptorBankSelector
from .utils import (
    _predict_scores,
    _evaluate_predictions,
)


class QubeEngine(BaseEstimator, ClassifierMixin):
    """End-to-end QUBE 2.0 representation plus Random Forest classifier."""

    def __init__(
        self,
        config: Qube2Config | None = None,
        feature_names: Iterable[str] | None = None,
        edge_list: pd.DataFrame | None = None,
        source_col: str = "source",
        target_col: str = "target",
        weight_col: str | None = None,
    ):
        self.config = config or Qube2Config()
        self.feature_names = None if feature_names is None else list(feature_names)
        self.edge_list = edge_list
        self.source_col = source_col
        self.target_col = target_col
        self.weight_col = weight_col

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.transformer_ = GraphSpatialQuantumTransformer(
            config=self.config,
            feature_names=self.feature_names,
            edge_list=self.edge_list,
            source_col=self.source_col,
            target_col=self.target_col,
            weight_col=self.weight_col,
        )
        self.transformer_.fit(X, y)
        expanded = None
        expanded_path = None
        try:
            if len(X) > max(1, int(self.config.batch_size)):
                expanded, expanded_path = self.transformer_.transform_to_memmap(X)
            else:
                expanded = self.transformer_.transform(X)

            self.n_expanded_features_ = expanded.shape[1]
            self.selector_ = DescriptorBankSelector(
                n_descriptors=self.config.n_descriptors,
                selector_trees=self.config.selector_trees,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )
            bank = self.selector_.fit_transform(expanded, y)
        finally:
            if expanded_path is not None:
                del expanded
                gc.collect()
                try:
                    os.remove(expanded_path)
                except OSError:
                    pass

        self.rf_ = RandomForestClassifier(
            n_estimators=self.config.rf_trees,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            class_weight="balanced",
            min_samples_leaf=1,
        )
        self.rf_.fit(bank, y)
        self.n_bank_features_ = bank.shape[1]
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, ["transformer_", "selector_"])
        n_samples = len(X)
        batch_size = max(1, int(self.config.batch_size))
        if n_samples == 0:
            expanded = self.transformer_._transform_batch(
                self.transformer_._slice_rows(X, 0, 0)
            )
            return self.selector_.transform(expanded)

        first_end = min(batch_size, n_samples)
        first_expanded = self.transformer_._transform_batch(
            self.transformer_._slice_rows(X, 0, first_end)
        )
        first_bank = self.selector_.transform(first_expanded)
        bank = np.empty((n_samples, first_bank.shape[1]), dtype=np.float32)
        bank[:first_end] = first_bank

        for start in range(first_end, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            expanded = self.transformer_._transform_batch(
                self.transformer_._slice_rows(X, start, end)
            )
            bank[start:end] = self.selector_.transform(expanded)

        return bank

    def predict(self, X):
        check_is_fitted(self, ["rf_"])
        return self.rf_.predict(self.transform(X))

    def predict_proba(self, X):
        check_is_fitted(self, ["rf_"])
        return self.rf_.predict_proba(self.transform(X))

    def report(self) -> dict:
        check_is_fitted(self, ["transformer_", "n_expanded_features_", "n_bank_features_"])
        return {
            "config": asdict(self.config),
            "graph": self.transformer_.describe_graph(),
            "expanded_features": int(self.n_expanded_features_),
            "descriptor_bank_features": int(self.n_bank_features_),
            "descriptor_quality": self.selector_.report(),
        }


def read_table(path: str) -> pd.DataFrame:
    file_path = pathlib.Path(path)
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)


def sanitize_feature_names(columns) -> list[str]:
    """Make dataframe feature names safe for XGBoost and similar estimators."""
    replacements = {
        "[": "_",
        "]": "_",
        "<": "_",
        ">": "_",
        " ": "_",
        "(": "_",
        ")": "_",
        "/": "_",
        "\\": "_",
        ":": "_",
        ";": "_",
        ",": "_",
    }
    seen: dict[str, int] = {}
    used: set[str] = set()
    sanitized: list[str] = []

    for column in columns:
        name = str(column)
        for old, new in replacements.items():
            name = name.replace(old, new)
        while "__" in name:
            name = name.replace("__", "_")
        name = name.strip("_") or "feature"

        base = name
        count = seen.get(base, 0) + 1
        candidate = base if count == 1 else f"{base}_{count}"
        while candidate in used:
            count += 1
            candidate = f"{base}_{count}"
        seen[base] = count
        used.add(candidate)
        sanitized.append(candidate)

    return sanitized


def load_dataset(
    path: str,
    target: str | None = None,
    drug: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    if drug:
        return load_ccle_gdsc_drug_dataset(path, drug)

    if target is None:
        raise ValueError("Either --target or --drug must be supplied.")

    df = read_table(path)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")

    y_raw = df[target]
    X_df = df.drop(columns=[target])
    numeric_cols = [col for col in X_df.columns if pd.api.types.is_numeric_dtype(X_df[col])]
    if not numeric_cols:
        raise ValueError("No numeric gene/features columns found.")
    X_df = X_df[numeric_cols]
    X_df.columns = sanitize_feature_names(X_df.columns)
    feature_names = X_df.columns.tolist()
    print("Sanitized feature names for ML compatibility.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    summary = {
        "mode": "target",
        "target": target,
        "samples": int(len(X_df)),
        "features": int(len(feature_names)),
        "classes": {
            str(label): int(count)
            for label, count in zip(*np.unique(y_raw.astype(str), return_counts=True))
        },
    }
    return X_df, y, feature_names, summary


def load_ccle_gdsc_drug_dataset(
    path: str,
    drug: str,
) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    file_path = pathlib.Path(path)
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        df = read_table(path)
    else:
        df = read_drug_rows_csv(path, drug)

    required = {"DRUG_NAME", "LN_IC50"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Drug mode requires columns {missing}.")

    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        drug_df = df.loc[df["DRUG_NAME"] == drug].copy()
    else:
        drug_df = df

    if drug_df.empty:
        raise ValueError(f"No samples found for drug: {drug}")

    drug_df["LN_IC50"] = pd.to_numeric(drug_df["LN_IC50"], errors="coerce")
    drug_df = drug_df.dropna(subset=["LN_IC50"]).copy()
    if drug_df.empty:
        raise ValueError(f"No non-missing LN_IC50 values found for drug '{drug}'.")

    median = drug_df["LN_IC50"].median()
    drug_df["label"] = (drug_df["LN_IC50"] < median).astype(int)
    y = drug_df["label"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError(
            f"Median split for drug '{drug}' produced one class only. "
            "Check whether LN_IC50 values are all identical."
        )

    candidate_cols = [
        col
        for col in drug_df.columns
        if col not in CCLE_GDSC_METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(drug_df[col])
    ]
    if not candidate_cols:
        raise ValueError(
            "No numeric gene-expression columns found after excluding CCLE/GDSC metadata."
        )

    X_df = drug_df[candidate_cols].astype(np.float32, copy=False)
    X_df.columns = sanitize_feature_names(X_df.columns)
    feature_names = X_df.columns.tolist()
    print("Sanitized feature names for ML compatibility.")
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    print(f"Filtered samples: {len(X_df)}")
    print(f"Genes: {len(feature_names)}")
    print("Memory-efficient loading complete.")
    summary = {
        "mode": "drug",
        "drug": drug,
        "samples": int(len(X_df)),
        "genes": int(len(feature_names)),
        "positive_class": positives,
        "negative_class": negatives,
        "median_ln_ic50": float(median),
    }
    return X_df, y, feature_names, summary


def read_drug_rows_csv(path: str, drug: str, chunksize: int = 50000) -> pd.DataFrame:
    print("Loading dataset in chunks...")
    filtered_chunks: list[pd.DataFrame] = []

    for chunk_number, chunk in enumerate(pd.read_csv(path, chunksize=chunksize), start=1):
        print(f"Chunk {chunk_number}...")
        required = {"DRUG_NAME", "LN_IC50"}
        missing = sorted(required - set(chunk.columns))
        if missing:
            raise ValueError(f"Drug mode requires columns {missing}.")

        chunk = chunk[chunk["DRUG_NAME"] == drug]
        if chunk.empty:
            continue
        filtered_chunks.append(chunk.copy())

    if not filtered_chunks:
        raise ValueError(f"No samples found for drug: {drug}")

    return pd.concat(filtered_chunks, axis=0, ignore_index=True, copy=False)


def print_dataset_summary(summary: dict) -> None:
    if summary["mode"] == "drug":
        print(f"Drug: {summary['drug']}")
        print(f"Samples: {summary['samples']}")
        print(f"Genes: {summary['genes']}")
        print(f"Positive class: {summary['positive_class']}")
        print(f"Negative class: {summary['negative_class']}")
        return

    print(f"Target: {summary['target']}")
    print(f"Samples: {summary['samples']}")
    print(f"Features: {summary['features']}")
    for label, count in summary["classes"].items():
        print(f"Class {label}: {count}")


def load_edge_list(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return read_table(path)


def count_drug_samples(path: str, chunksize: int = 50000) -> pd.Series:
    file_path = pathlib.Path(path)
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        df = read_table(path)
        if "DRUG_NAME" not in df.columns:
            raise ValueError("--all-drugs requires a DRUG_NAME column.")
        return df["DRUG_NAME"].dropna().astype(str).value_counts()

    counts = {}
    print("Counting samples per drug in chunks...")
    for chunk_number, chunk in enumerate(pd.read_csv(path, chunksize=chunksize), start=1):
        print(f"Count chunk {chunk_number}...")
        if "DRUG_NAME" not in chunk.columns:
            raise ValueError("--all-drugs requires a DRUG_NAME column.")
        chunk_counts = chunk["DRUG_NAME"].dropna().astype(str).value_counts()
        for drug, count in chunk_counts.items():
            counts[drug] = counts.get(drug, 0) + int(count)
    return pd.Series(counts, dtype=np.int64).sort_values(ascending=False)


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

    qube = Qube2RFClassifier(
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
