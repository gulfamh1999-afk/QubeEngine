import math

import numpy as np
import pandas as pd

from typing import Iterable

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from .config import (
    EPS,
    Qube2Config,
    QUANTUM_BACKEND_MATHEMATICAL,
    QUANTUM_BACKEND_QISKIT_AER,
    QUANTUM_BACKEND_ALIASES,
)

def _as_numpy(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def normalize_quantum_backend(value: str | None) -> str:
    if value is None:
        return QUANTUM_BACKEND_MATHEMATICAL
    text = str(value).strip()
    if text in {QUANTUM_BACKEND_MATHEMATICAL, QUANTUM_BACKEND_QISKIT_AER}:
        return text
    normalized = text.lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    if normalized in QUANTUM_BACKEND_ALIASES:
        return QUANTUM_BACKEND_ALIASES[normalized]
    raise ValueError(
        "Quantum Backend must be one of: "
        f"{QUANTUM_BACKEND_MATHEMATICAL}, {QUANTUM_BACKEND_QISKIT_AER}."
    )


def _load_qiskit_aer_components():
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import ParameterVector
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit_aer import AerSimulator
    except Exception as exc:
        raise ImportError(
            "Qiskit Aer backend requested, but qiskit/qiskit-aer is not installed "
            "in the active Python environment. Install qiskit and qiskit-aer or "
            "switch Quantum Backend to Mathematical."
        ) from exc

    return QuantumCircuit, ParameterVector, ZZFeatureMap, AerSimulator, transpile


def _safe_auc(y_true, proba) -> float:
    labels = np.unique(y_true)
    if len(labels) == 2:
        return roc_auc_score(y_true, proba[:, 1])
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro")


def _entropy_from_values(values: np.ndarray, bins: int = 16) -> float:
    hist, _ = np.histogram(values, bins=bins)
    probs = hist.astype(np.float64) / max(hist.sum(), 1)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _class_counts(y: np.ndarray) -> dict:
    values, counts = np.unique(y, return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


def _score_distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def _confidence_interval(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    half_width = 1.96 * std / np.sqrt(arr.size) if arr.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - half_width),
        "ci95_high": float(mean + half_width),
    }


def _paired_t_test(values_a: Iterable[float], values_b: Iterable[float]) -> dict:
    a = np.asarray(list(values_a), dtype=np.float64)
    b = np.asarray(list(values_b), dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 2:
        return {"t_statistic": None, "p_value": None, "n": int(a.size)}

    diff = a - b
    std = diff.std(ddof=1)
    if std <= EPS:
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "n": int(a.size),
            "mean_difference": float(diff.mean()),
        }

    t_stat = float(diff.mean() / (std / np.sqrt(diff.size)))
    try:
        from scipy.stats import t as student_t

        p_value = float(2.0 * student_t.sf(abs(t_stat), df=diff.size - 1))
    except Exception:
        # Fallback when scipy is unavailable: normal approximation.
        p_value = float(math.erfc(abs(t_stat) / np.sqrt(2.0)))

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "n": int(diff.size),
        "mean_difference": float(diff.mean()),
    }


def _predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            probs = 1.0 / (1.0 + np.exp(-scores))
            return np.vstack([1.0 - probs, probs]).T
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), EPS)
    pred = model.predict(X)
    classes = getattr(model, "classes_", np.unique(pred))
    proba = np.zeros((len(pred), len(classes)), dtype=np.float64)
    for i, label in enumerate(pred):
        proba[i, np.where(classes == label)[0][0]] = 1.0
    return proba


def _evaluate_predictions(y_true, pred, proba) -> dict:
    labels = np.unique(y_true)
    average = "binary" if len(labels) == 2 else "macro"
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, pred, average=average, zero_division=0)),
        "roc_auc": float(_safe_auc(y_true, proba)),
    }

def _make_stronger_baselines(config: Qube2Config, seed: int) -> dict:
    models = {
        "rf": RandomForestClassifier(
            n_estimators=config.rf_trees,
            random_state=seed,
            n_jobs=config.n_jobs,
            class_weight="balanced",
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=config.selector_trees,
            random_state=seed,
            n_jobs=config.n_jobs,
            class_weight="balanced",
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=max(1, config.n_jobs),
        )
    except Exception as exc:
        models["xgboost_unavailable"] = str(exc)

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=config.n_jobs,
            class_weight="balanced",
            verbose=-1,
        )
    except Exception as exc:
        models["lightgbm_unavailable"] = str(exc)

    return models

def _metric_mean(result: dict, model: str, metric: str) -> float | None:
    value = result.get("aggregate", {}).get(model, {}).get(metric, {}).get("mean")
    return None if value is None else float(value)


def _format_optional(value: float | None) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.4f}"

