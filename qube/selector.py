import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_is_fitted

from .utils import _as_numpy, _score_distribution


class DescriptorBankSelector(BaseEstimator, TransformerMixin):
    """Learns a compact 256-feature descriptor bank from expanded descriptors."""

    def __init__(
        self,
        n_descriptors: int = 256,
        selector_trees: int = 400,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_descriptors = n_descriptors
        self.selector_trees = selector_trees
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        Xn = _as_numpy(X)
        y = np.asarray(y)
        mi = mutual_info_classif(Xn, y, random_state=self.random_state)
        forest = ExtraTreesClassifier(
            n_estimators=self.selector_trees,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight="balanced",
        )
        forest.fit(Xn, y)
        importance = forest.feature_importances_
        mi = np.nan_to_num(mi, nan=0.0)
        importance = np.nan_to_num(importance, nan=0.0)
        score = self._rank_normalize(mi) + self._rank_normalize(importance)
        keep = min(self.n_descriptors, Xn.shape[1])
        self.selected_indices_ = np.argsort(score)[::-1][:keep]
        self.mi_scores_ = mi
        self.importance_scores_ = importance
        self.scores_ = score
        return self

    def transform(self, X):
        check_is_fitted(self, ["selected_indices_"])
        return _as_numpy(X)[:, self.selected_indices_].astype(np.float32)

    @staticmethod
    def _rank_normalize(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.linspace(0.0, 1.0, len(values))
        return ranks

    def report(self, top_n: int = 20) -> dict:
        check_is_fitted(self, ["selected_indices_", "mi_scores_", "importance_scores_", "scores_"])
        selected = self.selected_indices_[:top_n]
        return {
            "mutual_information": _score_distribution(self.mi_scores_),
            "feature_importance": _score_distribution(self.importance_scores_),
            "top_selected_descriptors": [
                {
                    "rank": int(rank + 1),
                    "descriptor": f"descriptor_{int(idx)}",
                    "index": int(idx),
                    "combined_score": float(self.scores_[idx]),
                    "mutual_information": float(self.mi_scores_[idx]),
                    "feature_importance": float(self.importance_scores_[idx]),
                }
                for rank, idx in enumerate(selected)
            ],
        }

