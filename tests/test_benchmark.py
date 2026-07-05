from sklearn.datasets import load_breast_cancer

from qube.benchmark import benchmark_holdout
from qube.config import Qube2Config


def test_benchmark():

    X, y = load_breast_cancer(return_X_y=True)

    results = benchmark_holdout(
        X=X,
        y=y,
        feature_names=[f"F{i}" for i in range(X.shape[1])],
        config=Qube2Config(),
        edge_list=None,
        source_col="source",
        target_col="target",
        weight_col=None,
    )

    assert isinstance(results, dict)