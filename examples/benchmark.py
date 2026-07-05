from sklearn.datasets import load_breast_cancer

from qube.benchmark import benchmark_holdout

X, y = load_breast_cancer(return_X_y=True)

results = benchmark_holdout(
    X=X,
    y=y,
    feature_names=[f"gene_{i}" for i in range(X.shape[1])],
)

print(results)