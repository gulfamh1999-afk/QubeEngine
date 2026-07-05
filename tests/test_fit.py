from sklearn.datasets import load_breast_cancer

from qube import QubeEngine

def test_fit():

    X, y = load_breast_cancer(return_X_y=True)

    model = QubeEngine()

    model.fit(X, y)

    assert hasattr(model, "rf_")