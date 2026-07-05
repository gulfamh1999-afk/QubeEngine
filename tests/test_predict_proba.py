from sklearn.datasets import load_breast_cancer

from qube import QubeEngine

def test_predict_proba():

    X, y = load_breast_cancer(return_X_y=True)

    model = QubeEngine()

    model.fit(X, y)

    proba = model.predict_proba(X)

    assert proba.shape[0] == len(X)
    assert proba.shape[1] == 2