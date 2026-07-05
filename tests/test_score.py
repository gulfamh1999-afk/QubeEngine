from sklearn.datasets import load_breast_cancer

from qube import QubeEngine

def test_score():

    X, y = load_breast_cancer(return_X_y=True)

    model = QubeEngine()

    model.fit(X, y)

    score = model.score(X, y)

    assert 0.0 <= score <= 1.0