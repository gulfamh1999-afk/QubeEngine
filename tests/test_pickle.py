import joblib

from sklearn.datasets import load_breast_cancer

from qube import QubeEngine

def test_pickle():

    X, y = load_breast_cancer(return_X_y=True)

    model = QubeEngine()

    model.fit(X, y)

    joblib.dump(model, "temp.pkl")

    loaded = joblib.load("temp.pkl")

    pred = loaded.predict(X)

    assert len(pred) == len(X)