import joblib

from sklearn.datasets import load_breast_cancer

from qube import QubeEngine

X, y = load_breast_cancer(return_X_y=True)

model = QubeEngine()

model.fit(X, y)

joblib.dump(model, "qube_model.pkl")

loaded = joblib.load("qube_model.pkl")

print(loaded.predict(X[:5]))