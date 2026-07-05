from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from qube import QubeEngine

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = QubeEngine()

model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
print("Prediction:", model.predict(X_test[:5]))
print("Probability:", model.predict_proba(X_test[:5]))