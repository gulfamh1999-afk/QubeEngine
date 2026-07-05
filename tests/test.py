import numpy as np
from qube import QubeEngine

X = np.random.rand(100, 20)
y = np.random.randint(0, 2, 100)

model = QubeEngine()

model.fit(X, y)

pred = model.predict(X)

prob = model.predict_proba(X)

print("Prediction shape:", pred.shape)
print("Probability shape:", prob.shape)
print("Score:", model.score(X, y))