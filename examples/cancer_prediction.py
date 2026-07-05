from qube import QubeEngine
import pandas as pd

df = pd.read_csv("breast_cancer.csv")

X = df.drop(columns=["label"])
y = df["label"]

model = QubeEngine()

model.fit(X, y)

print(model.score(X, y))