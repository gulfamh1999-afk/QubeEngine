from sklearn.datasets import load_wine

from qube.engine import GraphSpatialQuantumTransformer

X, y = load_wine(return_X_y=True)

transformer = GraphSpatialQuantumTransformer()

descriptors = transformer.fit_transform(X, y)

print("Original shape:", X.shape)
print("Descriptor shape:", descriptors.shape)