import math
import numpy as np
import pandas as pd
import networkx as nx

from typing import Iterable
from .config import EPS

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
    MinMaxScaler,
)

from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

from .config import (
    Qube2Config,
    QUANTUM_BACKEND_MATHEMATICAL,
    QUANTUM_BACKEND_QISKIT_AER,
)

from .utils import (
    _as_numpy,
    normalize_quantum_backend,
    _load_qiskit_aer_components,
    _entropy_from_values,
)

from typing import Iterable

class InteractionGraphBuilder:
    """Builds a weighted gene graph from priors or from training data."""

    def __init__(
        self,
        k: int = 8,
        min_abs_corr: float = 0.05,
        random_state: int = 42,
    ):
        self.k = k
        self.min_abs_corr = min_abs_corr
        self.random_state = random_state

    def fit(
        self,
        X,
        feature_names: Iterable[str] | None = None,
        edge_list: pd.DataFrame | None = None,
        source_col: str = "source",
        target_col: str = "target",
        weight_col: str | None = None,
    ) -> "InteractionGraphBuilder":
        Xn = _as_numpy(X)
        n_features = Xn.shape[1]
        self.feature_names_ = list(feature_names or [f"gene_{i}" for i in range(n_features)])
        self.name_to_idx_ = {name: i for i, name in enumerate(self.feature_names_)}

        if edge_list is not None:
            adjacency = self._graph_from_edge_list(
                edge_list=edge_list,
                n_features=n_features,
                source_col=source_col,
                target_col=target_col,
                weight_col=weight_col,
            )
            if np.count_nonzero(adjacency) == 0:
                raise ValueError("The supplied edge-list did not match any feature names.")
        else:
            adjacency = self._graph_from_correlations(Xn)

        adjacency = np.maximum(adjacency, adjacency.T)
        np.fill_diagonal(adjacency, 0.0)

        self.adjacency_ = adjacency
        self.normalized_adjacency_ = self._normalize_with_self_loops(self.adjacency_)
        self.laplacian_ = np.eye(n_features, dtype=np.float64) - self.normalized_adjacency_
        self.spectral_coordinates_ = self._spectral_coordinates(self.laplacian_)
        self.degree_ = self.adjacency_.sum(axis=1)
        self.weighted_clustering_ = self._weighted_clustering(self.adjacency_)
        self.connected_components_ = self._connected_components(self.adjacency_)
        return self

    def _graph_from_edge_list(
        self,
        edge_list: pd.DataFrame,
        n_features: int,
        source_col: str,
        target_col: str,
        weight_col: str | None,
    ) -> np.ndarray:
        self.removed_constant_features_ = 0
        adjacency = np.zeros((n_features, n_features), dtype=np.float64)
        for _, row in edge_list.iterrows():
            src = row[source_col]
            dst = row[target_col]
            if src not in self.name_to_idx_ or dst not in self.name_to_idx_:
                continue
            weight = 1.0 if weight_col is None else float(row[weight_col])
            if not np.isfinite(weight) or weight <= 0:
                continue
            adjacency[self.name_to_idx_[src], self.name_to_idx_[dst]] = weight
        return adjacency

    def _graph_from_correlations(self, X: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        variances = np.nanvar(X, axis=0)
        active_mask = np.isfinite(variances) & (variances > EPS)
        active_indices = np.flatnonzero(active_mask)
        self.removed_constant_features_ = int(n_features - len(active_indices))
        adjacency = np.zeros((n_features, n_features), dtype=np.float64)

        if len(active_indices) < 2:
            # Correlation is undefined with fewer than two non-constant columns.
            for i in range(n_features):
                adjacency[i, (i + 1) % n_features] = 0.1
            return adjacency

        X_active = X[:, active_indices]
        corr = np.corrcoef(X_active, rowvar=False)
        corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)

        for active_i, original_i in enumerate(active_indices):
            order = np.argsort(corr[active_i])[::-1]
            picked = 0
            for active_j in order:
                if active_i == active_j or corr[active_i, active_j] < self.min_abs_corr:
                    continue
                original_j = active_indices[active_j]
                adjacency[original_i, original_j] = float(corr[active_i, active_j])
                picked += 1
                if picked >= self.k:
                    break

        if np.count_nonzero(adjacency) == 0:
            # Fallback for tiny or nearly independent datasets: keep a weak ring.
            for i in range(n_features):
                adjacency[i, (i + 1) % n_features] = 0.1

        return adjacency

    @staticmethod
    def _normalize_with_self_loops(adjacency: np.ndarray) -> np.ndarray:
        graph = adjacency + np.eye(adjacency.shape[0], dtype=np.float64)
        degree = graph.sum(axis=1)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, EPS))
        return inv_sqrt[:, None] * graph * inv_sqrt[None, :]

    def _spectral_coordinates(self, laplacian: np.ndarray) -> np.ndarray:
        n = laplacian.shape[0]
        dims = min(max(1, getattr(self, "spectral_dims_", 8)), n)
        vals, vecs = np.linalg.eigh(laplacian)
        order = np.argsort(vals)
        # Skip the first trivial eigenvector when possible.
        start = 1 if n > 1 else 0
        chosen = order[start : start + dims]
        coords = vecs[:, chosen]
        if coords.shape[1] < dims:
            coords = np.pad(coords, ((0, 0), (0, dims - coords.shape[1])))
        return coords.astype(np.float32)

    @staticmethod
    def _weighted_clustering(adjacency: np.ndarray) -> np.ndarray:
        A = adjacency
        n = A.shape[0]
        clustering = np.zeros(n, dtype=np.float64)
        binary = A > 0
        for i in range(n):
            neighbors = np.flatnonzero(binary[i])
            k = len(neighbors)
            if k < 2:
                continue
            sub = A[np.ix_(neighbors, neighbors)]
            triangles = sub.sum() / 2.0
            strength = max(A[i, neighbors].sum(), EPS)
            clustering[i] = triangles / (strength * (k - 1) + EPS)
        return np.clip(clustering, 0.0, 1.0)

    @staticmethod
    def _connected_components(adjacency: np.ndarray) -> int:
        n = adjacency.shape[0]
        if n == 0:
            return 0
        neighbors = adjacency > 0
        visited = np.zeros(n, dtype=bool)
        components = 0
        for start in range(n):
            if visited[start]:
                continue
            components += 1
            stack = [start]
            visited[start] = True
            while stack:
                node = stack.pop()
                for nxt in np.flatnonzero(neighbors[node]):
                    if not visited[nxt]:
                        visited[nxt] = True
                        stack.append(nxt)
        return components


class GraphSpatialQuantumTransformer(BaseEstimator, TransformerMixin):
    """Creates an expanded descriptor matrix from gene expression features."""

    def __init__(
        self,
        config: Qube2Config | None = None,
        feature_names: Iterable[str] | None = None,
        edge_list: pd.DataFrame | None = None,
        source_col: str = "source",
        target_col: str = "target",
        weight_col: str | None = None,
    ):
        self.config = config or Qube2Config()
        self.feature_names = None if feature_names is None else list(feature_names)
        self.edge_list = edge_list
        self.source_col = source_col
        self.target_col = target_col
        self.weight_col = weight_col

    def fit(self, X, y=None):
        self.config.quantum_backend = normalize_quantum_backend(self.config.quantum_backend)
        Xn = _as_numpy(X)
        self.imputer_ = SimpleImputer(strategy="median")
        Xi = self.imputer_.fit_transform(Xn)
        self.scaler_ = RobustScaler()
        Xs = self.scaler_.fit_transform(Xi)

        self.graph_ = InteractionGraphBuilder(
            k=self.config.graph_k,
            min_abs_corr=self.config.graph_min_abs_corr,
            random_state=self.config.random_state,
        )
        self.graph_.spectral_dims_ = self.config.spectral_dims
        self.graph_.fit(
            Xs,
            feature_names=self.feature_names,
            edge_list=self.edge_list,
            source_col=self.source_col,
            target_col=self.target_col,
            weight_col=self.weight_col,
        )

        rng = np.random.default_rng(self.config.random_state)
        n_features = Xs.shape[1]
        n_channels = self.config.quantum_channels

        self.quantum_gene_weights_ = rng.normal(0.0, 1.0, size=(n_features, n_channels))
        self.quantum_phase_ = rng.uniform(0.0, 2.0 * np.pi, size=n_channels)

        edges = np.vstack(np.nonzero(self.graph_.adjacency_)).T
        if len(edges) == 0:
            edges = np.array([[i, (i + 1) % n_features] for i in range(n_features)])
        sample_count = min(self.config.quantum_pair_samples, len(edges))
        chosen = rng.choice(len(edges), size=sample_count, replace=len(edges) < sample_count)
        self.quantum_edges_ = edges[chosen]
        self.quantum_edge_weights_ = rng.normal(0.0, 1.0, size=sample_count)
        self.quantum_output_width_ = 3 * n_channels + 2 * sample_count + 3
        if self.config.quantum_backend == QUANTUM_BACKEND_QISKIT_AER:
            self._setup_qiskit_aer(n_features=n_features, rng=rng)
        return self

    def transform(self, X):
        check_is_fitted(self, ["graph_", "imputer_", "scaler_"])
        n_samples = len(X)
        batch_size = max(1, int(self.config.batch_size))
        if n_samples == 0:
            return self._transform_batch(self._slice_rows(X, 0, 0))

        first_end = min(batch_size, n_samples)
        first_batch = self._transform_batch(self._slice_rows(X, 0, first_end))
        descriptors = np.empty((n_samples, first_batch.shape[1]), dtype=np.float32)
        descriptors[:first_end] = first_batch

        for start in range(first_end, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            descriptors[start:end] = self._transform_batch(self._slice_rows(X, start, end))

        return descriptors

    def transform_to_memmap(self, X) -> tuple[np.memmap, str]:
        check_is_fitted(self, ["graph_", "imputer_", "scaler_"])
        n_samples = len(X)
        batch_size = max(1, int(self.config.batch_size))

        first_end = min(batch_size, n_samples)
        first_batch = self._transform_batch(self._slice_rows(X, 0, first_end))

        temp = tempfile.NamedTemporaryFile(
            prefix="qube2_descriptors_",
            suffix=".dat",
            delete=False,
        )
        temp_path = temp.name
        temp.close()

        descriptors = np.memmap(
            temp_path,
            dtype=np.float64,
            mode="w+",
            shape=(n_samples, first_batch.shape[1]),
        )
        descriptors[:first_end] = first_batch.astype(np.float64, copy=False)

        for start in range(first_end, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = self._transform_batch(self._slice_rows(X, start, end))
            descriptors[start:end] = batch.astype(np.float64, copy=False)

        descriptors.flush()
        return descriptors, temp_path

    @staticmethod
    def _slice_rows(X, start: int, end: int) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.iloc[start:end].to_numpy(dtype=np.float64)
        return np.asarray(X[start:end], dtype=np.float64)

    def _transform_batch(self, X_batch: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(self.imputer_.transform(X_batch))

        spatial = self._spatial_bank(Xs)
        topology = self._topology_bank(Xs)
        spectral = self._spectral_bank(Xs)
        quantum = self._quantum_bank(Xs, spatial)

        descriptors = np.hstack([Xs, spatial, topology, spectral, quantum])
        descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
        return descriptors.astype(np.float32)

    def _diffusions(self, Xs: np.ndarray) -> dict[int, np.ndarray]:
        S = self.graph_.normalized_adjacency_
        diffused: dict[int, np.ndarray] = {}
        current = Xs
        max_scale = max(self.config.diffusion_scales)
        for scale in range(max_scale + 1):
            if scale in self.config.diffusion_scales:
                diffused[scale] = current
            current = current @ S
        return diffused

    def _spatial_bank(self, Xs: np.ndarray) -> np.ndarray:
        diffused = self._diffusions(Xs)
        chunks: list[np.ndarray] = []
        for scale in self.config.diffusion_scales:
            H = diffused[scale]
            chunks.append(H)
            chunks.append(np.abs(Xs - H))
            chunks.append(Xs * H)
        return np.hstack(chunks)

    def _topology_bank(self, Xs: np.ndarray) -> np.ndarray:
        degree = self.graph_.degree_
        clustering = self.graph_.weighted_clustering_
        degree_norm = degree / max(degree.max(), EPS)
        topo = np.vstack([degree_norm, clustering]).T
        weighted = Xs[:, :, None] * topo[None, :, :]
        per_sample = [
            weighted.mean(axis=1),
            weighted.std(axis=1),
            weighted.max(axis=1),
            weighted.min(axis=1),
        ]

        abs_x = np.abs(Xs)
        topological_energy = [
            abs_x @ degree_norm[:, None],
            abs_x @ clustering[:, None],
            (Xs * Xs) @ degree_norm[:, None],
            (Xs * Xs) @ clustering[:, None],
        ]
        return np.hstack(per_sample + topological_energy)

    def _spectral_bank(self, Xs: np.ndarray) -> np.ndarray:
        coords = self.graph_.spectral_coordinates_
        projections = Xs @ coords
        abs_projections = np.abs(Xs) @ np.abs(coords)
        return np.hstack(
            [
                projections,
                np.sin(projections),
                np.cos(projections),
                projections * projections,
                abs_projections,
            ]
        )

    def _quantum_bank(self, Xs: np.ndarray, spatial: np.ndarray) -> np.ndarray:
        if self.config.quantum_backend == QUANTUM_BACKEND_QISKIT_AER:
            return self._qiskit_aer_quantum_bank(Xs)
        return self._mathematical_quantum_bank(Xs, spatial)

    def _mathematical_quantum_bank(self, Xs: np.ndarray, spatial: np.ndarray) -> np.ndarray:
        # Quantum-inspired feature map retained as the Mathematical backend.
        weighted_projection = Xs @ self.quantum_gene_weights_
        phase = weighted_projection + self.quantum_phase_
        amplitudes = np.hstack([np.sin(phase), np.cos(phase), np.sin(0.5 * phase) ** 2])

        i = self.quantum_edges_[:, 0]
        j = self.quantum_edges_[:, 1]
        pair_phase = (
            Xs[:, i] * Xs[:, j] * self.quantum_edge_weights_[None, :]
            + Xs[:, i]
            - Xs[:, j]
        )
        entangled = np.hstack([np.sin(pair_phase), np.cos(pair_phase)])

        field = np.hstack([amplitudes, entangled, spatial[:, : min(spatial.shape[1], 256)]])
        probs = np.abs(field)
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), EPS)
        purity = (probs * probs).sum(axis=1, keepdims=True)
        entropy = -(probs * np.log2(np.maximum(probs, EPS))).sum(axis=1, keepdims=True)
        l2 = np.sqrt((field * field).mean(axis=1, keepdims=True))
        return np.hstack([amplitudes, entangled, purity, entropy, l2])

    def _setup_qiskit_aer(self, n_features: int, rng: np.random.Generator) -> None:
        QuantumCircuit, ParameterVector, ZZFeatureMap, AerSimulator, transpile = (
            _load_qiskit_aer_components()
        )
        n_qubits = max(1, int(self.config.qiskit_qubits))
        reps = max(1, int(self.config.qiskit_reps))

        parameters = ParameterVector("x", length=n_qubits)
        feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=reps)
        feature_parameters = sorted(feature_map.parameters, key=lambda param: param.name)
        feature_map = feature_map.assign_parameters(
            {param: parameters[i] for i, param in enumerate(feature_parameters)},
            inplace=False,
        )

        circuit = QuantumCircuit(n_qubits)
        circuit.compose(feature_map, inplace=True)
        circuit.save_statevector()

        backend = AerSimulator(
            method="statevector",
            seed_simulator=int(self.config.random_state),
        )
        self.qiskit_parameter_vector_ = parameters
        self.qiskit_backend_ = backend
        self.qiskit_circuit_ = transpile(
            circuit,
            backend,
            seed_transpiler=int(self.config.random_state),
            optimization_level=1,
        )
        self.qiskit_projection_weights_ = rng.normal(0.0, 1.0, size=(n_features, n_qubits))
        self.qiskit_parameter_phase_ = rng.uniform(-np.pi, np.pi, size=n_qubits)
        self.qiskit_z_signs_ = self._qiskit_z_signs(n_qubits)

    @staticmethod
    def _qiskit_z_signs(n_qubits: int) -> np.ndarray:
        states = np.arange(2**n_qubits)
        signs = []
        for qubit in range(n_qubits):
            bit = (states >> qubit) & 1
            signs.append(np.where(bit == 0, 1.0, -1.0))
        return np.asarray(signs, dtype=np.float64)

    def _qiskit_parameter_values(self, Xs: np.ndarray) -> np.ndarray:
        raw = Xs @ self.qiskit_projection_weights_ + self.qiskit_parameter_phase_
        return np.pi * (0.5 + 0.5 * np.tanh(raw))

    def _qiskit_aer_quantum_bank(self, Xs: np.ndarray) -> np.ndarray:
        check_is_fitted(
            self,
            [
                "qiskit_backend_",
                "qiskit_circuit_",
                "qiskit_parameter_vector_",
                "qiskit_projection_weights_",
            ],
        )
        target_signal_width = max(1, int(self.quantum_output_width_) - 3)
        if Xs.shape[0] == 0:
            return np.zeros((0, self.quantum_output_width_), dtype=np.float64)

        parameter_rows = self._qiskit_parameter_values(Xs)
        signal_rows = []
        for values in parameter_rows:
            binding = {
                self.qiskit_parameter_vector_[i]: float(values[i])
                for i in range(len(self.qiskit_parameter_vector_))
            }
            circuit = self.qiskit_circuit_.assign_parameters(binding, inplace=False)
            result = self.qiskit_backend_.run(circuit).result()
            statevector = np.asarray(result.get_statevector(), dtype=np.complex128)
            signal_rows.append(
                self._statevector_signal_features(
                    statevector=statevector,
                    parameters=values,
                    width=target_signal_width,
                )
            )

        signal = np.vstack(signal_rows)
        probs = np.abs(signal)
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), EPS)
        purity = (probs * probs).sum(axis=1, keepdims=True)
        entropy = -(probs * np.log2(np.maximum(probs, EPS))).sum(axis=1, keepdims=True)
        l2 = np.sqrt((signal * signal).mean(axis=1, keepdims=True))
        return np.hstack([signal, purity, entropy, l2])

    def _statevector_signal_features(
        self,
        statevector: np.ndarray,
        parameters: np.ndarray,
        width: int,
    ) -> np.ndarray:
        probabilities = np.abs(statevector) ** 2
        probabilities = probabilities / max(float(probabilities.sum()), EPS)
        amplitudes = np.abs(statevector)
        z_expectations = self.qiskit_z_signs_ @ probabilities

        zz_expectations = []
        for i in range(len(z_expectations)):
            for j in range(i + 1, len(z_expectations)):
                zz_expectations.append(float(z_expectations[i] * z_expectations[j]))

        base = np.concatenate(
            [
                probabilities,
                np.sqrt(probabilities),
                statevector.real,
                statevector.imag,
                amplitudes,
                z_expectations,
                np.asarray(zz_expectations, dtype=np.float64),
                np.sin(parameters),
                np.cos(parameters),
                parameters,
                np.asarray(
                    [
                        probabilities.max(),
                        probabilities.min(),
                        probabilities.mean(),
                        probabilities.std(),
                        float(-(probabilities * np.log2(np.maximum(probabilities, EPS))).sum()),
                        float((probabilities * probabilities).sum()),
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        if base.size >= width:
            return base[:width]
        repeats = int(np.ceil(width / max(base.size, 1)))
        return np.tile(base, repeats)[:width]
    def describe_graph(self) -> dict:
        check_is_fitted(self, ["graph_"])
        adjacency = self.graph_.adjacency_
        degree = self.graph_.degree_
        nodes = int(adjacency.shape[0])
        edges = int(np.count_nonzero(adjacency) // 2)
        isolated = int(np.sum(degree <= EPS))
        return {
            "nodes": nodes,
            "edges": edges,
            "mean_degree": float(degree.mean()),
            "max_degree": float(degree.max()),
            "connected_components": int(self.graph_.connected_components_),
            "density": float(np.count_nonzero(adjacency) / max(nodes * (nodes - 1), 1)),
            "isolated_nodes": isolated,
            "isolated_node_percent": float(100.0 * isolated / max(nodes, 1)),
            "removed_constant_features": int(
                getattr(self.graph_, "removed_constant_features_", 0)
            ),
        }

