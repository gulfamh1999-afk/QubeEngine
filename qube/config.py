
from dataclasses import dataclass

EPS = 1e-9

QUANTUM_BACKEND_MATHEMATICAL = "Mathematical"
QUANTUM_BACKEND_QISKIT_AER = "Qiskit Aer Simulator"

QUANTUM_BACKEND_ALIASES = {
    "mathematical": QUANTUM_BACKEND_MATHEMATICAL,
    "math": QUANTUM_BACKEND_MATHEMATICAL,
    "current": QUANTUM_BACKEND_MATHEMATICAL,
    "qiskit aer simulator": QUANTUM_BACKEND_QISKIT_AER,
    "qiskit_aer": QUANTUM_BACKEND_QISKIT_AER,
    "qiskit-aer": QUANTUM_BACKEND_QISKIT_AER,
    "aer": QUANTUM_BACKEND_QISKIT_AER,
    "aer simulator": QUANTUM_BACKEND_QISKIT_AER,
    "aer_simulator": QUANTUM_BACKEND_QISKIT_AER,
}

@dataclass
class Qube2Config:
    """Configuration for the QUBE 2.0 representation."""

    n_descriptors: int = 256
    graph_k: int = 8
    graph_min_abs_corr: float = 0.05
    spectral_dims: int = 8
    diffusion_scales: tuple[int, ...] = (0, 1, 2, 3, 5)
    quantum_channels: int = 96
    quantum_pair_samples: int = 128
    quantum_backend: str = QUANTUM_BACKEND_MATHEMATICAL
    qiskit_qubits: int = 6
    qiskit_reps: int = 2
    rf_trees: int = 600
    selector_trees: int = 400
    batch_size: int = 4096
    random_state: int = 42
    n_jobs: int = -1
