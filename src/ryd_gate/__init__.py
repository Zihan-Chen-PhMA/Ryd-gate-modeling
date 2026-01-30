"""ryd_gate – Simulation and optimisation of Rydberg-atom entangling gates."""

__version__ = "0.1.0"

from .blackman import blackman_pulse, blackman_pulse_sqrt, blackman_window
from .noise import PulseOptimizer

# Optional imports that require heavy dependencies (arc, qutip, jax)
try:
    from .full_error_model import jax_atom_Evolution
except ImportError:
    jax_atom_Evolution = None

try:
    from .ideal_cz import CZGateSimulator
except ImportError:
    CZGateSimulator = None

__all__ = [
    "jax_atom_Evolution",
    "CZGateSimulator",
    "blackman_pulse",
    "blackman_pulse_sqrt",
    "blackman_window",
    "PulseOptimizer",
]
