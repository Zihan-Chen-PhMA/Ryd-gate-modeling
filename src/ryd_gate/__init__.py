"""ryd_gate – Simulation and optimisation of Rydberg-atom entangling gates."""

__version__ = "0.1.0"

from .blackman import blackman_pulse, blackman_pulse_sqrt, blackman_window
from .full_error_model import jax_atom_Evolution
from .ideal_cz import CZGateSimulator
from .noise import PulseOptimizer

__all__ = [
    "jax_atom_Evolution",
    "CZGateSimulator",
    "blackman_pulse",
    "blackman_pulse_sqrt",
    "blackman_window",
    "PulseOptimizer",
]
