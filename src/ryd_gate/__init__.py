"""ryd_gate – Simulation and optimisation of Rydberg-atom entangling gates."""

__version__ = "0.1.0"

from .blackman import blackman_pulse, blackman_pulse_sqrt, blackman_window
from .ideal_cz import CZGateSimulator, MonteCarloResult

__all__ = [
    "CZGateSimulator",
    "MonteCarloResult",
    "blackman_pulse",
    "blackman_pulse_sqrt",
    "blackman_window",
]
