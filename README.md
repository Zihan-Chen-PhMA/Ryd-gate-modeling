# ryd-gate

[![CI](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simulation and optimization of Rydberg-atom entangling gates for neutral-atom quantum computing.

This package provides tools for modelling two-photon Rydberg excitation in ⁸⁷Rb, including hyperfine structure, Rydberg blockade, spontaneous decay, and AC Stark shifts. It uses a SciPy-based Schrödinger-equation solver with pulse-shape optimisation routines for time-optimal (TO) and amplitude-robust (AR) CZ gate protocols.

## Features

- **7-level Schrödinger solver** in a 49-dimensional two-atom Hilbert space
- **Full error model** including spontaneous decay (Rydberg + intermediate), dephasing, and position errors
- **12 SSS (Symmetric State Set) states** for efficient fidelity characterization
- **Monte Carlo noise analysis** for Rydberg dephasing and 3D position fluctuations
- **Pulse optimization** for TO and AR gate protocols

## Installation

```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# Using pip
pip install -e ".[dev]"
```

## Quickstart

```python
from ryd_gate.ideal_cz import CZGateSimulator
import numpy as np

sim = CZGateSimulator(param_set='our', strategy='TO')

# Time-optimal pulse parameters: [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale]
X_TO = [0.1122, 1.0431, -0.72565603, 0.0, 0.452, 1.219096]
infidelity = sim.gate_fidelity(X_TO)
print(f"Gate infidelity: {infidelity:.2e}")
```

## Package Layout

| Module | Description |
|--------|-------------|
| `ryd_gate.ideal_cz` | 7-level Schrödinger equation solver (SciPy ODE) with Monte Carlo noise |
| `ryd_gate.blackman` | Blackman window pulse shaping |

## Physical Model

The solver models two ⁸⁷Rb atoms with:

| Parameter | Value | Description |
|-----------|-------|-------------|
| n | 70 | Rydberg principal quantum number |
| Ω_eff | 2π × 7 MHz | Effective two-photon Rabi frequency |
| Δ | 2π × 6.1 GHz | Intermediate state detuning |
| d | 3 μm | Interatomic distance |
| C₆ | 2π × 874 GHz·μm⁶ | van der Waals coefficient |

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run fast tests only (skip slow dynamics)
uv run pytest tests/ -v -m "not slow"

# Run with coverage
uv run pytest tests/ --cov=ryd_gate --cov-report=html
```

### Test Files

| Test File | Description |
|-----------|-------------|
| `test_ideal_cz.py` | Schrödinger solver: initialization, Hamiltonian, fidelity, Monte Carlo |
| `test_cz_gate_phase.py` | CZ gate phase and SSS state verification |
| `test_blackman.py` | Blackman pulse shaping functions |
| `test_init.py` | Package-level imports and exports |

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting_started.md) | Installation and basic usage |
| [Schrödinger Solver](docs/schrodinger_solver.md) | 7-level model theory and API |
| [Error Budget](docs/error_budget_methodology.md) | Error decomposition methodology |
| [Validation](docs/validation.md) | Test suite documentation |

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/draft.py` | Main analysis / prototyping script |
| `scripts/error_deterministic.py` | Deterministic error budget calculation |
| `scripts/error_monte_carlo.py` | Monte Carlo error analysis |
| `scripts/generate_mc_data.py` | Generate Monte Carlo datasets |
| `scripts/generate_si_tables.py` | Generate SI tables for the paper |
| `scripts/opt_bright.py` | Optimize bright-detuning parameters |
| `scripts/opt_dark.py` | Optimize dark-detuning parameters |
| `scripts/plot_mid_pop.py` | Plot intermediate state populations |
| `scripts/plot_population_evolution_sch.py` | Plot population evolution |
| `scripts/verify_cz_dark.py` | Verify CZ gate with dark-detuning parameters |

## References

* Evered *et al.*, "High-fidelity parallel entangling gates on a neutral-atom quantum computer", Nature **622**, 268 (2023).
* Ma *et al.*, "Benchmarking and fidelity response theory of high-fidelity Rydberg entangling gates", PRX Quantum **6**, 010331 (2025).
* See `paper/en_v2.tex` for detailed theoretical derivations.

## License

MIT
