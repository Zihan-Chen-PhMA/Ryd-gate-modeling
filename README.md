# ryd-gate

[![CI](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ChanceSiyuan/Ryd-gate-modeling/graph/badge.svg)](https://codecov.io/gh/ChanceSiyuan/Ryd-gate-modeling)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-201%20passed-brightgreen.svg)](tests/)

Simulation and optimization of Rydberg-atom entangling gates for neutral-atom quantum computing.

This package provides tools for modelling two-photon Rydberg excitation in ⁸⁷Rb, including hyperfine structure, Rydberg blockade, spontaneous decay, and AC Stark shifts. It supports both JAX-accelerated density-matrix evolution and SciPy-based Schrödinger-equation solvers, together with pulse-shape optimisation routines for time-optimal, amplitude-robust, and Doppler-robust CZ gate protocols.

## Features

- **Two independent quantum solvers** with cross-validated results
- **Full error model** including spontaneous decay, BBR, and leakage
- **12 SSS (Symmetric State Set) states** for efficient fidelity characterization
- **Pulse optimization** for TO/AR/DR/SSR gate protocols
- **Comprehensive test suite** (201 tests) validating physical correctness

## Installation

```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# Using pip
pip install -e ".[dev]"
```

## Quickstart

### Density Matrix Solver (JAX)

```python
from ryd_gate.full_error_model import jax_atom_Evolution
import jax.numpy as jnp

model = jax_atom_Evolution(blockade=True, ryd_decay=True, mid_decay=True)

amp_420 = lambda t: 1.0
phase_420 = lambda t: 0.0
amp_1013 = lambda t: 1.0

tlist = jnp.linspace(0, 0.2, 100)  # 200 ns evolution
sol = model.integrate_rho_jax(
    tlist, amp_420, phase_420, amp_1013,
    rho0=model.psi_to_rho(model.SSS_initial_state_list[5]),  # |11⟩ state
)
```

### Schrödinger Solver (SciPy)

```python
from ryd_gate.ideal_cz import CZGateSimulator
import numpy as np

sim = CZGateSimulator(decayflag=False, param_set='our', strategy='TO')

# Time-optimal pulse parameters
X_TO = [0.1122, 1.0431, -0.72565603, 0.0, 0.452, 1.219096]
infidelity = sim.avg_fidelity(X_TO)
print(f"Gate infidelity: {infidelity:.2e}")
```

## Package Layout

| Module | Description |
|--------|-------------|
| `ryd_gate.full_error_model` | JAX-based 10-level density-matrix simulator with Lindblad decay |
| `ryd_gate.ideal_cz` | 7-level Schrödinger equation solver (SciPy ODE) |
| `ryd_gate.lightshift` | AC Stark shift and effective Rabi frequency calculations |
| `ryd_gate.blackman` | Blackman window pulse shaping |
| `ryd_gate.noise` | Discrete pulse optimisation (TO / AR / DR / SSR) |

## Physical Model

Both solvers model two ⁸⁷Rb atoms with:

| Parameter | Value | Description |
|-----------|-------|-------------|
| n | 70 | Rydberg principal quantum number |
| Ω_eff | 2π × 5 MHz | Effective two-photon Rabi frequency |
| Δ | 2π × 9.1 GHz | Intermediate state detuning (blue) |
| d | 3 μm | Interatomic distance |
| C₆ | 2π × 874 GHz·μm⁶ | van der Waals coefficient |

## Testing

The package includes **201 tests** covering both solvers:

```bash
# Run all tests
uv run pytest tests/ -v

# Run fast tests only (skip slow dynamics comparisons)
uv run pytest tests/ -v -m "not slow"

# Run with coverage
uv run pytest tests/ --cov=ryd_gate --cov-report=html
```

### Test Categories

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_ideal_cz.py` | 35 | Schrödinger solver: initialization, Hamiltonian, fidelity |
| `test_full_error_model.py` | 94 | Density matrix solver: decay, blockade, diagnostics |
| `test_solver_comparison.py` | 72 | Cross-validation between both solvers |

### Validation Highlights

The cross-validation suite verifies:

- **SSS state equivalence** — 12 states match between 49-dim and 100-dim representations
- **CPTP properties** — Trace preservation, Hermiticity, positive semi-definiteness
- **Parameter consistency** — Physical constants match between solvers (within 5%)
- **Ideal CZ fidelity** — F ≈ 1.0 for all 12 SSS states

See [docs/validation.md](docs/validation.md) for detailed test documentation.

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting_started.md) | Installation and basic usage |
| [Schrödinger Solver](docs/schrodinger_solver.md) | 7-level model theory and API |
| [Noise Analysis](docs/noise_analysis.md) | Error budget and infidelity decomposition |
| [Thermal Effects](docs/thermal_effects_analysis.md) | Doppler shifts and position spread |
| [Validation](docs/validation.md) | Test suite documentation and reproducibility |

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/generate_noise_report.py` | Generate comprehensive noise analysis |
| `scripts/plot_population_evolution.py` | SSS population comparison between solvers |

## References

* Evered *et al.*, "High-fidelity parallel entangling gates on a neutral-atom quantum computer", Nature **622**, 268 (2023).
* Ma *et al.*, "Benchmarking and fidelity response theory of high-fidelity Rydberg entangling gates", PRX Quantum **6**, 010331 (2025).
* See `paper/en_v2.tex` for detailed theoretical derivations.

## License

MIT
