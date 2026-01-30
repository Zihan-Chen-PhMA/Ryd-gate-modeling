# ryd-gate

[![CI](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ChanceSiyuan/Ryd-gate-modeling/graph/badge.svg)](https://codecov.io/gh/ChanceSiyuan/Ryd-gate-modeling)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simulation and optimization of Rydberg-atom entangling gates for neutral-atom quantum computing.

This package provides tools for modelling two-photon Rydberg excitation in
⁸⁷Rb, including hyperfine structure, Rydberg blockade, spontaneous decay, and
AC Stark shifts. It supports both JAX-accelerated density-matrix evolution and
SciPy-based Schrödinger-equation solvers, together with pulse-shape
optimisation routines for time-optimal, amplitude-robust, and Doppler-robust
CZ gate protocols.

## Installation

```bash
# editable install (recommended for development)
pip install -e ".[dev]"
```

## Quickstart

```python
from ryd_gate.full_error_model import jax_atom_Evolution
import jax.numpy as jnp

model = jax_atom_Evolution()

amp_420 = lambda t: 1
phase_420 = lambda t: 0
amp_1013 = lambda t: 1

tlist = jnp.linspace(0, 0.2, 100)
sol = model.integrate_rho_jax(
    tlist, amp_420, phase_420, amp_1013,
    model.psi_to_rho(model.SSS_initial_state_list[5]),
)
```

## Package layout

| Module | Description |
|---|---|
| `ryd_gate.full_error_model` | JAX-based two-atom density-matrix simulator with decay |
| `ryd_gate.ideal_cz` | Simplified CZ gate simulator (SciPy ODE) |
| `ryd_gate.lightshift` | AC Stark shift and effective Rabi frequency calculations |
| `ryd_gate.blackman` | Blackman window pulse shaping |
| `ryd_gate.noise` | Discrete pulse optimisation (TO / AR / DR / SSR) |

## References

* Evered *et al.*, "High-fidelity parallel entangling gates on a neutral-atom
  quantum computer", Nature **622**, 268 (2023).
* See `paper/en_v2.tex` for detailed theoretical derivations.

## License

MIT
