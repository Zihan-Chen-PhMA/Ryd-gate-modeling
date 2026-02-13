# Getting Started with ryd_gate

## Overview

`ryd_gate` is a Python package for simulating and optimizing Rydberg-atom entangling gates in neutral-atom quantum computers. It provides:

- **7-level Schrödinger solver** for two-atom CZ gate dynamics
- **49-dimensional Hilbert space** (7 levels × 2 atoms) with hyperfine structure
- **Rydberg blockade**, spontaneous decay, and AC Stark shift modeling
- **Monte Carlo noise analysis** for dephasing and position errors
- **Pulse optimization** for time-optimal (TO) and amplitude-robust (AR) protocols
- **Fidelity characterization** using 12 Symmetric State Set (SSS) states

## Installation

```bash
# From source (recommended for development)
git clone https://github.com/ChanceSiyuan/Ryd-gate-modeling.git
cd Ryd-gate-modeling
uv pip install -e ".[dev]"

# Dependencies installed automatically:
# - qutip (Bloch sphere visualization)
# - arc (atomic calculations for Rb)
# - numpy, scipy, matplotlib
```

## Module Index

| Module | Description |
|--------|-------------|
| `ryd_gate.ideal_cz` | CZ gate simulator using SciPy ODE with Monte Carlo noise |
| `ryd_gate.blackman` | Blackman window pulse shaping functions |

## Class Reference: `CZGateSimulator`

The main simulation class in `ryd_gate.ideal_cz`.

### Constructor

```python
CZGateSimulator(
    param_set: Literal['our', 'lukin'] = 'our',
    strategy: Literal['TO', 'AR'] = 'AR',
    blackmanflag: bool = True,
    *,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
    enable_rydberg_dephasing: bool = False,
    enable_position_error: bool = False,
    enable_polarization_leakage: bool = False,
    sigma_detuning: float = 0.0,
    sigma_pos_xyz: tuple = (0.0, 0.0, 0.0),
    n_mc_shots: int = 100,
    mc_seed: int | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param_set` | `str` | `'our'` | Physical parameter configuration (`'our'` = n=70, `'lukin'` = n=53) |
| `strategy` | `str` | `'AR'` | Pulse optimization strategy (`'TO'` or `'AR'`) |
| `blackmanflag` | `bool` | `True` | Apply Blackman envelope for smooth pulses |
| `enable_rydberg_decay` | `bool` | `False` | Include Rydberg state decay as imaginary energy shifts |
| `enable_intermediate_decay` | `bool` | `False` | Include intermediate state decay as imaginary energy shifts |
| `enable_rydberg_dephasing` | `bool` | `False` | Enable Monte Carlo T2* dephasing noise |
| `enable_position_error` | `bool` | `False` | Enable Monte Carlo 3D position fluctuations |
| `enable_polarization_leakage` | `bool` | `False` | Include coupling to unwanted Rydberg state \|r'⟩ |
| `sigma_detuning` | `float` | `0.0` | Dephasing noise std dev (Hz) |
| `sigma_pos_xyz` | `tuple` | `(0,0,0)` | Position noise std dev per axis (meters) |
| `n_mc_shots` | `int` | `100` | Number of Monte Carlo shots |
| `mc_seed` | `int\|None` | `None` | RNG seed for reproducibility |

### Key Methods

| Method | Description |
|--------|-------------|
| `setup_protocol(x)` | Store pulse parameters for repeated use |
| `optimize(x_initial)` | Run pulse parameter optimization (Nelder-Mead) |
| `gate_fidelity(x, fid_type)` | Average gate infidelity over 12 SSS states. Returns `(mean, std)` when MC noise is enabled |
| `diagnose_plot(x, initial_state)` | Plot population evolution for a given initial state |
| `diagnose_run(x, initial_state)` | Return population time series (mid, ryd, ryd_garb) |
| `plot_bloch(x, save=True)` | Generate Bloch sphere plots (TO only) |

## Use Cases

### 1. Basic Fidelity Calculation

```python
from ryd_gate.ideal_cz import CZGateSimulator

sim = CZGateSimulator(param_set='our', strategy='TO')

# Time-optimal pulse parameters
X_TO = [0.1122, 1.0431, -0.72565603, 0.0, 0.452, 1.219096]
infidelity = sim.gate_fidelity(X_TO)
print(f"Gate infidelity: {infidelity:.2e}")
```

### 2. Monte Carlo Noise Analysis

```python
sim = CZGateSimulator(
    param_set='our', strategy='TO',
    enable_rydberg_dephasing=True,
    enable_position_error=True,
    sigma_detuning=50e3,          # 50 kHz dephasing
    sigma_pos_xyz=(30e-9, 30e-9, 30e-9),  # 30 nm position spread
    n_mc_shots=200,
    mc_seed=42,
)

mean_infid, std_infid = sim.gate_fidelity(X_TO)
print(f"Infidelity: {mean_infid:.2e} ± {std_infid:.2e}")
```

### 3. Comparing TO vs AR Strategies

```python
import numpy as np

sim_TO = CZGateSimulator(strategy='TO')
x_TO = [0.1122, 1.0431, -0.72565603, 0.0, 0.452, 1.219096]
infid_TO = sim_TO.gate_fidelity(x_TO)

sim_AR = CZGateSimulator(strategy='AR')
x_AR = [0.85973359, 0.39146974, 0.99181418, 0.1924498,
        -1.17123748, -0.00826712, 1.67429728, 0.28527346]
infid_AR = sim_AR.gate_fidelity(x_AR)

print(f"TO infidelity: {infid_TO:.6f}")
print(f"AR infidelity: {infid_AR:.6f}")
```

### 4. Population Diagnostics

```python
sim = CZGateSimulator(strategy='TO')
x_opt = [0.1122, 1.0431, -0.72565603, 0.0, 0.452, 1.219096]

# Plot population dynamics for |11⟩ initial state
sim.diagnose_plot(x_opt, initial_state='11')

# Get raw arrays for custom analysis
mid_pop, ryd_pop, ryd_garb_pop = sim.diagnose_run(x_opt, '11')
```

### 5. Including Decay Effects

```python
sim_decay = CZGateSimulator(
    param_set='our', strategy='AR',
    enable_rydberg_decay=True,
    enable_intermediate_decay=True,
    enable_polarization_leakage=True,
)

x_AR = [0.85973359, 0.39146974, 0.99181418, 0.1924498,
        -1.17123748, -0.00826712, 1.67429728, 0.28527346]
infid = sim_decay.gate_fidelity(x_AR)
print(f"Infidelity with decay: {infid:.6f}")
```

---

## See Also

- `examples/run_cz_simulation.py` – Complete simulation example
- `docs/schrodinger_solver.md` – Detailed solver documentation
- `docs/error_budget_methodology.md` – Error decomposition methodology
- `paper/en_v2.tex` – Full derivations and references
