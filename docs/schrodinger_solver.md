# Schrödinger Solver for CZ Gate Simulation

This document describes the `CZGateSimulator` class in `ideal_cz.py`, which provides a SciPy-based Schrödinger equation solver for simulating two-qubit CZ gates in Rydberg atom systems.

## Overview

### When to Use This Module

| Use Case | Recommended Module |
|----------|-------------------|
| Fast pulse optimization | `ideal_cz.py` (this module) |
| Understanding ideal gate dynamics | `ideal_cz.py` |
| Comparing TO vs AR pulse strategies | `ideal_cz.py` |
| Full decoherence modeling | `full_error_model.py` |
| Thermal effects analysis | `full_error_model.py` |
| Leakage channel quantification | `full_error_model.py` |

### Comparison: Schrödinger vs Master Equation Solvers

| Feature | `CZGateSimulator` | `jax_atom_Evolution` |
|---------|-------------------|---------------------|
| **Equation** | Schrödinger (pure state) | Lindblad master equation |
| **State representation** | 49-dim vector | 100×100 density matrix |
| **Levels per atom** | 7 | 10 |
| **Decay modeling** | Imaginary energy shift | Full Lindblad operators |
| **Leakage states** | No | Yes (L0, L1) |
| **BBR transitions** | No | Yes (r→rP) |
| **Backend** | SciPy `solve_ivp` | JAX `odeint` |
| **GPU acceleration** | No | Yes |
| **Typical use** | Pulse optimization | Error budget analysis |

## Class Reference

### Constructor

```python
CZGateSimulator(
    decayflag: bool,
    param_set: Literal['our', 'lukin'] = 'our',
    strategy: Literal['TO', 'AR'] = 'AR',
    blackmanflag: bool = True
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decayflag` | `bool` | (required) | Include spontaneous emission as imaginary energy |
| `param_set` | `str` | `'our'` | Physical parameter configuration |
| `strategy` | `str` | `'AR'` | Pulse optimization strategy |
| `blackmanflag` | `bool` | `True` | Apply Blackman envelope for smooth pulses |

### Public Methods

| Method | Description |
|--------|-------------|
| `optimize(x_initial)` | Run pulse parameter optimization |
| `avg_fidelity(x)` | Calculate average gate infidelity |
| `diagnose_plot(x, initial_state)` | Plot population evolution |
| `diagnose_run(x, initial_state)` | Return population time series |
| `plot_bloch(x, save=True)` | Generate Bloch sphere plots (TO only) |

## Level Structure

Each atom has 7 levels forming a 49-dimensional two-atom Hilbert space:

```
Index   Label   Description                     Quantum Numbers
-----   -----   -----------                     ---------------
  0     |0⟩     Ground state (dark)             5S1/2, F=1
  1     |1⟩     Ground state (qubit)            5S1/2, F=2
  2     |e1⟩    Intermediate                    6P3/2, F'=1
  3     |e2⟩    Intermediate                    6P3/2, F'=2
  4     |e3⟩    Intermediate                    6P3/2, F'=3
  5     |r⟩     Target Rydberg                  nS1/2, mJ=-1/2
  6     |r'⟩    Unwanted Rydberg                nS1/2, mJ=+1/2

                     |r⟩ (5)     |r'⟩ (6)
                      ↑  ↑         ↑
                1013nm│  │         │
                      │  │         │
          ┌───────────┴──┴─────────┴───────────┐
          │   |e1⟩(2)   |e2⟩(3)   |e3⟩(4)      │  6P3/2
          └───────────┬──┬─────────────────────┘
                420nm │  │
                      ↓  ↓
          ┌───────────────────────────────────┐
          │       |0⟩(0)    |1⟩(1)            │  5S1/2
          └───────────────────────────────────┘
```

The two-atom basis states are tensor products: |ij⟩ = |i⟩ ⊗ |j⟩, giving indices 0-48.

## Physical Parameters

### 'our' Configuration (n=70 Rydberg)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Rydberg level | n=70 | 70S1/2 state |
| Intermediate detuning (Δ) | 6.1 GHz | From 6P3/2 resonance |
| Effective Rabi (Ω_eff) | 7 MHz | Two-photon coupling |
| Rydberg interaction (V) | 874 GHz / d⁶ | Van der Waals coefficient |
| Zeeman shift | -56 MHz | For |r'⟩ state |
| 6P3/2 lifetime | 110 ns | Intermediate state |
| 70S lifetime | 152 μs | Rydberg state |

### 'lukin' Configuration (n=53 Rydberg)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Rydberg level | n=53 | 53S1/2 state |
| Intermediate detuning (Δ) | 7.8 GHz | From 6P3/2 resonance |
| 420nm Rabi | 237 MHz | Blue laser coupling |
| 1013nm Rabi | 303 MHz | IR laser coupling |
| Rydberg interaction (V) | 450 MHz | At experimental distance |
| Zeeman shift | -2.4 GHz | For |r'⟩ state |
| 53S lifetime | 88 μs | Rydberg state |

## Optimization Strategies

### Time-Optimal (TO) Strategy

Phase modulation with single cosine:

```
φ(t) = A·cos(ωt + φ₀) + δ·t
```

**Parameter vector:** `x = [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale]`

| Index | Symbol | Description | Typical Range |
|-------|--------|-------------|---------------|
| 0 | A | Cosine amplitude | (-π, π) |
| 1 | ω/Ω_eff | Modulation frequency (normalized) | (-10, 10) |
| 2 | φ₀ | Initial phase | (-π, π) |
| 3 | δ/Ω_eff | Linear chirp rate (normalized) | (-2, 2) |
| 4 | θ | Single-qubit Z rotation | (-∞, ∞) |
| 5 | T/T_scale | Gate time (normalized) | (0, π) |

### Amplitude-Robust (AR) Strategy

Phase modulation with dual sine for first-order amplitude robustness:

```
φ(t) = A₁·sin(ωt + φ₁) + A₂·sin(2ωt + φ₂) + δ·t
```

**Parameter vector:** `x = [ω/Ω_eff, A₁, φ₁, A₂, φ₂, δ/Ω_eff, T/T_scale, θ]`

| Index | Symbol | Description |
|-------|--------|-------------|
| 0 | ω/Ω_eff | Base modulation frequency |
| 1 | A₁ | First sine amplitude |
| 2 | φ₁ | First sine phase |
| 3 | A₂ | Second sine amplitude |
| 4 | φ₂ | Second sine phase |
| 5 | δ/Ω_eff | Linear chirp rate |
| 6 | T/T_scale | Gate time |
| 7 | θ | Single-qubit Z rotation |

## Usage Examples

### Basic Optimization

```python
from ryd_gate.ideal_cz import CZGateSimulator

# Initialize with TO strategy, no decay
sim = CZGateSimulator(decayflag=False, param_set='our', strategy='TO')

# Initial guess for pulse parameters
x0 = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]

# Run optimization
result = sim.optimize(x0)
print(f"Optimized infidelity: {result.fun:.6f}")
print(f"Optimal parameters: {result.x}")
```

### Population Diagnostics

```python
# Visualize population dynamics for |11⟩ initial state
sim.diagnose_plot(result.x, initial_state='11')

# Get raw population arrays for custom analysis
mid_pop, ryd_pop, ryd_garb_pop = sim.diagnose_run(result.x, '11')
```

### Comparing TO vs AR Strategies

```python
import numpy as np

# TO strategy
sim_TO = CZGateSimulator(decayflag=False, strategy='TO')
x_TO = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
result_TO = sim_TO.optimize(x_TO)

# AR strategy
sim_AR = CZGateSimulator(decayflag=False, strategy='AR')
x_AR = [1.0, 0.1, 0.0, 0.05, 0.0, 0.0, 1.0, 0.0]
result_AR = sim_AR.optimize(x_AR)

print(f"TO infidelity: {result_TO.fun:.6f}")
print(f"AR infidelity: {result_AR.fun:.6f}")
```

### Bloch Sphere Visualization (TO only)

```python
sim = CZGateSimulator(decayflag=False, strategy='TO')
x_opt = [0.1122, 1.0431, -0.726, 0.0, 0.452, 1.219]

# Generate Bloch sphere plots for |01⟩→|0r⟩ and |11⟩→|W⟩ transitions
sim.plot_bloch(x_opt, save=True)
# Saves: 10-r0_Bloch.png, 11-W_Bloch.png
```

### Including Decay Effects

```python
# With spontaneous emission (imaginary energy shift approximation)
sim_decay = CZGateSimulator(decayflag=True, param_set='our', strategy='AR')

x0 = [1.0, 0.1, 0.0, 0.05, 0.0, 0.0, 1.0, 0.0]
infid = sim_decay.avg_fidelity(x0)
print(f"Infidelity with decay: {infid:.6f}")

# Note: For accurate decay modeling, use full_error_model.py instead
```

## Hamiltonian Structure

The total Hamiltonian is:

```
H(t) = H_const + amplitude(t) × [e^{-iφ(t)} H_420 + e^{iφ(t)} H_420† + H_1013 + H_1013†]
```

where:
- `H_const`: Diagonal terms (detunings, Zeeman shifts, VdW interaction)
- `H_420`: 420nm laser coupling (ground → intermediate)
- `H_1013`: 1013nm laser coupling (intermediate → Rydberg)
- `amplitude(t)`: Blackman envelope (if enabled)
- `φ(t)`: Phase modulation (TO or AR strategy)

The Clebsch-Gordan coefficients from ARC library account for the hyperfine structure and polarization selection rules.

## Notes

1. **Imaginary energy approximation**: When `decayflag=True`, decay is modeled as imaginary energy shifts in the diagonal Hamiltonian. This is approximate; for accurate decay/leakage modeling, use `full_error_model.py`.

2. **Optimization algorithm**: Uses SciPy's Nelder-Mead optimizer with `fatol=1e-9`. The callback writes intermediate parameters to `opt_hf_new.txt`.

3. **Time discretization**: State evolution is computed on 1000 time points uniformly distributed over the gate duration.

4. **Numerical precision**: ODE integration uses DOP853 (8th order Runge-Kutta) with `rtol=1e-8`, `atol=1e-12`.

## See Also

- `full_error_model.py` - Full density matrix simulation with Lindblad decay
- `noise_analysis.md` - Error budget analysis using the full model
- `thermal_effects_analysis.md` - Temperature dependence of gate fidelity
