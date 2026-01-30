# Getting Started with ryd_gate

## Overview

`ryd_gate` is a Python package for simulating and optimizing Rydberg-atom entangling gates in neutral-atom quantum computers. It provides:

- **JAX-accelerated density matrix evolution** with full Lindblad master equation
- **10-level atomic structure** for ⁸⁷Rb including hyperfine states
- **Rydberg blockade**, spontaneous decay, and AC Stark shift modeling
- **Pulse optimization** routines for time-optimal and robust gate protocols
- **Fidelity characterization** using Symmetric State Sets (SSS)

## Installation

```bash
# From source (recommended for development)
git clone https://github.com/your-repo/Ryd-gate-modeling.git
cd Ryd-gate-modeling
pip install -e ".[dev]"

# Dependencies installed automatically:
# - jax, jaxopt (GPU acceleration)
# - qutip (quantum operations)
# - arc (atomic calculations for Rb)
# - numpy, scipy, matplotlib
```

## Module Index

| Module | Description |
|--------|-------------|
| `ryd_gate.full_error_model` | JAX-based two-atom density matrix simulator with decay channels |
| `ryd_gate.ideal_cz` | Simplified CZ gate simulator using SciPy ODE |
| `ryd_gate.lightshift` | AC Stark shift and effective Rabi frequency calculations |
| `ryd_gate.blackman` | Blackman window pulse shaping functions |
| `ryd_gate.noise` | Discrete pulse optimization (TO/AR/DR/SSR protocols) |

## Class Reference: `jax_atom_Evolution`

The main simulation class in `ryd_gate.full_error_model`.

### Constructor

```python
jax_atom_Evolution(blockade=True, ryd_decay=True, mid_decay=True, 
                   distance=3, psi0=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `blockade` | bool | True | Enable Rydberg-Rydberg van der Waals blockade |
| `ryd_decay` | bool | True | Enable Rydberg state decay (BBR + radiative) |
| `mid_decay` | bool | True | Enable intermediate 6P₃/₂ state decay |
| `distance` | float | 3 | Interatomic distance in μm |
| `psi0` | array | None | Custom initial state; defaults to \|11⟩ |

### Key Methods

#### State Initialization
- `init_state()` – Initialize single-atom basis states
- `init_state0(psi0)` – Set custom initial state
- `psi_to_rho(psi)` – Convert pure state to density matrix

#### Hamiltonian Construction
- `init_blockade(distance)` – Setup van der Waals interaction
- `init_diag_ham(blockade)` – Initialize diagonal Hamiltonian
- `init_420_ham()` – 420nm laser coupling (ground → 6P₃/₂)
- `init_1013_ham()` – 1013nm laser coupling (6P₃/₂ → 70S₁/₂)
- `hamiltonian(t, args)` – Full time-dependent Hamiltonian
- `hamiltonian_sparse(t, args)` – Sparse representation for efficiency

#### Time Evolution
- `integrate_rho_jax(tlist, amp_420, phase_420, amp_1013, rho0)` – Single initial state evolution
- `integrate_rho_multi_jax(...)` – Batch evolution with JAX vmap
- `integrate_rho_multi2_jax(...)` – Alternative batch method

#### Light Shifts & Decay
- `init_420_light_shift()` – 420nm AC Stark shifts
- `init_1013_light_shift()` – 1013nm AC Stark shifts
- `mid_state_decay(rho0)` – Instantaneous intermediate state decay
- `rydberg_RD_branch_ratio()` – Rydberg decay branching ratios
- `mid_branch_ratio(level_label)` – 6P₃/₂ decay branching ratios

#### Gate Characterization
- `init_SSS_states()` – Initialize Symmetric State Set
- `SSS_rec(idx)` – Recovery unitary for SSS measurement
- `CZ_ideal()` – Ideal CZ gate matrix
- `CZ_fidelity(state_final, state_initial, theta)` – Gate fidelity with Z-correction
- `CZ_fidelity_with_leakage(state_final, state_initial, theta)` – Fidelity treating |L0⟩ as |0⟩ (for experimental comparison)
- `CZ_back_to_00(state_final, state_idx, ...)` – Recovery to \|00⟩ basis

#### Infidelity Diagnostics
- `occ_operator(level_label)` – Create occupation operator for a given level
- `diagnose_population(sol)` – Extract population evolution from trajectory
- `diagnose_infidelity(rho_final, psi0, theta)` – Decompose infidelity into error sources
- `diagnose_plot(tlist, sol, initial_label, save_path)` – Plot population dynamics

#### Polarizability Calculations
- `get_polarizability_fs(K, nLJ_target, ...)` – Fine structure polarizability
- `get_polarizability_hfs_from_fs(K, F, I, ...)` – HFS from FS matrix elements
- `get_polarizability_hfs(K, I, nLJF_target, ...)` – Direct HFS polarizability

---

## Use Cases

### 1. Basic CZ Gate Simulation

```python
import jax.numpy as jnp
from ryd_gate.full_error_model import jax_atom_Evolution

# Initialize simulator with default parameters
model = jax_atom_Evolution(distance=3.0)

# Define pulse functions (constant amplitude example)
amp_420 = lambda t: 1.0
phase_420 = lambda t: 0.0
amp_1013 = lambda t: 1.0

# Time grid (μs)
tlist = jnp.linspace(0, 0.2, 500)

# Evolve from |11⟩ initial state
sol = model.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013)

# Final density matrix
rho_final = sol[-1]
print(f"Trace: {jnp.trace(rho_final).real:.6f}")
```

### 2. Fidelity Calculation with SSS States

```python
import jax.numpy as jnp
from ryd_gate.full_error_model import jax_atom_Evolution

model = jax_atom_Evolution()

# Prepare SSS initial states (12 states for complete characterization)
rho0_list = [model.psi_to_rho(psi) for psi in model.SSS_initial_state_list]

# Your pulse functions
amp_420 = lambda t: jnp.where(t < 0.1, 1.0, 0.0)
phase_420 = lambda t: 0.0
amp_1013 = lambda t: 1.0

tlist = jnp.linspace(0, 0.2, 500)

# Batch evolution
sol = model.integrate_rho_multi_jax(tlist, amp_420, phase_420, amp_1013, rho0_list)

# Calculate average fidelity
fidelities = []
for i, (rho_f, psi0) in enumerate(zip(sol[:, -1], model.SSS_initial_state_list)):
    fid, theta = model.CZ_fidelity(rho_f, psi0)
    fidelities.append(fid)

avg_fidelity = jnp.mean(jnp.array(fidelities))
print(f"Average gate fidelity: {avg_fidelity:.4f}")
```

### 3. Custom Pulse Optimization

```python
import jax.numpy as jnp
from ryd_gate.blackman import blackman_pulse
from ryd_gate.full_error_model import jax_atom_Evolution

model = jax_atom_Evolution()

# Blackman-shaped pulse
gate_time = 0.15  # μs
amp_420 = lambda t: blackman_pulse(t, gate_time)
phase_420 = lambda t: 0.0
amp_1013 = lambda t: blackman_pulse(t, gate_time)

tlist = jnp.linspace(0, gate_time, 300)

# Single state evolution
psi0 = model.SSS_initial_state_list[5]  # |11⟩ state
rho0 = model.psi_to_rho(psi0)

sol = model.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013, rho0)

fid, theta = model.CZ_fidelity(sol[-1], psi0)
print(f"Fidelity with Blackman pulse: {fid:.5f}")
print(f"Optimal Z-rotation: {theta:.4f} rad")
```

### 4. Parameter Sweep Example

```python
import jax.numpy as jnp
import numpy as np
from ryd_gate.full_error_model import jax_atom_Evolution

# Sweep over interatomic distances
distances = np.linspace(2.5, 4.0, 10)
fidelities = []

for d in distances:
    model = jax_atom_Evolution(distance=d)
    
    amp_420 = lambda t: 1.0
    phase_420 = lambda t: 0.0
    amp_1013 = lambda t: 1.0
    
    tlist = jnp.linspace(0, 0.2, 300)
    sol = model.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013)
    
    fid, _ = model.CZ_fidelity(sol[-1], model.psi0)
    fidelities.append(fid)
    print(f"d = {d:.2f} μm: F = {fid:.4f}")

# Find optimal distance
opt_idx = np.argmax(fidelities)
print(f"\nOptimal distance: {distances[opt_idx]:.2f} μm")
```

---

## Level Structure Diagram

The 10-level ⁸⁷Rb system modeled in `jax_atom_Evolution`:

```
                    70S₁/₂
                   ┌───────┐
                   │ |r1⟩  │ mⱼ = -1/2
                   │ |r2⟩  │ mⱼ = +1/2
                   │ |rP⟩  │ (BBR-induced P state)
                   └───┬───┘
                       │ 1013 nm (σ⁺)
                       │
                   ┌───┴───┐
        6P₃/₂      │ |e1⟩  │ F=1, mF=-1
        Δ ≈ 9 GHz  │ |e2⟩  │ F=2, mF=-1
                   │ |e3⟩  │ F=3, mF=-1
                   └───┬───┘
                       │ 420 nm (σ⁻)
                       │
    ┌──────────────────┴──────────────────┐
    │              5S₁/₂                   │
    │  ┌─────┐              ┌─────┐       │
    │  │ |0⟩ │ F=1, mF=0    │ |1⟩ │ F=2   │  Computational basis
    │  └─────┘              └─────┘       │
    │  ┌─────┐              ┌─────┐       │
    │  │|L0⟩ │ F=1, mF≠0    │|L1⟩ │ F=2   │  Leakage states
    │  └─────┘              └─────┘       │
    └─────────────────────────────────────┘
```

### Key Physical Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Intermediate detuning | Δ | -9.1 GHz | 6P₃/₂ detuning |
| Effective Rabi freq | Ω_eff | 5 MHz | Two-photon Rabi frequency |
| Blockade coefficient | C₆ | 874 GHz·μm⁶ | van der Waals coefficient |
| 6P₃/₂ lifetime | τ_mid | 0.11 μs | Intermediate state lifetime |
| 70S₁/₂ lifetime | τ_ryd | 151.55 μs | Rydberg radiative lifetime |
| BBR decay time | τ_BBR | 410.41 μs | Black-body radiation decay |

---

## Physical Model Summary

### Rydberg Blockade

The van der Waals interaction between two atoms in Rydberg states:

$$V = \frac{C_6}{d^6}$$

For d = 3 μm with C₆ = 874 GHz·μm⁶, this gives V ≈ 1.2 GHz, far exceeding typical Rabi frequencies and enabling the blockade mechanism for CZ gates.

### Decay Channels

1. **Rydberg radiative decay**: 70S → 5P → 5S cascade with branching to \|0⟩, \|1⟩, and leakage states
2. **BBR-induced decay**: 70S → 70P transitions at room temperature
3. **Intermediate state decay**: 6P₃/₂ → 5S₁/₂ with τ ≈ 110 ns

### AC Stark Shifts

Both lasers induce differential light shifts on the qubit states:

- **420 nm**: Shifts \|0⟩ and \|1⟩ differently via 6P₃/₂ coupling
- **1013 nm**: Shifts Rydberg state and ground states via off-resonant D-line coupling

These shifts are automatically calculated and compensated in the Hamiltonian.

---

## See Also

- `examples/run_cz_simulation.py` – Complete simulation example
- `examples/noise_analysis.py` – Noise and error analysis
- `docs/theory.rst` – Detailed theoretical background
- `paper/en_v2.tex` – Full derivations and references
