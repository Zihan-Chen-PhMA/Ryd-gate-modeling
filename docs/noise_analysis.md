# CZ Gate Noise Analysis

This document provides a detailed analysis of noise sources in the Rydberg CZ gate implementation using the 10-level `jax_atom_Evolution` model.

## Overview

The CZ gate is implemented via a two-photon Rydberg excitation scheme in 87Rb atoms. Gate errors arise from several physical mechanisms:

1. **Spontaneous emission** from intermediate and Rydberg states
2. **Leakage** to non-computational hyperfine states
3. **Residual Rydberg excitation** at gate completion
4. **Coherent errors** from imperfect pulse calibration

This analysis quantifies each error source using the diagnostic methods in `full_error_model.py`.

## How to Reproduce

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/ChanceSiyuan/Ryd-gate-modeling.git
cd Ryd-gate-modeling

# Install dependencies
pip install -e ".[dev]"
# or with uv:
uv sync
```

### Running the Analysis

```bash
# Generate full noise report with plots
uv run python scripts/generate_noise_report.py --save-plots --n-points 50

# Or without plots (faster)
uv run python scripts/generate_noise_report.py --n-points 50
```

Output files:
- `docs/noise_analysis_results.json` - Numerical results
- `docs/figures/population_*.png` - Population evolution plots

### Using the Diagnostic API Directly

```python
import jax.numpy as jnp
from ryd_gate.full_error_model import jax_atom_Evolution

# Initialize model
model = jax_atom_Evolution(blockade=True, ryd_decay=True, mid_decay=True, distance=3)

# Define pulse (example: optimized phase-modulated protocol)
Omega = model.rabi_eff / (2 * jnp.pi)  # ~5 MHz
tf = 1.219096 / Omega  # Gate time ~244 ns

amp_420 = lambda t: 1.0
phase_420 = lambda t: 2 * jnp.pi * (0.1122 * jnp.cos(2*jnp.pi*1.0431*Omega*t + 0.7257))
amp_1013 = lambda t: 1.0

# Run simulation
tlist = jnp.linspace(0, tf, 200)
sol = model.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013)

# Analyze infidelity
diagnostics = model.diagnose_infidelity(sol[-1], model.psi0)
print(f"Fidelity: {diagnostics['fidelity']:.4f}")
print(f"Leakage error: {diagnostics['leakage_error']:.6f}")

# Plot population evolution
model.diagnose_plot(tlist, sol, initial_label='11', save_path='population.png')
```

## Physical Model

### 10-Level Atomic Structure

The model includes 10 single-atom levels forming a 100-dimensional two-atom Hilbert space:

| Level | Label | Description | Quantum Numbers |
|-------|-------|-------------|-----------------|
| 0 | `0` | Ground state (qubit 0) | 5S1/2, F=1, mF=0 |
| 1 | `1` | Ground state (qubit 1) | 5S1/2, F=2, mF=0 |
| 2 | `e1` | Intermediate | 6P3/2, F=1, mF=-1 |
| 3 | `e2` | Intermediate | 6P3/2, F=2, mF=-1 |
| 4 | `e3` | Intermediate | 6P3/2, F=3, mF=-1 |
| 5 | `r1` | Target Rydberg | 70S1/2, mJ=-1/2 |
| 6 | `r2` | Unwanted Rydberg | 70S1/2, mJ=+1/2 |
| 7 | `rP` | BBR-induced P state | 70P (thermal) |
| 8 | `L0` | Leakage | 5S1/2, F=1, mF!=0 |
| 9 | `L1` | Leakage | 5S1/2, F=2, mF!=0 |

### Key Physical Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Intermediate detuning (Δ) | -9.1 GHz | 6P3/2 detuning from resonance |
| Effective Rabi frequency (Ω_eff) | 5 MHz | Two-photon coupling strength |
| 6P3/2 decay rate (Γ_mid) | 9.09 MHz | τ = 110 ns lifetime |
| Rydberg radiative decay (Γ_RD) | 0.0042 MHz | τ = 152 μs lifetime |
| BBR decay rate (Γ_BBR) | 0.0024 MHz | τ = 410 μs at 300K |
| Blockade shift | ~1.2 GHz | At d = 3 μm separation |

### Decay Channels

```
                70S1/2 (Rydberg)
                    │
          ┌─────────┼─────────┐
          │ BBR     │ Radiative
          ▼         │         │
        70P (rP)    │         │
                    ▼         ▼
              ┌─────────────────┐
              │    5P states    │
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
        |0⟩          |1⟩      |L0⟩,|L1⟩
     (qubit 0)    (qubit 1)   (leakage)
```

## Results Summary

### Gate Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Raw Fidelity** | 99.51% | Standard CZ fidelity |
| **P00_withL** | **99.54%** | Experimental comparison metric |
| **L0 Hidden Contribution** | 0.031% | Leakage appearing as |0⟩ |
| **Gate Time** | 243.8 ns | |
| **Optimal θ** | 0.452 rad | Z-rotation correction |

> **Note**: For experimental comparison, use **P00_withL** (99.54%) instead of raw fidelity (99.51%). 
> This accounts for |L0⟩ being indistinguishable from |0⟩ in detection.
> Expected experimental value: ~99.56-99.60%

### Infidelity Breakdown

| Error Source | Mean | Std | Contribution |
|--------------|------|-----|--------------|
| **Coherent Error** | 2.89 × 10⁻³ | 6.5 × 10⁻⁴ | 58.6% |
| **Leakage Error** | 1.42 × 10⁻³ | 3.2 × 10⁻⁴ | 28.7% |
| **Rydberg Residual** | 6.24 × 10⁻⁴ | 2.1 × 10⁻⁴ | 12.7% |
| **Decay Error** | ~0 | ~0 | <0.1% |
| **Intermediate Residual** | 0 | 0 | 0% |

### Per-State Fidelities

| Initial State | Raw Fidelity | P00_withL | L0 Contrib | Leakage |
|---------------|--------------|-----------|------------|---------|
| \|00⟩ | 99.52% | 99.54% | 0.026% | 0.13% |
| \|01⟩ | 99.45% | 99.49% | 0.036% | 0.15% |
| \|10⟩ | 99.48% | 99.51% | 0.032% | 0.14% |
| \|11⟩ | 99.49% | 99.52% | 0.030% | 0.14% |
| \|0+⟩ | 99.85% | 99.93% | 0.082% | 0.06% |
| \|0-⟩ | 99.32% | 99.32% | 0.000% | 0.22% |
| \|1+⟩ | 99.51% | 99.54% | 0.026% | 0.13% |
| \|1-⟩ | 99.46% | 99.49% | 0.037% | 0.15% |
| \|+0⟩ | 99.48% | 99.51% | 0.033% | 0.15% |
| \|-0⟩ | 99.49% | 99.52% | 0.030% | 0.14% |
| \|+1⟩ | 99.51% | 99.53% | 0.020% | 0.14% |
| \|-1⟩ | 99.51% | 99.53% | 0.020% | 0.14% |

## Detailed Analysis

### P00_withL Metric Explained

In experiments, the fluorescence detection cannot distinguish between:
- Ground state |0⟩ (5S1/2, F=1, mF=0)
- Leakage state |L0⟩ (5S1/2, F=1, mF≠0)

Both appear as "atom present in F=1" during state-selective detection. The `CZ_fidelity_with_leakage()` method accounts for this by mapping |L0⟩ → |0⟩ in the density matrix before computing fidelity.

```python
# Using the experimental comparison metric
fid_withL, theta, L0_contrib = model.CZ_fidelity_with_leakage(rho_final, psi0)
# fid_withL = 99.54% (use this for experimental comparison)
# L0_contrib = 0.031% (hidden leakage contribution)
```

### 1. Coherent Error (Dominant: 58.6%)

Coherent errors are the largest contributor to infidelity. These arise from:

- **Imperfect Rydberg blockade**: At d=3μm, the blockade shift (~1.2 GHz) is large but finite. Small residual double-excitation amplitude introduces phase errors.

- **AC Stark shifts**: Differential light shifts from 420nm and 1013nm lasers cause state-dependent phase accumulation. The phase-modulated pulse protocol partially compensates this but residual errors remain.

- **Pulse calibration**: The optimized parameters (A=0.1122, ω=1.0431, φ₀=-0.726) minimize but don't eliminate coherent errors.

**Mitigation strategies**:
- Increase atomic separation (stronger blockade)
- Further pulse optimization
- Composite pulse sequences

### 2. Leakage Error (28.7%)

Population leaks from computational states |0⟩, |1⟩ to other hyperfine states |L0⟩, |L1⟩ during:

- **Intermediate state decay**: 6P3/2 states decay to various 5S1/2 hyperfine levels with different branching ratios
- **Rydberg radiative decay**: Similar branching to ground hyperfine manifold

The leakage accumulates monotonically during the gate (see population plots) reaching ~0.1% by gate completion.

**Mitigation strategies**:
- Faster gates (reduce decay probability)
- Optical pumping between gate operations
- Leakage reduction units (LRU)

### 3. Rydberg Residual (12.7%)

At gate completion, ~0.06% population remains in Rydberg states instead of returning to ground states. This is due to:

- **Finite pulse bandwidth**: The Blackman-like pulse envelope has finite spectral width
- **Blockade-induced asymmetry**: The |11⟩ → |rr⟩ pathway is blockade-shifted, causing imperfect population return

The |0+⟩ superposition state shows minimal Rydberg residual (0.005%) while |0-⟩ shows maximum (0.11%), indicating interference effects.

### 4. Decay Error (~0%)

Spontaneous emission causing trace loss is negligible for this gate time:

- Gate time: 244 ns
- 6P3/2 lifetime: 110 ns (but only transiently populated)
- Rydberg lifetime: 152 μs >> gate time

The intermediate states are populated for <50 ns total, and the large detuning Δ = -9.1 GHz suppresses their steady-state population.

### 5. State-Dependent Variation

The superposition states |0±⟩ = (|00⟩ ± |01⟩)/√2 show interesting behavior:

- |0+⟩ has highest fidelity (99.85%) - constructive interference
- |0-⟩ has lowest fidelity (99.32%) - destructive interference

This ~0.5% variation suggests the gate has residual state-dependent phase errors that partially cancel for certain superpositions.

## Population Dynamics

The population evolution plots reveal the gate mechanism:

1. **t = 0-50 ns**: Population transfers from computational basis to intermediate states, then to Rydberg states
2. **t = 50-120 ns**: Peak Rydberg population (~28%) with blockade preventing double excitation
3. **t = 120-244 ns**: Population returns to computational basis via stimulated emission

Key observations:
- Intermediate state population stays below 0.2% (large detuning)
- Rydberg population peaks around t ≈ 140 ns
- Leakage grows linearly throughout the gate
- Trace remains unity (no significant decay)

## Conclusions

The current CZ gate implementation achieves:
- **Raw fidelity**: 99.51%
- **P00_withL (experimental comparison)**: **99.54%**

The ~0.03% difference comes from |L0⟩ leakage being indistinguishable from |0⟩ in detection.

### Error Budget

1. **Coherent errors dominating** (59% of total error) - addressable through pulse optimization
2. **Leakage as secondary concern** (29%) - requires faster gates or active correction
3. **Rydberg residual contributing** (13%) - pulse shaping improvements needed
4. **Decay negligible** (<1%) - gate is fast enough

### Comparison with Experiment

| Metric | Simulation | Expected | Status |
|--------|-----------|----------|--------|
| P00_withL | 99.54% | ~99.56% | Gap: 0.02% |
| Raw fidelity | 99.51% | - | - |

The remaining ~0.02% gap may be due to:
- Unmapped leakage channels (|L1⟩)
- Parameter uncertainties
- Additional coherence effects

### Recommended Improvements

| Priority | Action | Expected Gain |
|----------|--------|---------------|
| High | Optimize pulse for coherent error | +0.2-0.3% |
| Medium | Increase atom separation to 4μm | +0.1% |
| Medium | Add leakage reduction unit | +0.1% |
| Low | Reduce gate time | +0.05% |

Target fidelity with improvements: **99.8-99.9%**

## References

1. Levine et al., "High-Fidelity Control and Entanglement of Rydberg-Atom Qubits", PRL 123, 170503 (2019)
2. Graham et al., "Multi-qubit entanglement and algorithms on a neutral-atom quantum computer", Nature 604, 457 (2022)
3. Project paper: `paper/en_v2.tex`
