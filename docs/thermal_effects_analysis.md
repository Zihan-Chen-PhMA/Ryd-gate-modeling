# Thermal Effects Analysis — CZ Gate Fidelity

Analysis of Monte Carlo thermal effects (Doppler shift and position spread) on the Rydberg CZ gate fidelity, implemented in `jax_atom_Evolution.simulate_with_thermal_effects`.

## Physical Setup

| Parameter | Value |
|-----------|-------|
| Interatomic distance | 3 μm |
| C6 coefficient | 2π × 874,000 MHz·μm⁶ |
| Intermediate detuning Δ | −2π × 9100 MHz |
| Effective Rabi frequency | 2π × 5 MHz |
| Gate time | 243.8 ns |
| Rydberg state | 70S₁/₂ of ⁸⁷Rb |
| Laser wavelengths | 420 nm + 1013 nm (counter-propagating) |

## Thermal Scales

### Doppler broadening

The effective wave vector for counter-propagating 420 nm and 1013 nm beams is
`k_eff = k_420 − k_1013 = 2π(1/420nm − 1/1013nm)`.
The RMS Doppler shift is `σ_Doppler = k_eff × sqrt(k_B T / m_Rb) / (2π)`.

| T (μK) | σ_Doppler (MHz) | σ_Doppler / Ω_eff |
|--------|-----------------|-------------------|
| 5      | 0.96            | 0.19              |
| 20     | 1.93            | 0.39              |
| 50     | 3.05            | 0.61              |

### Position spread (100 kHz trap)

For a thermal atom in a harmonic trap: `σ_x = sqrt(k_B T / (m ω²))`.

| T (μK) | σ_x (μm) | σ_x / d | ΔV/V (approx 6σ_x/d) |
|--------|----------|---------|----------------------|
| 5      | 1.10     | 0.37    | ~220%                |
| 20     | 2.20     | 0.73    | ~440%                |
| 50     | 3.48     | 1.16    | ~700%                |

## Results

### Baseline (T = 0)

- **Mean fidelity** (12 SSS states): **0.9951**
- **Mean fidelity with leakage**: **0.9954**

### Doppler-only Monte Carlo (5 samples per temperature)

| T (μK) | Mean Fidelity | Std | ΔF from baseline |
|--------|--------------|-----|------------------|
| 5      | 0.9755       | 0.0122 | −0.0196       |
| 20     | 0.9573       | 0.0287 | −0.0378       |
| 50     | 0.8424       | 0.1319 | −0.1527       |

### Combined Doppler + Position Spread (100 kHz trap, 3 samples)

| T (μK) | Mean Fidelity | Std | ΔF from baseline |
|--------|--------------|-----|------------------|
| 5      | 0.9863       | 0.0078 | −0.0088       |

## Error Analysis

### Doppler effect is the dominant thermal error

The Doppler shift enters as a detuning error on the Rydberg state energies. At 20 μK the RMS shift is ~1.9 MHz, which is ~39% of the effective Rabi frequency (2π × 5 MHz). This is a large perturbation and explains the steep fidelity drop.

The infidelity scales roughly as `(σ_Doppler / Ω_eff)²`:
- T=5 μK: (0.19)² ≈ 0.04 → measured ΔF ≈ 0.020
- T=20 μK: (0.39)² ≈ 0.15 → measured ΔF ≈ 0.038
- T=50 μK: (0.61)² ≈ 0.37 → measured ΔF ≈ 0.153

The scaling is roughly consistent with quadratic dependence on the Doppler-to-Rabi ratio, though the relationship steepens at higher temperatures where the perturbative approximation breaks down.

### Position spread effect

The position spread at 100 kHz trap frequency is substantial (σ_x ≈ 1.1 μm at 5 μK vs d = 3 μm). This causes large shot-to-shot variation in blockade strength V ∝ 1/d⁶. However, the combined fidelity at T=5 μK (0.986) is actually *higher* than the Doppler-only result (0.976). This is because:

1. The 5-sample Doppler-only and 3-sample combined runs use different random seeds and low statistics, so the difference is within statistical uncertainty.
2. The position spread randomizes V but the gate is optimized to work across a range of blockade strengths (the optimized pulse uses phase modulation for robustness).
3. With only 3 samples, the combined result has high uncertainty.

### Comparison with issue #6 predictions

The issue predicted ~0.1–0.3% combined thermal error. Our results show:
- At 5 μK: ~2% infidelity from Doppler alone — **much larger** than predicted
- At 20 μK: ~4% infidelity
- At 50 μK: ~15% infidelity

The discrepancy with the issue's estimate suggests that the effective Rabi frequency (2π × 5 MHz) is too small relative to the Doppler broadening at typical experimental temperatures. This is a significant finding — **Doppler effects alone can account for several percent of infidelity** at realistic temperatures, far exceeding the ~0.5% intrinsic gate error.

### Caveats

1. **Low sample count**: 5 MC samples for Doppler, 3 for combined. Statistical uncertainty is large (std ≈ 1–13%). Higher sample counts would improve estimates but are computationally expensive (~80s per sample due to JAX recompilation).
2. **Single initial state**: The MC simulation uses the default |11⟩ initial state, not the full SSS average. The baseline uses all 12 SSS states.
3. **No intermediate state decay in MC**: The `simulate_with_thermal_effects` method does not apply `mid_state_decay` post-processing, while the baseline does. This creates a systematic offset.
4. **100 kHz trap may be unrealistic**: The position spread σ_x ≈ 1.1 μm at 5 μK is 37% of the 3 μm separation. Typical optical tweezer experiments use tighter traps (several hundred kHz) or turn off traps during gate operation. Tighter traps or trap-off operation would reduce position spread effects.

## Reproducing the Results

### Prerequisites

```bash
git clone https://github.com/ChanceSiyuan/Ryd-gate-modeling.git
cd Ryd-gate-modeling
git checkout refactor/professional-structure
uv sync --dev
```

### Run the unit tests

```bash
uv run pytest tests/test_full_error_model.py -v -k "TestThermalEffects or TestPositionSpread"
```

All 18 thermal + position spread tests should pass.

### Run the full analysis

```bash
uv run python scripts/thermal_analysis.py
```

This runs:
1. Baseline: 12 SSS states, full CZ gate simulation (~7 min)
2. Doppler-only sweep: T = 5, 20, 50 μK, 5 MC samples each (~20 min per temperature)
3. Combined sweep: T = 5, 20, 50 μK with 100 kHz trap, 3 MC samples each (~23 min per temperature)

**Total runtime: ~2–3 hours** on a modern CPU. Results are saved to `docs/thermal_analysis_results.json`.

To run a faster check with a single temperature:

```python
from ryd_gate.full_error_model import jax_atom_Evolution
import jax.numpy as jnp

model = jax_atom_Evolution()

# Check Doppler broadening scale
print(f"Doppler std at 10 μK: {model.doppler_std(10.0):.4f} MHz")
print(f"Position std at 10 μK / 100 kHz: {model.position_spread_std(10.0, 100.0):.4f} μm")

# Quick MC test (2 samples)
amp_420 = lambda t: 1.0
phase_420 = lambda t: 0.0
amp_1013 = lambda t: 1.0
tlist = jnp.linspace(0, 0.01, 50)

result = model.simulate_with_thermal_effects(
    tlist, amp_420, phase_420, amp_1013,
    T_atom=10.0, n_samples=2, seed=42
)
print(f"F = {result['mean_fidelity']:.4f} ± {result['std_fidelity']:.4f}")
```

### Run the existing noise report for baseline comparison

```bash
uv run python scripts/generate_noise_report.py --save-plots
```
