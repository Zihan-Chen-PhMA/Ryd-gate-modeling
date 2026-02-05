# CZ Gate Fidelity Simulation Results

*Generated: 2026-01-30T05:53:10.000762*

## Summary

| Metric | Value |
|--------|-------|
| **Mean Raw Fidelity** | 0.995066 |
| **Mean Fidelity (θ-corrected)** | 0.995066 |
| **Infidelity** | 4.93e-03 |
| **Mean θ correction** | 0.4516 rad |

## Simulation Parameters

### Gate Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| t_f | 1.219096 | Gate time (normalized) |
| A | 0.1122 | Phase modulation amplitude |
| ω_f | 1.0431 | Phase modulation frequency |
| φ₀ | -0.72566 rad | Initial phase offset |
| Ω_eff | 5.00 MHz | Effective Rabi frequency |

### Physical Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Interatomic distance | 3.0 μm | Distance between atoms |
| Δ (detuning) | -9100.0 MHz | Intermediate state detuning |
| C₆ | 874000 MHz·μm⁶ | Van der Waals coefficient |
| V (blockade) | 1198.9 MHz | Rydberg blockade shift |
| Ω₄₂₀ | 491.71 MHz | 420 nm Rabi frequency |
| Ω₁₀₁₃ | 185.07 MHz | 1013 nm Rabi frequency |

### Decay Rates

| Parameter | Value | Description |
|-----------|-------|-------------|
| Γ_BBR | 0.002437 MHz | Black-body radiation decay |
| Γ_RD | 0.004162 MHz | Radiative decay |
| Γ_mid | 9.09 MHz | Intermediate state decay |

## Individual State Fidelities

| State | Initial State | Raw Fidelity | θ (rad) | Corrected Fidelity |
|-------|---------------|--------------|---------|-------------------|
| 1 | (|00⟩+|01⟩+|10⟩+|11⟩)/2 | 0.995151 | 0.4510 | 0.995150 |
| 2 | (|00⟩-|01⟩-|10⟩+|11⟩)/2 | 0.994548 | 0.4521 | 0.994547 |
| 3 | (|00⟩+i|01⟩+i|10⟩-|11⟩)/2 | 0.994775 | 0.4504 | 0.994774 |
| 4 | (|00⟩-i|01⟩-i|10⟩-|11⟩)/2 | 0.994926 | 0.4527 | 0.994925 |
| 5 | |00⟩ | 0.998500 | 0.4683 | 0.998500 |
| 6 | |11⟩ | 0.993208 | 0.7992 | 0.993208 |
| 7 | (|00⟩+|01⟩+|10⟩-|11⟩)/2 | 0.995146 | 0.4515 | 0.995146 |
| 8 | (|00⟩-|01⟩-|10⟩-|11⟩)/2 | 0.994567 | 0.4516 | 0.994567 |
| 9 | (|00⟩+i|01⟩+i|10⟩+|11⟩)/2 | 0.994787 | 0.4525 | 0.994787 |
| 10 | (|00⟩-i|01⟩-i|10⟩+|11⟩)/2 | 0.994929 | 0.4507 | 0.994929 |
| 11 | (|00⟩+i|11⟩)/√2 | 0.995129 | 0.4516 | 0.995129 |
| 12 | (|00⟩-i|11⟩)/√2 | 0.995128 | 0.4516 | 0.995128 |

## Error Analysis

The main sources of error in the CZ gate include:

1. **Spontaneous emission from Rydberg states** (Γ_BBR + Γ_RD ≈ 6.598 kHz)
2. **Intermediate state scattering** (Γ_mid = 9.09 MHz)
3. **Imperfect blockade** (V/Ω_eff = 239.8)

## Computation Time

| Phase | Time |
|-------|------|
| Model initialization | 3.40 s |
| Density matrix evolution | 444.21 s |
| **Total** | 450.66 s |

## References

- Evered *et al.*, "High-fidelity parallel entangling gates on a neutral-atom quantum computer", Nature **622**, 268 (2023)
- See `paper/en_v2.tex` for detailed theoretical derivations

---
*Simulation performed using JAX-accelerated density matrix evolution with full error model including Rydberg blockade, spontaneous decay, and AC Stark shifts.*
