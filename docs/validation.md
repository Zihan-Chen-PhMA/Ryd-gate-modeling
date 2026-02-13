# Solver Validation

This document describes the validation test suite that verifies the correctness of the `CZGateSimulator` implementation.

## Overview

The `ryd-gate` package provides a Schrödinger equation solver for CZ gate simulation:

| Solver | Module | Hilbert Space | Method |
|--------|--------|---------------|--------|
| **Schrödinger** | `ryd_gate.ideal_cz` | 49-dim (7-level × 2 atoms) | SciPy `solve_ivp` (DOP853) |

## Test Categories

### 1. SSS State Construction

The Symmetric State Set (SSS) consists of 12 specific two-qubit input states that span the computational subspace:

```python
# The 12 SSS states include:
SSS-0: 0.5(|00⟩ + |01⟩ + |10⟩ + |11⟩)   # Equal superposition
SSS-1: 0.5(|00⟩ - |01⟩ - |10⟩ + |11⟩)   # Bell-like
SSS-4: |00⟩                              # Computational basis
SSS-5: |11⟩                              # Computational basis
# ... and 8 more with various phase factors
```

**Validated properties:**
- Exactly 12 SSS states
- Each state normalized to 1
- States only occupy |0⟩, |1⟩ computational basis

### 2. State Evolution Properties

**Tests:**
- `test_normalization_preserved` — ||ψ|| = 1 throughout evolution (no decay)
- `test_normalization_decreases_with_decay` — ||ψ|| decreases when decay enabled

### 3. Parameter Validation

Verifies that physical parameters are within physically reasonable ranges:

| Parameter | Acceptable Range | Typical Value |
|-----------|-----------------|---------------|
| Effective Rabi frequency | 1-20 MHz | ~7 MHz |
| Intermediate detuning | 1-20 GHz | ~6 GHz |
| Blockade strength (d=3μm) | 0.1-10 GHz | ~1.2 GHz |

### 4. Fidelity Validation

**Tests:**
- `test_schrodinger_fidelity_bounds` — 0 ≤ F ≤ 1
- `test_ideal_cz_fidelity_is_one[0-11]` — Ideal CZ gives F ≈ 1 for all 12 SSS states

## How to Run Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run fast tests only
uv run pytest tests/ -v -m "not slow"

# Run with coverage
uv run pytest tests/ --cov=ryd_gate --cov-report=html
```

### Run Specific Test Files

```bash
# Schrödinger solver tests
uv run pytest tests/test_ideal_cz.py -v

# Phase and SSS tests
uv run pytest tests/test_cz_gate_phase.py -v

# Package init tests
uv run pytest tests/test_init.py -v
```

## Numerical Tolerances

| Property | Tolerance | Justification |
|----------|-----------|---------------|
| State normalization | 1e-10 | Direct computation |
| Fidelity bounds | ±1e-10 | Floating point precision |
| ODE integration | rtol=1e-8, atol=1e-12 | DOP853 adaptive stepping |

## Key Findings

The validation suite confirms:

1. **SSS States Correct** — All 12 states are properly constructed in the computational subspace

2. **Norm Preservation** — State norm is preserved without decay and monotonically decreases with decay enabled

3. **Ideal CZ Verified** — Applying the ideal CZ gate to any SSS state produces fidelity F ≈ 1.0

## References

- [PRX Quantum: Benchmarking and Fidelity Response Theory of High-Fidelity Rydberg Entangling Gates](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.6.010331)
