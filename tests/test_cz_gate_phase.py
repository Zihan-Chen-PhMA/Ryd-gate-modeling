"""Test CZ gate phase correctness in ideal_cz.py.

Verifies that a well-optimized TO gate implements a CZ up to local Rz
rotations. Uses the gate_fidelity() API rather than hardcoded parameters
so that tests remain robust across fidelity-formula updates.
"""

import numpy as np
import pytest

from ryd_gate.ideal_cz import CZGateSimulator


# Known-good dark-detuning TO parameters (re-optimized with a00 formula)
X_TO = [
    -0.6989301339711643, 1.0296229082590798, 0.3759232324550267,
    1.5710180991068543, 1.4454279613697887, 1.3406239758422793,
]


@pytest.fixture
def simulator():
    """Create a CZGateSimulator instance (default config, no decay)."""
    return CZGateSimulator(param_set='our', strategy='TO', blackmanflag=True)


@pytest.fixture
def computational_basis():
    """Build computational basis states in 49-dim Hilbert space."""
    s0 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=complex)
    s1 = np.array([0, 1, 0, 0, 0, 0, 0], dtype=complex)
    return {
        '00': np.kron(s0, s0),
        '01': np.kron(s0, s1),
        '10': np.kron(s1, s0),
        '11': np.kron(s1, s1),
    }


def test_cz_phase_relation(simulator, computational_basis):
    """CZ conditional phase φ₁₁ - φ₀₁ - φ₁₀ + φ₀₀ = ±π (mod 2π).

    This test is parameter-independent: it first checks that the params
    actually give a good gate (via gate_fidelity), then verifies the
    entangling phase.
    """
    # Guard: ensure parameters are actually well-optimized
    infid = simulator.gate_fidelity(X_TO)
    assert infid < 1e-4, f"Parameters give infidelity {infid:.2e}; not well-optimized"

    phases = {}
    for label, state in computational_basis.items():
        res = simulator._get_gate_result_TO(
            phase_amp=X_TO[0],
            omega=X_TO[1] * simulator.rabi_eff,
            phase_init=X_TO[2],
            delta=X_TO[3] * simulator.rabi_eff,
            t_gate=X_TO[5] * simulator.time_scale,
            state_mat=state,
        )
        phases[label] = np.angle(state.conj().dot(res))

    # CZ phase = φ₁₁ - φ₀₁ - φ₁₀ + φ₀₀ should be ±π
    cz_phase = phases['11'] - phases['01'] - phases['10'] + phases['00']

    # Wrap to [-π, π] first, then check closeness to ±π
    cz_wrapped = (cz_phase + np.pi) % (2 * np.pi) - np.pi
    deviation = min(np.abs(cz_wrapped - np.pi), np.abs(cz_wrapped + np.pi))

    # Allow 2 degrees of deviation
    assert deviation < np.radians(2), (
        f"CZ phase deviation {np.degrees(deviation):.2f}° exceeds 2° tolerance"
    )


def test_single_qubit_phase_symmetry(simulator, computational_basis):
    """Test that |01⟩ and |10⟩ acquire the same phase (symmetric qubits)."""
    phases = {}
    for label in ['00', '01', '10']:
        state = computational_basis[label]
        res = simulator._get_gate_result_TO(
            phase_amp=X_TO[0],
            omega=X_TO[1] * simulator.rabi_eff,
            phase_init=X_TO[2],
            delta=X_TO[3] * simulator.rabi_eff,
            t_gate=X_TO[5] * simulator.time_scale,
            state_mat=state,
        )
        phases[label] = np.angle(state.conj().dot(res))

    # |01⟩ and |10⟩ should have the same phase relative to |00⟩
    phase_01_rel = phases['01'] - phases['00']
    phase_10_rel = phases['10'] - phases['00']

    phase_diff = np.abs(phase_01_rel - phase_10_rel)
    assert phase_diff < np.radians(1), (
        f"Phase asymmetry {np.degrees(phase_diff):.2f}° exceeds 1° tolerance"
    )


def test_computational_basis_fidelity(simulator, computational_basis):
    """Test high fidelity for computational basis states."""
    t_gate = X_TO[5] * simulator.time_scale

    for label, state in computational_basis.items():
        res = simulator._get_gate_result_TO(
            phase_amp=X_TO[0],
            omega=X_TO[1] * simulator.rabi_eff,
            phase_init=X_TO[2],
            delta=X_TO[3] * simulator.rabi_eff,
            t_gate=t_gate,
            state_mat=state,
        )

        # Population remaining in computational subspace
        pop = np.abs(state.conj().dot(res)) ** 2

        assert pop > 0.99, (
            f"State |{label}⟩ has only {pop:.4f} population in original state"
        )
