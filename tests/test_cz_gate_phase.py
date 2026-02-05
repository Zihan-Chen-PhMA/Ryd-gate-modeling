"""Test CZ gate phase correctness in ideal_cz.py.

Verifies that the TO pulse parameters produce a gate equivalent to CZ
up to local Rz rotations.
"""

import numpy as np
import pytest

from ryd_gate.ideal_cz import CZGateSimulator


# Optimized TO pulse parameters
X_TO = [-0.64168872, 1.14372811, 0.35715965, 1.51843443, 2.96448688, 1.21214853]


@pytest.fixture
def simulator():
    """Create a CZGateSimulator instance."""
    return CZGateSimulator(
        decayflag=False, param_set='our', strategy='TO', blackmanflag=False
    )


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
    """Test that the gate implements CZ up to local phases.

    For a CZ gate: φ₁₁ - φ₀₁ - φ₁₀ + φ₀₀ = ±π
    """
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
        phases[label] = np.angle(state.conj().dot(res[:, -1]))

    # CZ phase = φ₁₁ - φ₀₁ - φ₁₀ + φ₀₀ should be ±π
    cz_phase = phases['11'] - phases['01'] - phases['10'] + phases['00']

    # Account for 2π periodicity
    deviation = min(np.abs(cz_phase - np.pi), np.abs(cz_phase + np.pi))

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
        phases[label] = np.angle(state.conj().dot(res[:, -1]))

    # |01⟩ and |10⟩ should have the same phase relative to |00⟩
    phase_01_rel = phases['01'] - phases['00']
    phase_10_rel = phases['10'] - phases['00']

    phase_diff = np.abs(phase_01_rel - phase_10_rel)
    assert phase_diff < np.radians(1), (
        f"Phase asymmetry {np.degrees(phase_diff):.2f}° exceeds 1° tolerance"
    )


def test_computational_basis_fidelity(simulator, computational_basis):
    """Test high fidelity for computational basis states."""
    theta = X_TO[4]
    t_gate = X_TO[5] * simulator.time_scale

    for label, state in computational_basis.items():
        res = simulator._get_gate_result_TO(
            phase_amp=X_TO[0],
            omega=X_TO[1] * simulator.rabi_eff,
            phase_init=X_TO[2],
            delta=X_TO[3] * simulator.rabi_eff,
            t_gate=t_gate,
            state_mat=state,
        )[:, -1]

        # Population remaining in computational subspace
        pop = np.abs(state.conj().dot(res)) ** 2

        assert pop > 0.99, (
            f"State |{label}⟩ has only {pop:.4f} population in original state"
        )
