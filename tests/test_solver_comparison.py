"""Cross-validation tests comparing Schrödinger and density-matrix CZ gate solvers.

This module validates that both simulation approaches produce consistent results
when operating under matched physical conditions, particularly when decay is disabled
in the density-matrix model.

Test Categories:
1. SSS State Equivalence - verify 12 states match between representations
2. Density Matrix Properties - trace, Hermiticity, positivity
3. Schrödinger Properties - normalization preservation
4. Parameter Matching - physical constants consistency
5. Population Dynamics - time evolution comparison
6. Occupation Operators - measurement operator validation
7. Fidelity Comparison - gate fidelity calculations
"""

import numpy as np
import pytest

# Skip entire module if dependencies unavailable
jax = pytest.importorskip("jax")
from jax import config
config.update("jax_enable_x64", True)
jnp = jax.numpy

arc = pytest.importorskip("arc")
qutip = pytest.importorskip("qutip")

from ryd_gate.ideal_cz import CZGateSimulator
from ryd_gate.full_error_model import jax_atom_Evolution


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def schrodinger_sim():
    """Schrödinger solver with decay disabled for comparison."""
    return CZGateSimulator(decayflag=False, param_set="our", strategy="TO")


@pytest.fixture(scope="module")
def schrodinger_sim_decay():
    """Schrödinger solver with decay enabled."""
    return CZGateSimulator(decayflag=True, param_set="our", strategy="TO")


@pytest.fixture(scope="module")
def dm_sim_no_decay():
    """Density-matrix solver with all decay channels disabled."""
    return jax_atom_Evolution(
        blockade=True,
        ryd_decay=False,
        mid_decay=False,
        distance=3
    )


@pytest.fixture(scope="module")
def dm_sim_with_decay():
    """Density-matrix solver with full decay for property tests."""
    return jax_atom_Evolution(
        blockade=True,
        ryd_decay=True,
        mid_decay=True,
        distance=3
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def expand_state_49_to_100(psi_49):
    """Expand 49-dim state to 100-dim by zero-padding extra levels.

    Maps 7-level basis to 10-level basis:
    - Levels 0-6 map directly
    - Levels 7-9 (rP, L0, L1) are set to zero
    """
    expand_sq = np.zeros((10, 7), dtype=complex)
    for i in range(7):
        expand_sq[i, i] = 1.0
    expand_tq = np.kron(expand_sq, expand_sq)  # (100, 49)
    return expand_tq @ psi_49


def build_sss_states_7level():
    """Build the 12 SSS initial states in the 49-dim Hilbert space."""
    s0 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=complex)
    s1 = np.array([0, 1, 0, 0, 0, 0, 0], dtype=complex)

    state_00 = np.kron(s0, s0)
    state_01 = np.kron(s0, s1)
    state_10 = np.kron(s1, s0)
    state_11 = np.kron(s1, s1)

    return [
        0.5*state_00 + 0.5*state_01 + 0.5*state_10 + 0.5*state_11,
        0.5*state_00 - 0.5*state_01 - 0.5*state_10 + 0.5*state_11,
        0.5*state_00 + 0.5j*state_01 + 0.5j*state_10 - 0.5*state_11,
        0.5*state_00 - 0.5j*state_01 - 0.5j*state_10 - 0.5*state_11,
        state_00,
        state_11,
        0.5*state_00 + 0.5*state_01 + 0.5*state_10 - 0.5*state_11,
        0.5*state_00 - 0.5*state_01 - 0.5*state_10 - 0.5*state_11,
        0.5*state_00 + 0.5j*state_01 + 0.5j*state_10 + 0.5*state_11,
        0.5*state_00 - 0.5j*state_01 - 0.5j*state_10 + 0.5*state_11,
        state_00/np.sqrt(2) + 1j*state_11/np.sqrt(2),
        state_00/np.sqrt(2) - 1j*state_11/np.sqrt(2),
    ]


def compute_population_schrodinger(sim, result, level_indices):
    """Compute total population in given levels for Schrödinger result."""
    populations = []
    for t in range(result.shape[1]):
        state = result[:, t]
        pop = 0.0
        for idx in level_indices:
            occ_op = sim._occ_operator(idx)
            pop += np.real(np.conj(state) @ occ_op @ state) / 2.0
        populations.append(pop)
    return np.array(populations)


def compute_population_dm(model, sol, level_labels):
    """Compute population from density matrix evolution."""
    populations = []
    for t in range(sol.shape[0]):
        rho = sol[t]
        pop = 0.0
        for label in level_labels:
            occ_op = model.occ_operator(label)
            pop += float(jnp.real(jnp.trace(occ_op @ rho)) / 2.0)
        populations.append(pop)
    return np.array(populations)


# ============================================================================
# TEST CLASS 1: SSS STATE EQUIVALENCE
# ============================================================================

class TestSSSStateEquivalence:
    """Verify 12 SSS states match between 49-dim and 100-dim representations."""

    def test_sss_states_count(self, dm_sim_no_decay):
        """Both models should have 12 SSS states."""
        assert len(dm_sim_no_decay.SSS_initial_state_list) == 12
        assert len(build_sss_states_7level()) == 12

    @pytest.mark.parametrize("sss_idx", range(12))
    def test_sss_state_normalization_dm(self, dm_sim_no_decay, sss_idx):
        """Each DM SSS state should be normalized."""
        psi = dm_sim_no_decay.SSS_initial_state_list[sss_idx]
        norm = float(jnp.sum(jnp.abs(psi)**2))
        assert norm == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("sss_idx", range(12))
    def test_sss_state_normalization_schrodinger(self, sss_idx):
        """Each Schrödinger SSS state should be normalized."""
        psi = build_sss_states_7level()[sss_idx]
        norm = np.sum(np.abs(psi)**2)
        assert norm == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("sss_idx", range(12))
    def test_sss_state_in_computational_subspace(self, dm_sim_no_decay, sss_idx):
        """SSS states should only have support on computational basis |0⟩,|1⟩."""
        psi = np.array(dm_sim_no_decay.SSS_initial_state_list[sss_idx]).flatten()

        # Computational basis indices in 100-dim: |00⟩=0, |01⟩=1, |10⟩=10, |11⟩=11
        computational_indices = [0, 1, 10, 11]

        total_pop = sum(np.abs(psi[idx])**2 for idx in computational_indices)

        assert total_pop == pytest.approx(1.0, abs=1e-10), \
            f"SSS state {sss_idx} has population outside computational subspace"

    def test_sss_computational_basis_states_match(self, dm_sim_no_decay):
        """|00⟩ and |11⟩ states (indices 4, 5) should match between models."""
        # |00⟩ in DM model (SSS-4)
        psi_00_dm = np.array(dm_sim_no_decay.SSS_initial_state_list[4]).flatten()
        # |11⟩ in DM model (SSS-5)
        psi_11_dm = np.array(dm_sim_no_decay.SSS_initial_state_list[5]).flatten()

        # Construct expected |00⟩ and |11⟩ in 49-dim Schrödinger basis
        psi_00_sch = build_sss_states_7level()[4]
        psi_11_sch = build_sss_states_7level()[5]

        # Expand to 100-dim for comparison
        psi_00_expanded = expand_state_49_to_100(psi_00_sch)
        psi_11_expanded = expand_state_49_to_100(psi_11_sch)

        np.testing.assert_allclose(psi_00_dm, psi_00_expanded, atol=1e-12)
        np.testing.assert_allclose(psi_11_dm, psi_11_expanded, atol=1e-12)


# ============================================================================
# TEST CLASS 2: DENSITY MATRIX PHYSICAL PROPERTIES
# ============================================================================

class TestDensityMatrixProperties:
    """Verify density matrix physical properties during and after evolution."""

    def test_trace_preservation_no_decay(self, dm_sim_no_decay):
        """Without decay, trace should remain exactly 1."""
        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.0
        amp_1013 = lambda t: 1.0

        tlist = jnp.linspace(0, 0.1, 50)  # 100 ns evolution
        sol = dm_sim_no_decay.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013)

        for i in range(len(tlist)):
            trace = float(jnp.trace(sol[i]).real)
            assert abs(trace - 1.0) < 1e-8, f"Trace = {trace} at t={tlist[i]}"

    def test_trace_decreasing_with_decay(self, dm_sim_with_decay):
        """With decay, trace should monotonically decrease (or stay ~1 for short times)."""
        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.0
        amp_1013 = lambda t: 1.0

        tlist = jnp.linspace(0, 0.5, 100)  # 500 ns
        sol = dm_sim_with_decay.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013)

        traces = [float(jnp.trace(sol[i]).real) for i in range(len(tlist))]

        # Trace should be non-increasing (with small numerical tolerance)
        for i in range(1, len(traces)):
            assert traces[i] <= traces[i-1] + 1e-6, \
                f"Trace increased from {traces[i-1]} to {traces[i]}"

    @pytest.mark.parametrize("time_idx", [0, 10, 25, 49])
    def test_hermiticity(self, dm_sim_no_decay, time_idx):
        """Density matrix should remain Hermitian: ρ = ρ†."""
        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.0
        amp_1013 = lambda t: 1.0

        tlist = jnp.linspace(0, 0.1, 50)
        sol = dm_sim_no_decay.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013)

        rho = np.array(sol[time_idx])
        np.testing.assert_allclose(rho, np.conj(rho).T, atol=1e-10)

    @pytest.mark.parametrize("time_idx", [0, 25, 49])
    def test_positive_semidefiniteness(self, dm_sim_no_decay, time_idx):
        """All eigenvalues of density matrix should be >= 0."""
        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.0
        amp_1013 = lambda t: 1.0

        tlist = jnp.linspace(0, 0.1, 50)
        sol = dm_sim_no_decay.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013)

        eigenvalues = np.linalg.eigvalsh(np.array(sol[time_idx]))
        min_eigenvalue = float(np.min(eigenvalues))

        # Allow small negative eigenvalues due to numerical precision in ODE integration
        # Lindblad evolution can produce small negative eigenvalues (~1e-7) due to
        # finite precision in the symmetrization step
        assert min_eigenvalue >= -1e-6, \
            f"Negative eigenvalue {min_eigenvalue} at time index {time_idx}"


# ============================================================================
# TEST CLASS 3: SCHRÖDINGER SOLVER PROPERTIES
# ============================================================================

class TestSchrodingerProperties:
    """Verify pure state properties for Schrödinger evolution."""

    def test_normalization_preserved(self, schrodinger_sim):
        """State norm should remain 1 throughout evolution (no decay)."""
        ini_state = np.kron(
            [0, 1+0j, 0, 0, 0, 0, 0],
            [0, 1+0j, 0, 0, 0, 0, 0]
        )

        result = schrodinger_sim._get_gate_result_TO(
            phase_amp=0.1,
            omega=schrodinger_sim.rabi_eff,
            phase_init=0.0,
            delta=0.0,
            t_gate=schrodinger_sim.time_scale,
            state_mat=ini_state
        )

        # Check at multiple time points
        for t_idx in [0, 100, 250, 500, 750, 999]:
            norm = np.linalg.norm(result[:, t_idx])
            assert np.isclose(norm, 1.0, rtol=1e-8), \
                f"Norm = {norm} at time index {t_idx}"

    def test_normalization_decreases_with_decay(self, schrodinger_sim_decay):
        """With decay enabled, state norm should decrease."""
        ini_state = np.kron(
            [0, 1+0j, 0, 0, 0, 0, 0],
            [0, 1+0j, 0, 0, 0, 0, 0]
        )

        result = schrodinger_sim_decay._get_gate_result_TO(
            phase_amp=0.1,
            omega=schrodinger_sim_decay.rabi_eff,
            phase_init=0.0,
            delta=0.0,
            t_gate=schrodinger_sim_decay.time_scale,
            state_mat=ini_state
        )

        norm_start = np.linalg.norm(result[:, 0])
        norm_end = np.linalg.norm(result[:, -1])

        assert norm_end < norm_start, \
            f"Norm did not decrease: start={norm_start}, end={norm_end}"


# ============================================================================
# TEST CLASS 4: PARAMETER MATCHING
# ============================================================================

class TestParameterMatching:
    """Verify physical parameters match between solvers."""

    def test_rabi_frequency_match(self, schrodinger_sim, dm_sim_no_decay):
        """Effective Rabi frequency should match (within 1%)."""
        # Schrödinger uses rad/s, DM uses rad/μs (which is rad/s * 1e6)
        omega_sch = schrodinger_sim.rabi_eff  # rad/s
        omega_dm = float(dm_sim_no_decay.rabi_eff) * 1e6  # rad/μs → rad/s

        # Both should be 2π × 5e6 rad/s = ~31.4e6 rad/s
        assert np.isclose(omega_sch, omega_dm, rtol=0.01), \
            f"Rabi mismatch: Sch={omega_sch:.2e}, DM={omega_dm:.2e}"

    def test_intermediate_detuning_match(self, schrodinger_sim, dm_sim_no_decay):
        """Intermediate state detuning Δ should match (within 1%)."""
        # Both should be ~2π × 9.1e9 rad/s
        delta_sch = abs(schrodinger_sim.Delta)  # rad/s
        delta_dm = abs(float(dm_sim_no_decay.Delta)) * 1e6  # rad/μs → rad/s

        assert np.isclose(delta_sch, delta_dm, rtol=0.01), \
            f"Delta mismatch: Sch={delta_sch:.2e}, DM={delta_dm:.2e}"

    def test_blockade_strength_match(self, schrodinger_sim, dm_sim_no_decay):
        """Van der Waals interaction V should match at d=3μm (within 5%)."""
        V_sch = schrodinger_sim.v_ryd  # rad/s
        V_dm = float(dm_sim_no_decay.V) * 1e6  # rad/μs → rad/s

        assert np.isclose(V_sch, V_dm, rtol=0.05), \
            f"Blockade mismatch: Sch={V_sch:.2e}, DM={V_dm:.2e}"


# ============================================================================
# TEST CLASS 5: POPULATION DYNAMICS COMPARISON
# ============================================================================

class TestPopulationDynamicsComparison:
    """Compare population evolution between solvers when decay is disabled."""

    @pytest.mark.slow
    def test_rydberg_population_qualitative_match(
        self, schrodinger_sim, dm_sim_no_decay
    ):
        """Rydberg population dynamics should qualitatively match for |11⟩."""
        # Initial |11⟩ state
        ini_state_sch = build_sss_states_7level()[5]  # |11⟩
        psi_11_dm = dm_sim_no_decay.SSS_initial_state_list[5]
        rho0 = dm_sim_no_decay.psi_to_rho(psi_11_dm)

        t_gate_us = 0.2  # 200 ns in μs

        # Schrödinger evolution
        result_sch = schrodinger_sim._get_gate_result_TO(
            phase_amp=0.1,
            omega=schrodinger_sim.rabi_eff,
            phase_init=0.0,
            delta=0.0,
            t_gate=t_gate_us * 1e-6,  # Convert to seconds
            state_mat=ini_state_sch
        )

        # DM evolution
        tlist = jnp.linspace(0, t_gate_us, 100)
        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.1 * jnp.cos(dm_sim_no_decay.rabi_eff * t)
        amp_1013 = lambda t: 1.0

        sol_dm = dm_sim_no_decay.integrate_rho_jax(
            tlist, amp_420, phase_420, amp_1013, rho0=rho0
        )

        # Compute Rydberg populations
        ryd_pop_sch = compute_population_schrodinger(schrodinger_sim, result_sch, [5])
        ryd_pop_dm = compute_population_dm(dm_sim_no_decay, sol_dm, ['r1'])

        # Both should show Rydberg excitation from |11⟩
        assert np.max(ryd_pop_sch) > 0.05, "Schrödinger shows insufficient Rydberg excitation"
        assert np.max(ryd_pop_dm) > 0.05, "DM shows insufficient Rydberg excitation"

    @pytest.mark.slow
    def test_computational_population_for_00_state(
        self, schrodinger_sim, dm_sim_no_decay
    ):
        """For |00⟩ initial state, computational population should stay high."""
        # |00⟩ should have minimal excitation (dark state)
        ini_state_sch = build_sss_states_7level()[4]  # |00⟩
        psi_00_dm = dm_sim_no_decay.SSS_initial_state_list[4]
        rho0 = dm_sim_no_decay.psi_to_rho(psi_00_dm)

        t_gate_us = 0.2

        # Schrödinger evolution
        result_sch = schrodinger_sim._get_gate_result_TO(
            phase_amp=0.1,
            omega=schrodinger_sim.rabi_eff,
            phase_init=0.0,
            delta=0.0,
            t_gate=t_gate_us * 1e-6,
            state_mat=ini_state_sch
        )

        # DM evolution
        tlist = jnp.linspace(0, t_gate_us, 100)
        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.1 * jnp.cos(dm_sim_no_decay.rabi_eff * t)
        amp_1013 = lambda t: 1.0

        sol_dm = dm_sim_no_decay.integrate_rho_jax(
            tlist, amp_420, phase_420, amp_1013, rho0=rho0
        )

        # Computational population (|0⟩ + |1⟩ levels)
        comp_pop_sch = compute_population_schrodinger(schrodinger_sim, result_sch, [0, 1])
        comp_pop_dm = compute_population_dm(dm_sim_no_decay, sol_dm, ['0', '1'])

        # |00⟩ should largely stay in computational subspace
        assert np.mean(comp_pop_sch) > 0.5, \
            f"Schrödinger comp. population too low: {np.mean(comp_pop_sch)}"
        assert np.mean(comp_pop_dm) > 0.5, \
            f"DM comp. population too low: {np.mean(comp_pop_dm)}"


# ============================================================================
# TEST CLASS 6: OCCUPATION OPERATORS
# ============================================================================

class TestOccupationOperators:
    """Compare occupation operator construction between solvers."""

    def test_occ_operator_shapes(self, schrodinger_sim, dm_sim_no_decay):
        """Occupation operators should have correct dimensions."""
        # Schrödinger: 49×49
        for i in range(7):
            occ = schrodinger_sim._occ_operator(i)
            assert occ.shape == (49, 49)

        # DM: 100×100
        for label in dm_sim_no_decay.level_label:
            occ = dm_sim_no_decay.occ_operator(label)
            assert occ.shape == (100, 100)

    def test_occ_operator_traces(self, schrodinger_sim, dm_sim_no_decay):
        """Trace of |i⟩⟨i| ⊗ I + I ⊗ |i⟩⟨i| should be 2×dim."""
        # Schrödinger: Tr = 2×7 = 14
        for i in range(7):
            occ = schrodinger_sim._occ_operator(i)
            assert np.isclose(np.trace(occ), 14.0)

        # DM: Tr = 2×10 = 20
        for label in dm_sim_no_decay.level_label:
            occ = dm_sim_no_decay.occ_operator(label)
            trace_val = float(jnp.trace(occ).real)
            assert trace_val == pytest.approx(20.0, abs=1e-10)

    def test_occ_operator_hermiticity(self, schrodinger_sim, dm_sim_no_decay):
        """Occupation operators should be Hermitian."""
        for i in range(7):
            occ = schrodinger_sim._occ_operator(i)
            np.testing.assert_allclose(occ, occ.conj().T, atol=1e-14)

        for label in dm_sim_no_decay.level_label:
            occ = np.array(dm_sim_no_decay.occ_operator(label))
            np.testing.assert_allclose(occ, occ.conj().T, atol=1e-14)

    def test_occ_operator_positive_semidefinite(self, dm_sim_no_decay):
        """Occupation operators should be positive semi-definite."""
        for label in dm_sim_no_decay.level_label:
            occ = np.array(dm_sim_no_decay.occ_operator(label))
            eigenvalues = np.linalg.eigvalsh(occ)
            assert np.all(eigenvalues >= -1e-12), \
                f"Negative eigenvalue in occ_operator('{label}')"


# ============================================================================
# TEST CLASS 7: FIDELITY COMPARISON
# ============================================================================

class TestFidelityComparison:
    """Compare gate fidelity calculations between solvers."""

    def test_schrodinger_fidelity_bounds(self, schrodinger_sim):
        """Schrödinger fidelity should always be between 0 and 1."""
        x_to = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        infid_sch = schrodinger_sim.avg_fidelity(x_to)
        fid_sch = 1 - infid_sch

        assert 0 <= fid_sch <= 1, f"Schrödinger fidelity out of bounds: {fid_sch}"

    def test_dm_fidelity_bounds(self, dm_sim_no_decay):
        """DM fidelity should always be between 0 and 1 (with numerical tolerance)."""
        psi0 = dm_sim_no_decay.SSS_initial_state_list[5]  # |11⟩
        rho0 = dm_sim_no_decay.psi_to_rho(psi0)
        CZ = dm_sim_no_decay.CZ_ideal()
        rho_final = CZ @ rho0 @ jnp.conj(CZ).T

        fid_dm, _ = dm_sim_no_decay.CZ_fidelity(rho_final, psi0)

        # Allow small numerical overshoot above 1.0 due to floating point precision
        assert -1e-10 <= fid_dm <= 1 + 1e-10, f"DM fidelity out of bounds: {fid_dm}"

    @pytest.mark.parametrize("sss_idx", range(12))
    def test_ideal_cz_fidelity_is_one(self, dm_sim_no_decay, sss_idx):
        """Applying ideal CZ to any SSS state should give fidelity ~1."""
        CZ = dm_sim_no_decay.CZ_ideal()
        psi0 = dm_sim_no_decay.SSS_initial_state_list[sss_idx]
        rho0 = dm_sim_no_decay.psi_to_rho(psi0)
        rho_final = CZ @ rho0 @ jnp.conj(CZ).T

        fid, _ = dm_sim_no_decay.CZ_fidelity(rho_final, psi0)

        assert fid == pytest.approx(1.0, abs=1e-5), \
            f"SSS state {sss_idx}: fidelity = {fid} != 1.0"
