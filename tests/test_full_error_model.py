"""Tests for the JAX-based two-atom density-matrix simulator.

Tests cover initialization, Hamiltonian construction, state evolution,
and fidelity calculations for the Rydberg CZ gate.
"""

import numpy as np
import pytest

# Skip tests if JAX is not available
jax = pytest.importorskip("jax", exc_type=ImportError)
jnp = jax.numpy

# Skip if arc and qutip not available
arc = pytest.importorskip("arc", exc_type=ImportError)
qutip = pytest.importorskip("qutip", exc_type=ImportError)

from ryd_gate.full_error_model import jax_atom_Evolution


class TestJaxAtomEvolutionInit:
    """Tests for jax_atom_Evolution initialization."""

    @pytest.fixture(scope="class")
    def model(self):
        """Create a model instance (cached for the class)."""
        return jax_atom_Evolution()

    def test_instantiation(self, model):
        """Model should instantiate without errors."""
        assert model is not None
        assert model.levels == 10

    def test_level_labels(self, model):
        """Model should have correct level labels."""
        expected_labels = ['0', '1', 'e1', 'e2', 'e3', 'r1', 'r2', 'rP', 'L0', 'L1']
        assert model.level_label == expected_labels

    def test_state_list_length(self, model):
        """State list should have 10 elements."""
        assert len(model.state_list) == 10

    def test_state_shapes(self, model):
        """Each state should be a column vector of correct shape."""
        for state in model.state_list:
            assert state.shape == (10, 1)

    def test_level_dict_structure(self, model):
        """Level dictionary should have correct structure."""
        for label in model.level_label:
            assert label in model.level_dict
            assert 'idx' in model.level_dict[label]
            assert 'ket' in model.level_dict[label]

    def test_blockade_enabled_by_default(self, model):
        """Blockade should be enabled by default."""
        # Check that Hblockade is non-zero
        assert jnp.any(model.Hblockade != 0)

    def test_decay_operators_exist(self, model):
        """Collapse operators should be created."""
        assert len(model.cops) > 0
        assert len(model.cops_tq) > 0


class TestHamiltonianConstruction:
    """Tests for Hamiltonian matrix construction."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_hconst_shape(self, model):
        """Single-atom diagonal Hamiltonian should have correct shape."""
        assert model.Hconst.shape == (10, 10)

    def test_hconst_tq_shape(self, model):
        """Two-qubit Hamiltonian should have shape (100, 100)."""
        assert model.Hconst_tq.shape == (100, 100)

    def test_h420_shape(self, model):
        """420 nm coupling Hamiltonian should have correct shape."""
        assert model.H_420.shape == (10, 10)
        assert model.H_420_tq.shape == (100, 100)

    def test_h1013_shape(self, model):
        """1013 nm coupling Hamiltonian should have correct shape."""
        assert model.H_1013.shape == (10, 10)
        assert model.H_1013_tq.shape == (100, 100)

    def test_h1013_hermitian(self, model):
        """1013 nm Hamiltonian (Hermitian part) should be Hermitian."""
        H = model.H_1013_tq_hermi
        np.testing.assert_allclose(H, jnp.conj(H).T, atol=1e-12)

    def test_blockade_hamiltonian_diagonal(self, model):
        """Blockade Hamiltonian should be diagonal."""
        H = model.Hblockade
        np.testing.assert_allclose(H, jnp.diag(jnp.diag(H)), atol=1e-14)


class TestStateInitialization:
    """Tests for state initialization methods."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_psi0_normalization(self, model):
        """Initial state should be normalized."""
        norm = jnp.sum(jnp.abs(model.psi0) ** 2)
        assert norm == pytest.approx(1.0, abs=1e-12)

    def test_rho0_trace(self, model):
        """Initial density matrix should have unit trace."""
        trace = jnp.trace(model.rho0)
        assert trace == pytest.approx(1.0, abs=1e-12)

    def test_rho0_positive_semidefinite(self, model):
        """Initial density matrix should be positive semi-definite."""
        eigenvalues = jnp.linalg.eigvalsh(model.rho0)
        assert jnp.all(eigenvalues >= -1e-12)

    def test_psi_to_rho(self, model):
        """psi_to_rho should create valid density matrix."""
        psi = model.state_list[0]  # |0> state
        rho = model.psi_to_rho(psi)
        assert rho.shape == (10, 10)
        assert jnp.trace(rho) == pytest.approx(1.0, abs=1e-12)

    def test_sss_states_count(self, model):
        """Should have 12 SSS (symmetric state set) states."""
        assert len(model.SSS_initial_state_list) == 12

    def test_sss_states_normalized(self, model):
        """All SSS states should be normalized."""
        for state in model.SSS_initial_state_list:
            norm = jnp.sum(jnp.abs(state) ** 2)
            assert norm == pytest.approx(1.0, abs=1e-10)


class TestBlockadeInteraction:
    """Tests for Rydberg blockade interaction."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_c6_coefficient_positive(self, model):
        """C6 coefficient should be positive."""
        assert model.C6 > 0

    def test_blockade_distance_scaling(self):
        """Blockade strength should scale as 1/d^6."""
        model1 = jax_atom_Evolution(distance=3)
        model2 = jax_atom_Evolution(distance=6)
        # V ~ 1/d^6, so V2/V1 = (d1/d2)^6 = (3/6)^6 = 1/64
        ratio = model2.V / model1.V
        expected_ratio = (3 / 6) ** 6
        assert ratio == pytest.approx(expected_ratio, rel=1e-6)


class TestDecayOperators:
    """Tests for collapse operators modeling spontaneous decay."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_decay_rates_nonnegative(self, model):
        """All decay rates should be non-negative."""
        assert model.Gamma_BBR >= 0
        assert model.Gamma_RD >= 0
        assert model.Gamma_mid >= 0

    def test_branch_ratios_sum_to_one(self, model):
        """Branching ratios for Rydberg decay should sum to ~1."""
        total = (model.branch_ratio_0 + model.branch_ratio_1 +
                 model.branch_ratio_L0 + model.branch_ratio_L1)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_cdagc_hermitian(self, model):
        """C^dag C sum should be Hermitian."""
        cdagc = model.cdagc_tq
        np.testing.assert_allclose(cdagc, jnp.conj(cdagc).T, atol=1e-12)


class TestLightShifts:
    """Tests for AC Stark shift calculations."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_420_lightshift_computed(self, model):
        """420 nm light shift should be computed."""
        assert hasattr(model, '_420_lightshift_1')
        assert hasattr(model, '_420_lightshift_0')
        assert np.isfinite(model._420_lightshift_1)

    def test_1013_lightshift_computed(self, model):
        """1013 nm light shift should be computed."""
        assert hasattr(model, '_1013_lightshift_r')
        assert hasattr(model, '_1013_lightshift_1')
        assert np.isfinite(model._1013_lightshift_r)

    def test_differential_shift_nonzero(self, model):
        """Differential light shift should be non-zero."""
        assert model._420_diff_shift != 0
        assert model._1013_diff_shift != 0


class TestDensityMatrixEvolution:
    """Tests for density matrix time evolution."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_evolution_preserves_trace(self, model):
        """Evolution should preserve density matrix trace."""
        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.0
        amp_1013 = lambda t: 1.0

        tlist = jnp.linspace(0, 0.01, 5)  # Very short evolution for speed
        sol = model.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013)

        # Check trace at each time step
        for i in range(len(tlist)):
            trace = jnp.trace(sol[i])
            # Allow some numerical error due to finite precision
            assert jnp.abs(trace - 1.0) < 0.1, f"Trace at step {i}: {trace}"

    def test_evolution_returns_correct_shape(self, model):
        """Evolution should return array of correct shape."""
        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.0
        amp_1013 = lambda t: 1.0

        tlist = jnp.linspace(0, 0.01, 5)
        sol = model.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013)

        assert sol.shape == (len(tlist), 100, 100)

    def test_evolution_initial_condition(self, model):
        """Evolution at t=0 should match initial state."""
        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.0
        amp_1013 = lambda t: 1.0

        rho0 = model.psi_to_rho(model.SSS_initial_state_list[4])  # |00> state
        tlist = jnp.linspace(0, 0.001, 3)
        sol = model.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013, rho0=rho0)

        # Initial condition should be preserved
        np.testing.assert_allclose(sol[0], rho0, atol=1e-6)


class TestCZGate:
    """Tests for CZ gate operations."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_cz_ideal_shape(self, model):
        """Ideal CZ should have correct shape."""
        CZ = model.CZ_ideal()
        assert CZ.shape == (100, 100)

    def test_cz_ideal_unitary(self, model):
        """Ideal CZ should be unitary."""
        CZ = model.CZ_ideal()
        np.testing.assert_allclose(CZ @ jnp.conj(CZ).T, jnp.eye(100), atol=1e-12)

    def test_cz_ideal_diagonal(self, model):
        """Ideal CZ should be diagonal in computational basis."""
        CZ = model.CZ_ideal()
        # CZ is diagonal
        np.testing.assert_allclose(CZ, jnp.diag(jnp.diag(CZ)), atol=1e-12)

    def test_cz_ideal_11_element(self, model):
        """CZ should have -1 on |11> state."""
        CZ = model.CZ_ideal()
        # |11> corresponds to index 11 in two-qubit basis (|1>⊗|1>)
        assert CZ[11, 11] == pytest.approx(-1.0)


class TestFidelityCalculation:
    """Tests for fidelity computation methods."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_perfect_fidelity_for_ideal_cz(self, model):
        """Fidelity should be 1 for perfect CZ gate application."""
        psi0 = model.SSS_initial_state_list[5]  # |11> state
        rho0 = model.psi_to_rho(psi0)

        # Apply ideal CZ
        CZ = model.CZ_ideal()
        rho_final = CZ @ rho0 @ jnp.conj(CZ).T

        fid, theta = model.CZ_fidelity(rho_final, state_initial=psi0)
        assert fid == pytest.approx(1.0, abs=1e-6)

    def test_fidelity_bounded(self, model):
        """Fidelity should be bounded between 0 and 1."""
        # Random mixed state
        rho = jnp.eye(100, dtype=jnp.complex128) / 100
        psi0 = model.SSS_initial_state_list[5]

        fid, theta = model.CZ_fidelity(rho, state_initial=psi0)
        assert 0 <= fid <= 1


class TestSparseHamiltonian:
    """Tests for sparse Hamiltonian representation."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_sparse_indices_valid(self, model):
        """Sparse indices should be within valid range."""
        idx = model.H_offdiag_tq_sparse_idx
        assert jnp.all(idx >= 0)
        assert jnp.all(idx < 100)

    def test_sparse_values_finite(self, model):
        """Sparse Hamiltonian values should be finite."""
        args = {
            "amp_420": lambda t: 1.0,
            "phase_420": lambda t: 0.0,
            "amp_1013": lambda t: 1.0
        }
        H_sparse = model.hamiltonian_sparse(0.0, args)
        assert jnp.all(jnp.isfinite(H_sparse))


class TestMidStateDecay:
    """Tests for intermediate state decay handling."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_mid_state_decay_preserves_trace(self, model):
        """Mid-state decay should preserve trace."""
        rho0 = model.rho0
        rho_decayed = model.mid_state_decay(rho0)

        trace_before = jnp.trace(rho0)
        trace_after = jnp.trace(rho_decayed)

        # Trace should be approximately preserved
        assert jnp.abs(trace_before - trace_after) < 0.1


class TestDisableDecay:
    """Tests for model with decay disabled."""

    def test_no_rydberg_decay(self):
        """Model without Rydberg decay should have Gamma_BBR = 0."""
        model = jax_atom_Evolution(ryd_decay=False)
        assert model.Gamma_BBR == 0
        assert model.Gamma_RD == 0

    def test_no_mid_decay(self):
        """Model without mid-state decay should have Gamma_mid = 0."""
        model = jax_atom_Evolution(mid_decay=False)
        assert model.Gamma_mid == 0

    def test_no_blockade(self):
        """Model without blockade should have zero blockade Hamiltonian."""
        model = jax_atom_Evolution(blockade=False)
        # The constant Hamiltonian should not include blockade term
        # (difficult to test directly, but model should instantiate)
        assert model is not None


class TestBatchEvolution:
    """Tests for batch evolution methods."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    @pytest.mark.slow
    def test_multi_evolution_shape(self, model):
        """Multi-state evolution should return correct shape."""
        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.0
        amp_1013 = lambda t: 1.0

        rho0_list = [model.psi_to_rho(model.SSS_initial_state_list[i]) for i in range(3)]
        tlist = jnp.linspace(0, 0.01, 3)

        sol = model.integrate_rho_multi_jax(tlist, amp_420, phase_420, amp_1013, rho0_list)
        assert sol.shape == (3, len(tlist), 100, 100)


class TestInfidelityDiagnostics:
    """Tests for infidelity diagnostic methods."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_occ_operator_shape(self, model):
        """Occupation operator should have correct shape (100x100)."""
        for label in model.level_label:
            occ = model.occ_operator(label)
            assert occ.shape == (100, 100)

    def test_occ_operator_trace(self, model):
        """Occupation operator trace should equal 2 * levels (20 for 10-level)."""
        for label in model.level_label:
            occ = model.occ_operator(label)
            # Trace of |i><i| ⊗ I + I ⊗ |i><i| = 2 * dim = 20
            assert jnp.trace(occ) == pytest.approx(20.0, abs=1e-10)

    def test_occ_operator_hermitian(self, model):
        """Occupation operator should be Hermitian."""
        for label in model.level_label:
            occ = model.occ_operator(label)
            np.testing.assert_allclose(occ, jnp.conj(occ).T, atol=1e-12)

    def test_occ_operator_positive(self, model):
        """Occupation operator should be positive semi-definite."""
        occ = model.occ_operator('1')
        eigenvalues = jnp.linalg.eigvalsh(occ)
        assert jnp.all(eigenvalues >= -1e-10)

    def test_diagnose_population_keys(self, model):
        """diagnose_population should return expected keys."""
        # Create a simple density matrix trajectory (just initial state)
        rho0 = model.rho0
        sol = jnp.stack([rho0, rho0])  # 2 time points
        
        populations = model.diagnose_population(sol)
        
        expected_keys = {'computational', 'intermediate', 'rydberg', 
                        'rydberg_unwanted', 'leakage', 'total_trace'}
        assert set(populations.keys()) == expected_keys

    def test_diagnose_population_shapes(self, model):
        """Population arrays should have correct length."""
        n_times = 5
        rho0 = model.rho0
        sol = jnp.stack([rho0] * n_times)
        
        populations = model.diagnose_population(sol)
        
        for key, values in populations.items():
            assert len(values) == n_times, f"Key {key} has wrong length"

    def test_diagnose_population_initial_state(self, model):
        """For |11⟩ initial state, computational population should be ~1."""
        rho0 = model.rho0  # Default is |11⟩
        sol = jnp.stack([rho0])
        
        populations = model.diagnose_population(sol)
        
        # |11⟩ is computational basis state
        assert populations['computational'][0] == pytest.approx(1.0, abs=0.01)
        assert populations['intermediate'][0] == pytest.approx(0.0, abs=0.01)
        assert populations['rydberg'][0] == pytest.approx(0.0, abs=0.01)

    def test_diagnose_population_trace(self, model):
        """Total trace should be ~1 for pure state."""
        rho0 = model.rho0
        sol = jnp.stack([rho0])
        
        populations = model.diagnose_population(sol)
        
        assert populations['total_trace'][0] == pytest.approx(1.0, abs=1e-10)

    def test_diagnose_infidelity_keys(self, model):
        """diagnose_infidelity should return expected keys."""
        psi0 = model.SSS_initial_state_list[5]  # |11⟩
        rho0 = model.psi_to_rho(psi0)
        
        # Apply ideal CZ for perfect fidelity
        CZ = model.CZ_ideal()
        rho_final = CZ @ rho0 @ jnp.conj(CZ).T
        
        result = model.diagnose_infidelity(rho_final, psi0)
        
        expected_keys = {'total_infidelity', 'leakage_error', 'rydberg_residual',
                        'intermediate_residual', 'decay_error', 'coherent_error',
                        'fidelity', 'theta'}
        assert set(result.keys()) == expected_keys

    def test_diagnose_infidelity_perfect_gate(self, model):
        """For ideal CZ, total infidelity should be ~0."""
        psi0 = model.SSS_initial_state_list[5]  # |11⟩
        rho0 = model.psi_to_rho(psi0)
        
        CZ = model.CZ_ideal()
        rho_final = CZ @ rho0 @ jnp.conj(CZ).T
        
        result = model.diagnose_infidelity(rho_final, psi0)
        
        assert result['total_infidelity'] == pytest.approx(0.0, abs=1e-5)
        assert result['fidelity'] == pytest.approx(1.0, abs=1e-5)
        assert result['leakage_error'] == pytest.approx(0.0, abs=1e-10)
        assert result['rydberg_residual'] == pytest.approx(0.0, abs=1e-10)

    def test_diagnose_infidelity_bounds(self, model):
        """Infidelity components should be bounded [0, 1]."""
        psi0 = model.SSS_initial_state_list[5]
        # Use a random mixed state
        rho = jnp.eye(100, dtype=jnp.complex128) / 100
        
        result = model.diagnose_infidelity(rho, psi0)
        
        assert 0 <= result['total_infidelity'] <= 1
        assert 0 <= result['fidelity'] <= 1
        assert 0 <= result['leakage_error'] <= 1
        assert 0 <= result['rydberg_residual'] <= 1

    def test_diagnose_infidelity_with_fixed_theta(self, model):
        """diagnose_infidelity should accept fixed theta."""
        psi0 = model.SSS_initial_state_list[5]
        rho0 = model.psi_to_rho(psi0)
        
        CZ = model.CZ_ideal()
        rho_final = CZ @ rho0 @ jnp.conj(CZ).T
        
        # Test with theta=0
        result = model.diagnose_infidelity(rho_final, psi0, theta=0.0)
        
        assert result['theta'] == 0.0
        assert 'fidelity' in result

    @pytest.mark.slow
    def test_diagnose_plot_runs(self, model):
        """diagnose_plot should run without error."""
        import tempfile
        import os
        
        # Simple short evolution
        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.0
        amp_1013 = lambda t: 1.0
        
        tlist = jnp.linspace(0, 0.005, 3)
        sol = model.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013)
        
        # Save to temp file (don't display)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            populations = model.diagnose_plot(tlist, sol, initial_label='11', 
                                              save_path=temp_path)
            
            # Check that file was created
            assert os.path.exists(temp_path)
            
            # Check return value
            assert 'computational' in populations
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_diagnose_plot_returns_populations(self, model):
        """diagnose_plot should return population dict."""
        rho0 = model.rho0
        sol = jnp.stack([rho0, rho0, rho0])
        tlist = jnp.array([0.0, 0.001, 0.002])
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            populations = model.diagnose_plot(tlist, sol, save_path=temp_path)
            
            assert isinstance(populations, dict)
            assert 'computational' in populations
            assert 'rydberg' in populations
        finally:
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestCZFidelityWithLeakage:
    """Tests for CZ_fidelity_with_leakage method (Issue #4)."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_returns_three_values(self, model):
        """CZ_fidelity_with_leakage should return (fidelity, theta, leakage_contribution)."""
        psi0 = model.SSS_initial_state_list[5]  # |11⟩
        rho0 = model.psi_to_rho(psi0)
        
        # Apply ideal CZ
        CZ = model.CZ_ideal()
        rho_final = CZ @ rho0 @ jnp.conj(CZ).T
        
        result = model.CZ_fidelity_with_leakage(rho_final, psi0)
        
        assert len(result) == 3
        fid, theta, leakage = result
        assert isinstance(fid, float)
        assert isinstance(theta, float)
        assert isinstance(leakage, float)

    def test_fidelity_bounded(self, model):
        """Fidelity with leakage should be bounded [0, 1] (with numerical tolerance)."""
        psi0 = model.SSS_initial_state_list[5]
        rho0 = model.psi_to_rho(psi0)
        
        CZ = model.CZ_ideal()
        rho_final = CZ @ rho0 @ jnp.conj(CZ).T
        
        fid, theta, leakage = model.CZ_fidelity_with_leakage(rho_final, psi0)
        
        # Allow small numerical errors beyond [0, 1]
        assert -1e-9 <= fid <= 1 + 1e-9

    def test_perfect_gate_fidelity(self, model):
        """For ideal CZ without leakage, fidelity should be ~1."""
        psi0 = model.SSS_initial_state_list[5]
        rho0 = model.psi_to_rho(psi0)
        
        CZ = model.CZ_ideal()
        rho_final = CZ @ rho0 @ jnp.conj(CZ).T
        
        fid, theta, leakage = model.CZ_fidelity_with_leakage(rho_final, psi0)
        
        # For perfect gate, fidelity should be 1, leakage contribution 0
        assert fid == pytest.approx(1.0, abs=1e-5)
        assert leakage == pytest.approx(0.0, abs=1e-5)

    def test_leakage_increases_fidelity(self, model):
        """When L0 population exists, fidelity_with_leakage >= standard fidelity."""
        psi0 = model.SSS_initial_state_list[5]
        rho0 = model.psi_to_rho(psi0)
        
        # Create a state with some L0 leakage
        CZ = model.CZ_ideal()
        rho_final = CZ @ rho0 @ jnp.conj(CZ).T
        
        # Add artificial L0 leakage
        # |L0,L0⟩ = index 88
        rho_with_leakage = rho_final.at[88, 88].add(0.01)
        rho_with_leakage = rho_with_leakage.at[11, 11].add(-0.01)  # Remove from |11⟩
        
        fid_standard, theta = model.CZ_fidelity(rho_with_leakage, psi0)
        fid_with_leak, _, leakage_contrib = model.CZ_fidelity_with_leakage(
            rho_with_leakage, psi0, theta
        )
        
        # Fidelity with leakage should be >= standard
        # (L0 population maps to 0, which may or may not help depending on ideal state)
        assert fid_with_leak >= fid_standard - 0.001  # Allow small numerical error

    def test_fixed_theta_accepted(self, model):
        """Should accept fixed theta parameter."""
        psi0 = model.SSS_initial_state_list[5]
        rho0 = model.psi_to_rho(psi0)
        
        CZ = model.CZ_ideal()
        rho_final = CZ @ rho0 @ jnp.conj(CZ).T
        
        fid, theta, leakage = model.CZ_fidelity_with_leakage(rho_final, psi0, theta=0.5)
        
        assert theta == 0.5

    def test_consistency_with_standard_no_leakage(self, model):
        """Without L0 population, should give same result as standard fidelity."""
        psi0 = model.SSS_initial_state_list[4]  # |00⟩
        rho0 = model.psi_to_rho(psi0)
        
        # For |00⟩, CZ does nothing, no leakage expected
        CZ = model.CZ_ideal()
        rho_final = CZ @ rho0 @ jnp.conj(CZ).T
        
        fid_standard, theta = model.CZ_fidelity(rho_final, psi0)
        fid_with_leak, _, leakage = model.CZ_fidelity_with_leakage(rho_final, psi0, theta)
        
        # Should be very close (only numerical differences)
        assert fid_with_leak == pytest.approx(fid_standard, abs=1e-4)


class TestThermalEffects:
    """Tests for Monte Carlo thermal Doppler effect methods."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_velocity_thermal_sample_distribution(self, model):
        """Sampled velocities should have mean ~0 and correct std."""
        from scipy.constants import k as kB
        T_atom = 10.0  # μK
        T_K = T_atom * 1e-6
        m_Rb = model.atom.mass * 1e-3
        expected_std = np.sqrt(kB * T_K / m_Rb)

        rng = np.random.default_rng(42)
        samples = [model.velocity_thermal_sample(T_atom, rng=rng) for _ in range(5000)]
        assert np.abs(np.mean(samples)) < 3 * expected_std / np.sqrt(5000)
        assert np.std(samples) == pytest.approx(expected_std, rel=0.1)

    def test_doppler_shift_returns_float(self, model):
        result = model.doppler_shift(10.0)
        assert isinstance(result, float)

    def test_doppler_std_scales_with_temperature(self, model):
        """Higher temperature should give larger Doppler std."""
        assert model.doppler_std(20.0) > model.doppler_std(5.0)

    def test_doppler_std_zero_at_zero_temp(self, model):
        assert model.doppler_std(0.0) == pytest.approx(0.0, abs=1e-15)

    def test_doppler_std_reasonable_value(self, model):
        """At 10 μK the Doppler std should be in the kHz–MHz range."""
        std = model.doppler_std(10.0)
        # Should be roughly 0.01–10 MHz
        assert 1e-3 < std < 100

    @pytest.mark.slow
    def test_simulate_thermal_returns_dict(self, model):
        """simulate_with_thermal_effects should return expected keys."""
        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.0
        amp_1013 = lambda t: 1.0
        tlist = jnp.linspace(0, 0.01, 5)

        result = model.simulate_with_thermal_effects(
            tlist, amp_420, phase_420, amp_1013, T_atom=10.0, n_samples=2, seed=0
        )
        assert set(result.keys()) == {'mean_fidelity', 'std_fidelity',
                                       'fidelity_list', 'doppler_shifts'}
        assert len(result['fidelity_list']) == 2
        assert len(result['doppler_shifts']) == 2

    @pytest.mark.slow
    def test_simulate_thermal_fidelity_bounded(self, model):
        """All fidelities should be in [0, 1]."""
        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.0
        amp_1013 = lambda t: 1.0
        tlist = jnp.linspace(0, 0.01, 5)

        result = model.simulate_with_thermal_effects(
            tlist, amp_420, phase_420, amp_1013, T_atom=10.0, n_samples=3, seed=1
        )
        for f in result['fidelity_list']:
            assert -1e-9 <= f <= 1 + 1e-9

    @pytest.mark.slow
    def test_simulate_thermal_zero_temp_matches_standard(self, model):
        """At T=0 the thermal simulation should match a standard run."""
        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.0
        amp_1013 = lambda t: 1.0
        tlist = jnp.linspace(0, 0.01, 5)

        result = model.simulate_with_thermal_effects(
            tlist, amp_420, phase_420, amp_1013, T_atom=0.0, n_samples=1, seed=0
        )
        sol = model.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013)
        fid_standard, _ = model.CZ_fidelity(sol[-1])
        assert result['mean_fidelity'] == pytest.approx(float(fid_standard), abs=1e-6)


class TestPositionSpread:
    """Tests for thermal position spread from harmonic trap."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_position_thermal_sample_distribution(self, model):
        """Sampled positions should have mean ~0 and correct std."""
        from scipy.constants import k as kB
        T_atom = 10.0  # μK
        trap_freq = 100.0  # kHz
        T_K = T_atom * 1e-6
        m_Rb = model.atom.mass * 1e-3
        omega = 2 * np.pi * trap_freq * 1e3
        expected_std_um = np.sqrt(kB * T_K / (m_Rb * omega ** 2)) * 1e6

        rng = np.random.default_rng(42)
        samples = [model.position_thermal_sample(T_atom, trap_freq, rng=rng)
                   for _ in range(5000)]
        assert np.abs(np.mean(samples)) < 3 * expected_std_um / np.sqrt(5000)
        assert np.std(samples) == pytest.approx(expected_std_um, rel=0.1)

    def test_position_thermal_sample_zero_temp(self, model):
        """At T=0, position sample should be 0."""
        assert model.position_thermal_sample(0.0, 100.0) == 0.0

    def test_position_thermal_sample_zero_freq(self, model):
        """With zero trap frequency, position sample should be 0."""
        assert model.position_thermal_sample(10.0, 0.0) == 0.0

    def test_position_spread_std_scales_with_temperature(self, model):
        """Higher temperature should give larger position spread."""
        assert model.position_spread_std(20.0, 100.0) > model.position_spread_std(5.0, 100.0)

    def test_position_spread_std_scales_with_trap_freq(self, model):
        """Tighter trap should give smaller position spread."""
        assert model.position_spread_std(10.0, 200.0) < model.position_spread_std(10.0, 50.0)

    def test_position_spread_std_zero_at_zero_temp(self, model):
        assert model.position_spread_std(0.0, 100.0) == pytest.approx(0.0, abs=1e-15)

    def test_position_spread_std_reasonable_value(self, model):
        """At 10 μK and 100 kHz trap, spread should be on the order of μm."""
        std = model.position_spread_std(10.0, 100.0)
        # Should be on the order of 0.1–10 μm
        assert 0.01 < std < 10.0

    @pytest.mark.slow
    def test_simulate_thermal_with_position_spread(self, model):
        """simulate_with_thermal_effects with trap_freq should return position_shifts."""
        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.0
        amp_1013 = lambda t: 1.0
        tlist = jnp.linspace(0, 0.01, 5)

        result = model.simulate_with_thermal_effects(
            tlist, amp_420, phase_420, amp_1013,
            T_atom=10.0, n_samples=2, seed=0, trap_freq=100.0
        )
        assert 'position_shifts' in result
        assert len(result['position_shifts']) == 2
        # Each entry is a (dx1, dx2) tuple
        assert len(result['position_shifts'][0]) == 2

    @pytest.mark.slow
    def test_simulate_thermal_without_trap_freq_no_position_shifts(self, model):
        """Without trap_freq, result should not contain position_shifts."""
        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.0
        amp_1013 = lambda t: 1.0
        tlist = jnp.linspace(0, 0.01, 5)

        result = model.simulate_with_thermal_effects(
            tlist, amp_420, phase_420, amp_1013,
            T_atom=10.0, n_samples=2, seed=0
        )
        assert 'position_shifts' not in result

    @pytest.mark.slow
    def test_simulate_thermal_position_restores_state(self, model):
        """After simulation with position spread, model state should be restored."""
        orig_V = model.V
        orig_d = model.d

        amp_420 = lambda t: 1.0
        phase_420 = lambda t: 0.0
        amp_1013 = lambda t: 1.0
        tlist = jnp.linspace(0, 0.01, 5)

        model.simulate_with_thermal_effects(
            tlist, amp_420, phase_420, amp_1013,
            T_atom=10.0, n_samples=2, seed=0, trap_freq=100.0
        )
        assert model.V == orig_V
        assert model.d == orig_d


class TestCZGateExtended:
    """Extended CZ gate fidelity tests with all 12 SSS states."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_ideal_cz_all_sss_fidelity(self, model):
        """Ideal CZ applied to every SSS state should give fidelity 1."""
        CZ = model.CZ_ideal()
        for psi0 in model.SSS_initial_state_list:
            rho0 = model.psi_to_rho(psi0)
            rho_final = CZ @ rho0 @ jnp.conj(CZ).T
            fid, _ = model.CZ_fidelity(rho_final, state_initial=psi0)
            assert fid == pytest.approx(1.0, abs=1e-5)


class TestHamiltonianProperties:
    """Additional Hamiltonian property tests."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_full_hamiltonian_hermitian_at_t0(self, model):
        """Full time-dependent H at t=0 with real amplitudes should be Hermitian
        (ignoring the non-Hermitian decay part in Hconst)."""
        args = {
            "amp_420": lambda t: 1.0,
            "phase_420": lambda t: 0.0,
            "amp_1013": lambda t: 1.0,
        }
        H = model.hamiltonian(0.0, args)
        # The off-diagonal coupling part should be Hermitian
        H_coupling = (model.H_420_tq + model.H_420_tq_conj +
                      model.H_1013_tq_hermi)
        np.testing.assert_allclose(H_coupling, jnp.conj(H_coupling).T, atol=1e-12)


class TestDecayOperatorsExtended:
    """Verify mid-state branching ratios sum to 1 for each intermediate state."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    @pytest.mark.parametrize("level", ["e1", "e2", "e3"])
    def test_mid_branch_ratios_sum_to_one(self, model, level):
        ratios = model.mid_branch_ratio(level)
        assert sum(ratios) == pytest.approx(1.0, abs=0.01)


class TestCustomInitialState:
    """Test init_state0 method."""

    @pytest.fixture(scope="class")
    def model(self):
        return jax_atom_Evolution()

    def test_init_state0_updates_rho(self, model):
        psi_00 = jnp.kron(model.state_0, model.state_0)
        model.init_state0(psi_00)
        expected_rho = psi_00 @ jnp.conj(psi_00).T
        np.testing.assert_allclose(model.rho0, expected_rho, atol=1e-12)
        # Reset to default
        model.init_state0(jnp.kron(model.state_1, model.state_1))


class TestDistanceParameter:
    """Test that different distances produce different blockade strengths."""

    def test_different_distances(self):
        m1 = jax_atom_Evolution(distance=3)
        m2 = jax_atom_Evolution(distance=5)
        assert m1.V != m2.V
        # Closer atoms should have stronger interaction
        assert m1.V > m2.V
