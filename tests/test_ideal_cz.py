"""Tests for the CZGateSimulator class in ideal_cz.py."""

import numpy as np
import pytest


# ==================================================================
# TESTS FOR INITIALIZATION
# ==================================================================


class TestCZGateSimulatorInit:
    """Tests for CZGateSimulator instantiation and parameter handling."""

    def test_instantiation_our_TO(self):
        """CZGateSimulator should instantiate with 'our' params and TO strategy."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        assert sim.param_set == "our"
        assert sim.strategy == "TO"
        assert sim.ryd_level == 70

    def test_instantiation_our_AR(self):
        """CZGateSimulator should instantiate with 'our' params and AR strategy."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="AR")
        assert sim.param_set == "our"
        assert sim.strategy == "AR"

    def test_instantiation_lukin_TO(self):
        """CZGateSimulator should instantiate with 'lukin' params."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="lukin", strategy="TO")
        assert sim.param_set == "lukin"
        assert sim.ryd_level == 53

    def test_instantiation_lukin_AR(self):
        """CZGateSimulator should instantiate with 'lukin' params and AR strategy."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="lukin", strategy="AR")
        assert sim.param_set == "lukin"
        assert sim.strategy == "AR"

    def test_instantiation_with_decay(self):
        """CZGateSimulator should instantiate with decay enabled."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(
            param_set="our",
            enable_rydberg_decay=True,
            enable_intermediate_decay=True,
            enable_polarization_leakage=True,
        )
        assert sim.param_set == "our"
        # Decay should affect the constant Hamiltonian (adds imaginary parts)
        assert np.any(np.imag(sim.tq_ham_const) != 0)

    def test_instantiation_without_decay(self):
        """CZGateSimulator with decay disabled should have real diagonal."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our")
        # Without decay, diagonal should be purely real
        diagonal = np.diag(sim.tq_ham_const)
        assert np.allclose(np.imag(diagonal), 0)

    def test_invalid_param_set(self):
        """CZGateSimulator should raise ValueError for invalid param_set."""
        from ryd_gate.ideal_cz import CZGateSimulator

        with pytest.raises(ValueError, match="Unknown parameter set"):
            CZGateSimulator(param_set="invalid")

    def test_invalid_strategy_in_optimize(self):
        """optimize() should raise ValueError for invalid strategy."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        sim.strategy = "INVALID"  # Force invalid strategy
        with pytest.raises(ValueError, match="Unknown strategy"):
            sim.optimize([0.1, 1.0, 0.0, 0.0, 0.0, 1.0])

    def test_blackman_flag_true(self):
        """CZGateSimulator should respect blackmanflag=True."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(blackmanflag=True)
        assert sim.blackmanflag is True

    def test_blackman_flag_false(self):
        """CZGateSimulator should respect blackmanflag=False."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(blackmanflag=False)
        assert sim.blackmanflag is False


# ==================================================================
# TESTS FOR HAMILTONIAN CONSTRUCTION
# ==================================================================


class TestHamiltonianConstruction:
    """Tests for Hamiltonian matrix construction."""

    def test_ham_const_shape(self):
        """Constant Hamiltonian should have shape (49, 49)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator()
        assert sim.tq_ham_const.shape == (49, 49)

    def test_ham_420_shape(self):
        """420nm Hamiltonian should have shape (49, 49)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator()
        assert sim.tq_ham_420.shape == (49, 49)

    def test_ham_1013_shape(self):
        """1013nm Hamiltonian should have shape (49, 49)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator()
        assert sim.tq_ham_1013.shape == (49, 49)

    def test_ham_const_hermitian_no_decay(self):
        """Constant Hamiltonian should be Hermitian when all decay flags are off."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator()
        # Hermitian: H = H†
        assert np.allclose(sim.tq_ham_const, sim.tq_ham_const.conj().T)

    def test_ham_420_structure(self):
        """420nm Hamiltonian should couple ground to intermediate states."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our")
        # Check that |1⟩ → |e⟩ coupling exists (index 1 → 2,3,4 in single atom)
        # In two-atom space, this appears in specific matrix elements
        assert not np.allclose(sim.tq_ham_420, 0)

    def test_occ_operator_shape(self):
        """Occupation operator should have shape (49, 49)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator()
        occ_op = sim._occ_operator(0)
        assert occ_op.shape == (49, 49)

    def test_occ_operator_trace(self):
        """Occupation operator trace should be 2*7=14 (both atoms, 7 levels each)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator()
        occ_op = sim._occ_operator(0)
        # |0⟩ appears in 7 states for atom 1 (0X) and 7 states for atom 2 (X0)
        # But |00⟩ is counted once, so trace = 7 + 7 - 0 = 14? No wait...
        # Actually |i><i| ⊗ I + I ⊗ |i><i| has trace = 7 + 7 = 14
        assert np.isclose(np.trace(occ_op), 14)


# ==================================================================
# TESTS FOR FIDELITY CALCULATION
# ==================================================================


class TestFidelityCalculation:
    """Tests for average fidelity calculation."""

    def test_fidelity_TO_returns_float(self):
        """gate_fidelity with TO strategy should return a float."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        infid = sim.gate_fidelity(x)
        assert isinstance(infid, (float, np.floating))

    def test_fidelity_AR_returns_float(self):
        """gate_fidelity with AR strategy should return a float."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="AR")
        x = [1.0, 0.1, 0.0, 0.05, 0.0, 0.0, 1.0, 0.0]
        infid = sim.gate_fidelity(x)
        assert isinstance(infid, (float, np.floating))

    def test_fidelity_bounded_TO(self):
        """Infidelity should be between 0 and 1 for TO strategy."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        infid = sim.gate_fidelity(x)
        assert 0 <= infid <= 1

    def test_fidelity_bounded_AR(self):
        """Infidelity should be between 0 and 1 for AR strategy."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="AR")
        x = [1.0, 0.1, 0.0, 0.05, 0.0, 0.0, 1.0, 0.0]
        infid = sim.gate_fidelity(x)
        assert 0 <= infid <= 1

    def test_fidelity_lukin_params(self):
        """Fidelity calculation should work with lukin params."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="lukin", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        infid = sim.gate_fidelity(x)
        assert 0 <= infid <= 1


# ==================================================================
# TESTS FOR STATE EVOLUTION
# ==================================================================


class TestStateEvolution:
    """Tests for quantum state evolution methods."""

    def test_get_gate_result_TO_shape(self):
        """_get_gate_result_TO should return array of shape (49, 1000)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        ini_state = np.kron(
            [0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
        )
        result = sim._get_gate_result_TO(
            phase_amp=0.1,
            omega=sim.rabi_eff,
            phase_init=0.0,
            delta=0.0,
            t_gate=sim.time_scale,
            state_mat=ini_state,
            t_eval=np.linspace(0, sim.time_scale, 1000),
        )
        assert result.shape == (49, 1000)

    def test_get_gate_result_AR_shape(self):
        """_get_gate_result_AR should return array of shape (49, 1000)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="AR")
        ini_state = np.kron(
            [0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
        )
        result = sim._get_gate_result_AR(
            omega=sim.rabi_eff,
            phase_amp1=0.1,
            phase_init1=0.0,
            phase_amp2=0.05,
            phase_init2=0.0,
            delta=0.0,
            t_gate=sim.time_scale,
            state_mat=ini_state,
            t_eval=np.linspace(0, sim.time_scale, 1000),
        )
        assert result.shape == (49, 1000)

    def test_state_normalization_preserved(self):
        """State norm should be preserved during evolution (no decay)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        ini_state = np.kron(
            [0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
        )
        result = sim._get_gate_result_TO(
            phase_amp=0.1,
            omega=sim.rabi_eff,
            phase_init=0.0,
            delta=0.0,
            t_gate=sim.time_scale,
            state_mat=ini_state,
            t_eval=np.linspace(0, sim.time_scale, 1000),
        )
        # Check norm at several time points
        for t_idx in [0, 250, 500, 750, 999]:
            norm = np.linalg.norm(result[:, t_idx])
            assert np.isclose(norm, 1.0, rtol=1e-6)


# ==================================================================
# TESTS FOR DIAGNOSTIC METHODS
# ==================================================================


class TestDiagnosticMethods:
    """Tests for diagnostic run methods."""

    def test_diagnose_run_TO_returns_three_arrays(self):
        """diagnose_run with TO should return list of 3 arrays."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        result = sim.diagnose_run(x, "11")
        assert len(result) == 3
        assert all(isinstance(arr, np.ndarray) for arr in result)

    def test_diagnose_run_AR_returns_three_arrays(self):
        """diagnose_run with AR should return list of 3 arrays."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="AR")
        x = [1.0, 0.1, 0.0, 0.05, 0.0, 0.0, 1.0, 0.0]
        result = sim.diagnose_run(x, "11")
        assert len(result) == 3
        assert all(isinstance(arr, np.ndarray) for arr in result)

    def test_diagnose_run_array_shapes(self):
        """diagnose_run arrays should have length 1000."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        mid_pop, ryd_pop, ryd_garb_pop = sim.diagnose_run(x, "11")
        assert len(mid_pop) == 1000
        assert len(ryd_pop) == 1000
        assert len(ryd_garb_pop) == 1000

    def test_diagnose_run_populations_positive(self):
        """All population values should be non-negative."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        mid_pop, ryd_pop, ryd_garb_pop = sim.diagnose_run(x, "11")
        assert np.all(mid_pop >= 0)
        assert np.all(ryd_pop >= 0)
        assert np.all(ryd_garb_pop >= 0)

    def test_diagnose_run_invalid_initial_state(self):
        """diagnose_run should raise ValueError for invalid initial state."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        with pytest.raises(ValueError, match="Unsupported initial state"):
            sim.diagnose_run(x, "invalid")

    def test_diagnose_run_all_initial_states(self):
        """diagnose_run should work for all valid initial states."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        for initial in ["00", "01", "10", "11"]:
            result = sim.diagnose_run(x, initial)
            assert len(result) == 3


# ==================================================================
# TESTS FOR STORED-PARAMETER WORKFLOW
# ==================================================================


class TestStoredParameterWorkflow:
    """Tests for the setup_protocol / stored-parameter API."""

    def test_x_initial_default_none(self):
        """x_initial should be None before setup_protocol is called."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        assert sim.x_initial is None

    def test_setup_protocol_TO_stores_params(self):
        """setup_protocol should store TO parameters as x_initial."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        sim.setup_protocol(x)
        assert sim.x_initial == x

    def test_setup_protocol_AR_stores_params(self):
        """setup_protocol should store AR parameters as x_initial."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="AR")
        x = [1.0, 0.1, 0.0, 0.05, 0.0, 0.0, 1.0, 0.0]
        sim.setup_protocol(x)
        assert sim.x_initial == x

    def test_setup_protocol_TO_wrong_length_raises(self):
        """setup_protocol should raise ValueError for wrong TO parameter count."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        with pytest.raises(ValueError, match="6 elements"):
            sim.setup_protocol([1, 2, 3])

    def test_setup_protocol_AR_wrong_length_raises(self):
        """setup_protocol should raise ValueError for wrong AR parameter count."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="AR")
        with pytest.raises(ValueError, match="8 elements"):
            sim.setup_protocol([1, 2, 3])

    def test_gate_fidelity_no_params_raises(self):
        """gate_fidelity() with no stored params should raise ValueError."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        with pytest.raises(ValueError, match="No pulse parameters"):
            sim.gate_fidelity()

    def test_gate_fidelity_uses_stored_params(self):
        """gate_fidelity() should use stored params and match explicit call."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        sim.setup_protocol(x)
        infid_explicit = sim.gate_fidelity(x)
        infid_stored = sim.gate_fidelity()
        assert infid_explicit == infid_stored

    def test_diagnose_run_stored_params(self):
        """diagnose_run with stored params should work via keyword arg."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        sim.setup_protocol(x)
        result = sim.diagnose_run(initial_state="11")
        assert len(result) == 3

    def test_diagnose_run_AR_sss_states(self):
        """diagnose_run with AR strategy should support SSS states."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="AR")
        x = [1.0, 0.1, 0.0, 0.05, 0.0, 0.0, 1.0, 0.0]
        for i in range(12):
            result = sim.diagnose_run(x, f"SSS-{i}")
            assert len(result) == 3

    def test_gate_fidelity_explicit_does_not_mutate(self):
        """Passing explicit x to gate_fidelity should not change x_initial."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        x_stored = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        x_other = [0.2, 0.5, 0.1, 0.1, 0.1, 0.8]
        sim.setup_protocol(x_stored)
        sim.gate_fidelity(x_other)
        assert sim.x_initial == x_stored


# ==================================================================
# TESTS FOR PULSE OPTIMIZER MODULE (from original test file)
# ==================================================================


def test_pulse_optimizer_instantiation():
    """PulseOptimizer should instantiate for all supported pulse types."""
    from ryd_gate.noise import PulseOptimizer

    for pt in ["TO", "AR", "DR", "SSR"]:
        opt = PulseOptimizer(pulse_type=pt)
        assert opt.pulse_type == pt


def test_pulse_optimizer_invalid_type():
    """PulseOptimizer should raise ValueError for invalid pulse type."""
    from ryd_gate.noise import PulseOptimizer

    with pytest.raises(ValueError):
        PulseOptimizer(pulse_type="INVALID")


def test_analytical_pulse_shape():
    """Analytical pulse shape should return correct value at t=0."""
    from ryd_gate.noise import PulseOptimizer

    val = PulseOptimizer._analytical_pulse_shape(0, A=1, B=0, w=1, p1=0, p2=0, C=0, D=5)
    assert val == pytest.approx(5.0)


# ==================================================================
# TESTS FOR MONTE CARLO SIMULATION
# ==================================================================


class TestMonteCarloSimulation:
    """Tests for quasi-static Monte Carlo simulation capabilities."""

    def test_monte_carlo_result_dataclass(self):
        """MonteCarloResult dataclass should be importable and have correct fields."""
        from ryd_gate.ideal_cz import MonteCarloResult

        result = MonteCarloResult(
            mean_fidelity=0.99,
            std_fidelity=0.01,
            mean_infidelity=0.01,
            std_infidelity=0.01,
            n_shots=100,
            fidelities=np.array([0.99] * 100),
        )
        assert result.mean_fidelity == 0.99
        assert result.n_shots == 100

    def test_constructor_requires_sigma_detuning_when_dephasing_enabled(self):
        """Constructor should raise if enable_rydberg_dephasing=True without sigma_detuning."""
        from ryd_gate.ideal_cz import CZGateSimulator

        with pytest.raises(ValueError, match="sigma_detuning"):
            CZGateSimulator(
                param_set="our", strategy="TO",
                enable_rydberg_dephasing=True,
            )

    def test_constructor_requires_sigma_pos_xyz_when_position_enabled(self):
        """Constructor should raise if enable_position_error=True without sigma_pos_xyz."""
        from ryd_gate.ideal_cz import CZGateSimulator

        with pytest.raises(ValueError, match="sigma_pos_xyz"):
            CZGateSimulator(
                param_set="our", strategy="TO",
                enable_position_error=True,
            )

    def test_gate_fidelity_returns_tuple_when_mc_enabled(self):
        """gate_fidelity should return (mean, std) tuple when MC flags are on."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(
            param_set="our", strategy="TO",
            enable_rydberg_dephasing=True,
            sigma_detuning=170e3,
            n_mc_shots=3,
            mc_seed=42,
        )
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        result = sim.gate_fidelity(x)
        assert isinstance(result, tuple)
        assert len(result) == 2
        mean_inf, std_inf = result
        assert isinstance(mean_inf, float)
        assert isinstance(std_inf, float)
        assert mean_inf >= 0
        assert std_inf >= 0

    def test_3d_position_model_distances(self):
        """MC with 3D position model should produce positive distances."""
        from ryd_gate.ideal_cz import CZGateSimulator

        # Use small sigma to keep distances near nominal (3.0 μm)
        small_sigma = (0.05e-6, 0.05e-6, 0.05e-6)  # 50 nm in meters
        sim = CZGateSimulator(
            param_set="our", strategy="TO",
            enable_position_error=True,
            sigma_pos_xyz=small_sigma,
        )
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        result = sim.run_monte_carlo_simulation(
            x, n_shots=10,
            sigma_pos_xyz=small_sigma,
            seed=42,
        )
        assert result.distance_samples is not None
        # All distances should be positive and near nominal (3.0 μm)
        assert np.all(result.distance_samples > 0)
        assert np.all(result.distance_samples > 1.0)  # not unreasonably small
        assert np.all(result.distance_samples < 10.0)  # not unreasonably large

    def test_build_vdw_unit_operator(self):
        """_build_vdw_unit_operator should return correct shape and structure."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        op = sim._build_vdw_unit_operator()

        assert op.shape == (49, 49)
        # Should be non-negative on diagonal
        assert np.all(np.diag(op) >= 0)



# ==================================================================
# TESTS FOR JAX-ACCELERATED MONTE CARLO
# ==================================================================


class TestMonteCarloJax:
    """Tests for GPU-accelerated JAX Monte Carlo simulation."""

    def test_jax_mc_not_implemented_ar(self):
        """JAX MC should raise NotImplementedError for AR strategy."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="AR")
        x = [1.0, 0.5, 0.0, 0.3, 0.0, 0.0, 1.0, 0.0]
        with pytest.raises(NotImplementedError):
            sim.run_monte_carlo_jax(x, n_shots=5, seed=0)


# ==================================================================
# TESTS FOR INDEPENDENT ERROR SOURCE FLAGS
# ==================================================================


class TestIndependentErrorFlags:
    """Tests for the independent enable_* error source flags."""

    def test_enable_rydberg_decay_only(self):
        """Rydberg diagonal should have imaginary part, mid-state should not."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", enable_rydberg_decay=True)
        diag = np.diag(sim.tq_ham_const)
        # Single-atom index 5 → two-atom indices 5*7+j and i*7+5
        # Rydberg states should have imaginary part
        ryd_idx_5 = 5 * 7 + 0  # |r,0⟩
        ryd_idx_5b = 0 * 7 + 5  # |0,r⟩
        assert np.imag(diag[ryd_idx_5]) != 0
        assert np.imag(diag[ryd_idx_5b]) != 0
        # Intermediate states should NOT have imaginary part
        mid_idx = 2 * 7 + 0  # |e1,0⟩
        assert np.imag(diag[mid_idx]) == 0

    def test_enable_intermediate_decay_only(self):
        """Intermediate diagonal should have imaginary part, Rydberg should not."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", enable_intermediate_decay=True)
        diag = np.diag(sim.tq_ham_const)
        # Intermediate states should have imaginary part
        mid_idx = 2 * 7 + 0  # |e1,0⟩
        assert np.imag(diag[mid_idx]) != 0
        # Rydberg states should NOT have imaginary part
        ryd_idx = 5 * 7 + 0  # |r,0⟩
        assert np.imag(diag[ryd_idx]) == 0

    def test_polarization_leakage_disabled(self):
        """With leakage disabled, state 6 should be far-detuned (large Zeeman shift)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", enable_polarization_leakage=False)
        # Zeeman shift should be very large to push |r'⟩ off-resonance
        assert sim.ryd_zeeman_shift > 2 * np.pi * 1e9
        # Hamiltonian should be Hermitian (no decay)
        assert np.allclose(sim.tq_ham_const, sim.tq_ham_const.conj().T)

    def test_polarization_leakage_enabled(self):
        """With leakage enabled, garbage Rabi freqs should be nonzero."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", enable_polarization_leakage=True)
        assert sim.rabi_420_garbage != 0.0
        assert sim.rabi_1013_garbage != 0.0

    def test_zero_state_scattering_flag_off_near_perfect(self):
        """Optimized params (with always-on light shift) should give near-perfect gate."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(
            param_set="our", strategy="TO", blackmanflag=True,
            enable_zero_state_scattering=False,
        )
        x = [-0.9509172186259588, 1.105272315809505, 0.383911389220584,
             1.2848721417313045, 1.3035218398648376, 1.246566016566724]
        infidelity = sim.gate_fidelity(x)
        assert infidelity < 1e-6, f"Infidelity {infidelity} too large with optimized params"

    def test_zero_state_lightshift_matrix(self):
        """Light-shift matrix should always have nonzero real diagonal (always-on)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim_on = CZGateSimulator(param_set="our", enable_zero_state_scattering=True)
        sim_off = CZGateSimulator(param_set="our", enable_zero_state_scattering=False)
        diag_on = np.diag(sim_on.tq_ham_lightshift_zero)
        diag_off = np.diag(sim_off.tq_ham_lightshift_zero)

        # AC Stark shift is always present (real part)
        assert np.any(diag_on.real != 0), "Should have nonzero real light shifts"
        assert np.any(diag_off.real != 0), "Light shift should be present even with flag off"
        # Real parts should match (same AC Stark shift)
        assert np.allclose(diag_on.real, diag_off.real)
        # Flag ON adds imaginary scattering loss on |0⟩; flag OFF is purely real
        assert np.allclose(diag_off.imag, 0), "Flag off should have no imaginary part"
        assert np.any(diag_on.imag != 0), "Flag on should have imaginary scattering loss"
        # |1,1⟩ should remain unshifted (no |1⟩ light shift contribution)
        idx_11 = 1 * 7 + 1
        assert diag_on[idx_11] == 0

    def test_zero_state_lightshift_signs(self):
        """|0⟩ should shift downward, |eᵢ⟩ should shift upward (our params)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our")
        diag = np.diag(sim.tq_ham_lightshift_zero)
        # Extract single-atom shifts from two-atom diagonal entries
        s0 = diag[0 * 7 + 1]  # |0,1⟩ = s0 + s1 (s1=0)
        s2 = diag[2 * 7 + 1]  # |e1,1⟩ = s2
        s3 = diag[3 * 7 + 1]  # |e2,1⟩ = s3
        s4 = diag[4 * 7 + 1]  # |e3,1⟩ = s4

        assert s0.real < 0, "Expected negative |0⟩ light shift"
        assert s2.real > 0, "Expected positive |e1⟩ light shift"
        assert s3.real > 0, "Expected positive |e2⟩ light shift"
        assert s4.real > 0, "Expected positive |e3⟩ light shift"

    def test_zero_state_lightshift_no_coupling_in_h420(self):
        """420nm Hamiltonian should not directly couple |0⟩ to |eᵢ⟩."""
        from ryd_gate.ideal_cz import CZGateSimulator

        for param_set in ("our", "lukin"):
            sim_on = CZGateSimulator(
                param_set=param_set, enable_zero_state_scattering=True,
            )
            sim_off = CZGateSimulator(
                param_set=param_set, enable_zero_state_scattering=False,
            )
            for sim in (sim_on, sim_off):
                ham = sim.tq_ham_420
                for ei in (2, 3, 4):
                    assert ham[ei, 0] == 0, f"|0⟩→|e{ei-1}⟩ coupling should be zero ({param_set})"

    def test_all_flags_off_hermitian_hamiltonian(self):
        """All flags off should produce a purely real-diagonal, Hermitian Hamiltonian."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our")
        # No decay → diagonal should be purely real
        diagonal = np.diag(sim.tq_ham_const)
        assert np.allclose(np.imag(diagonal), 0)
        # Hamiltonian should be Hermitian
        assert np.allclose(sim.tq_ham_const, sim.tq_ham_const.conj().T)

    def test_all_flags_off_perfect_gate(self):
        """All flags off should give near-perfect gate (includes always-on light shift)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(
            param_set="our", strategy="TO", blackmanflag=True,
        )
        # Optimized TO params (re-optimized with always-on |0⟩ light shift)
        x = [-0.9509172186259588, 1.105272315809505, 0.383911389220584,
             1.2848721417313045, 1.3035218398648376, 1.246566016566724]
        infidelity = sim.gate_fidelity(x)
        assert infidelity < 1e-6, f"Infidelity {infidelity} too large for all-flags-off gate"

    def test_all_flags_off_norm_preserved(self):
        """All flags off should preserve state normalization (unitary evolution)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        ini_state = np.kron(
            [0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
        )
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        result = sim._get_gate_result_TO(
            phase_amp=x[0],
            omega=x[1] * sim.rabi_eff,
            phase_init=x[2],
            delta=x[3] * sim.rabi_eff,
            t_gate=x[5] * sim.time_scale,
            state_mat=ini_state,
        )
        # Norm should be exactly 1 (no decay, no leakage loss)
        final_norm = np.linalg.norm(result)
        assert np.isclose(final_norm, 1.0, rtol=1e-6)

    def test_dephasing_flag_gates_mc(self):
        """MC with enable_rydberg_dephasing=False should ignore sigma_detuning."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(
            param_set="our", strategy="TO", enable_rydberg_dephasing=False
        )
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        result = sim.run_monte_carlo_simulation(
            x, n_shots=3, sigma_detuning=170e3, seed=42
        )
        # Dephasing disabled → detuning_samples should be None
        assert result.detuning_samples is None
        # All shots should have identical fidelity (no noise)
        assert result.std_fidelity == pytest.approx(0.0, abs=1e-10)

    def test_position_flag_gates_mc(self):
        """MC with enable_position_error=False should ignore sigma_pos_xyz."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(
            param_set="our", strategy="TO", enable_position_error=False
        )
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        result = sim.run_monte_carlo_simulation(
            x, n_shots=3,
            sigma_pos_xyz=(70e-6, 70e-6, 170e-6),
            seed=42,
        )
        # Position error disabled → distance_samples should be None
        assert result.distance_samples is None
        # All shots should have identical fidelity (no noise)
        assert result.std_fidelity == pytest.approx(0.0, abs=1e-10)


class TestBranchingRatios:
    """Tests for branching ratio calculations and error budget."""

    def test_rydberg_branching_ratios_sum_to_one(self):
        """Rydberg branching ratios should sum to 1."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        br = sim._ryd_branch
        total = br["to_0"] + br["to_1"] + br["to_L0"] + br["to_L1"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_mid_branching_ratios_sum_to_one(self):
        """Mid-state branching ratios should sum to 1 for each F."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        for F in (1, 2, 3):
            br = sim._mid_branch[F]
            total = br["to_0"] + br["to_1"] + br["to_L0"] + br["to_L1"]
            assert total == pytest.approx(1.0, abs=1e-6), f"F={F} ratios sum to {total}"

    def test_rydberg_branching_cross_validation(self):
        """Rydberg branching ratios should match full_error_model.py."""
        from ryd_gate.ideal_cz import CZGateSimulator
        from ryd_gate.full_error_model import jax_atom_Evolution

        sim = CZGateSimulator(param_set="our", strategy="TO")
        model = jax_atom_Evolution(ryd_decay=False, mid_decay=False)

        assert sim._ryd_branch["to_0"] == pytest.approx(model.branch_ratio_0, abs=1e-6)
        assert sim._ryd_branch["to_1"] == pytest.approx(model.branch_ratio_1, abs=1e-6)
        assert sim._ryd_branch["to_L0"] == pytest.approx(model.branch_ratio_L0, abs=1e-6)
        assert sim._ryd_branch["to_L1"] == pytest.approx(model.branch_ratio_L1, abs=1e-6)

    def test_mid_branching_cross_validation(self):
        """Mid-state branching ratios should match full_error_model.py."""
        from ryd_gate.ideal_cz import CZGateSimulator
        from ryd_gate.full_error_model import jax_atom_Evolution

        sim = CZGateSimulator(param_set="our", strategy="TO")
        model = jax_atom_Evolution(ryd_decay=False, mid_decay=False)

        for F, label in [(1, "e1"), (2, "e2"), (3, "e3")]:
            br_sim = sim._mid_branch[F]
            r0, r1, rL0, rL1 = model.mid_branch_ratio(label)
            assert br_sim["to_0"] == pytest.approx(r0, abs=1e-6), f"F={F} to_0 mismatch"
            assert br_sim["to_1"] == pytest.approx(r1, abs=1e-6), f"F={F} to_1 mismatch"
            assert br_sim["to_L0"] == pytest.approx(rL0, abs=1e-6), f"F={F} to_L0 mismatch"
            assert br_sim["to_L1"] == pytest.approx(rL1, abs=1e-6), f"F={F} to_L1 mismatch"

    def test_error_budget_non_negative(self):
        """Error budget values should all be non-negative."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(
            param_set="our", strategy="TO", blackmanflag=True,
            enable_rydberg_decay=True, enable_intermediate_decay=True,
        )
        x = [-0.9509172186259588, 1.105272315809505, 0.383911389220584,
             1.2848721417313045, 1.3035218398648376, 1.246566016566724]
        budget = sim.error_budget(x)

        for source, errors in budget.items():
            for etype, val in errors.items():
                assert val >= 0, f"{source}/{etype} = {val} is negative"

    def test_error_budget_xyz_al_lg_sum(self):
        """XYZ + AL + LG should approximately equal total for each source."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(
            param_set="our", strategy="TO", blackmanflag=True,
            enable_rydberg_decay=True, enable_intermediate_decay=True,
        )
        x = [-0.9509172186259588, 1.105272315809505, 0.383911389220584,
             1.2848721417313045, 1.3035218398648376, 1.246566016566724]
        budget = sim.error_budget(x)

        for source, errors in budget.items():
            component_sum = errors["XYZ"] + errors["AL"] + errors["LG"]
            assert component_sum == pytest.approx(errors["total"], rel=1e-4), \
                f"{source}: XYZ+AL+LG={component_sum} != total={errors['total']}"

    def test_population_evolution_shapes(self):
        """Population evolution should return correct shapes and valid values."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(
            param_set="our", strategy="TO", blackmanflag=True,
            enable_rydberg_decay=True, enable_intermediate_decay=True,
        )
        x = [-0.9509172186259588, 1.105272315809505, 0.383911389220584,
             1.2848721417313045, 1.3035218398648376, 1.246566016566724]
        pops = sim._population_evolution(x, "01")

        assert pops["t_list"].shape == (1000,)
        for key in ["e1", "e2", "e3", "ryd", "ryd_garb"]:
            assert pops[key].shape == (1000,), f"{key} has wrong shape"
            assert np.all(pops[key] >= -1e-10), f"{key} has negative values"
            assert np.all(pops[key] <= 1.0 + 1e-10), f"{key} exceeds 1"
