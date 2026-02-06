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

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
        assert sim.param_set == "our"
        assert sim.strategy == "TO"
        assert sim.ryd_level == 70

    def test_instantiation_our_AR(self):
        """CZGateSimulator should instantiate with 'our' params and AR strategy."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="AR")
        assert sim.param_set == "our"
        assert sim.strategy == "AR"

    def test_instantiation_lukin_TO(self):
        """CZGateSimulator should instantiate with 'lukin' params."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="lukin", strategy="TO")
        assert sim.param_set == "lukin"
        assert sim.ryd_level == 53

    def test_instantiation_lukin_AR(self):
        """CZGateSimulator should instantiate with 'lukin' params and AR strategy."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="lukin", strategy="AR")
        assert sim.param_set == "lukin"
        assert sim.strategy == "AR"

    def test_instantiation_with_decay(self):
        """CZGateSimulator should instantiate with decay enabled."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=True, param_set="our")
        assert sim.param_set == "our"
        # Decay should affect the constant Hamiltonian (adds imaginary parts)
        assert np.any(np.imag(sim.tq_ham_const) != 0)

    def test_instantiation_without_decay(self):
        """CZGateSimulator with decay disabled should have real diagonal."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our")
        # Without decay, diagonal should be purely real
        diagonal = np.diag(sim.tq_ham_const)
        assert np.allclose(np.imag(diagonal), 0)

    def test_invalid_param_set(self):
        """CZGateSimulator should raise ValueError for invalid param_set."""
        from ryd_gate.ideal_cz import CZGateSimulator

        with pytest.raises(ValueError, match="Unknown parameter set"):
            CZGateSimulator(decayflag=False, param_set="invalid")

    def test_invalid_strategy_in_optimize(self):
        """optimize() should raise ValueError for invalid strategy."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
        sim.strategy = "INVALID"  # Force invalid strategy
        with pytest.raises(ValueError, match="Unknown strategy"):
            sim.optimize([0.1, 1.0, 0.0, 0.0, 0.0, 1.0])

    def test_blackman_flag_true(self):
        """CZGateSimulator should respect blackmanflag=True."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, blackmanflag=True)
        assert sim.blackmanflag is True

    def test_blackman_flag_false(self):
        """CZGateSimulator should respect blackmanflag=False."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, blackmanflag=False)
        assert sim.blackmanflag is False


# ==================================================================
# TESTS FOR HAMILTONIAN CONSTRUCTION
# ==================================================================


class TestHamiltonianConstruction:
    """Tests for Hamiltonian matrix construction."""

    def test_ham_const_shape(self):
        """Constant Hamiltonian should have shape (49, 49)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False)
        assert sim.tq_ham_const.shape == (49, 49)

    def test_ham_420_shape(self):
        """420nm Hamiltonian should have shape (49, 49)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False)
        assert sim.tq_ham_420.shape == (49, 49)

    def test_ham_1013_shape(self):
        """1013nm Hamiltonian should have shape (49, 49)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False)
        assert sim.tq_ham_1013.shape == (49, 49)

    def test_ham_const_hermitian_no_decay(self):
        """Constant Hamiltonian should be Hermitian when decayflag=False."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False)
        # Hermitian: H = H†
        assert np.allclose(sim.tq_ham_const, sim.tq_ham_const.conj().T)

    def test_ham_420_structure(self):
        """420nm Hamiltonian should couple ground to intermediate states."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our")
        # Check that |1⟩ → |e⟩ coupling exists (index 1 → 2,3,4 in single atom)
        # In two-atom space, this appears in specific matrix elements
        assert not np.allclose(sim.tq_ham_420, 0)

    def test_occ_operator_shape(self):
        """Occupation operator should have shape (49, 49)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False)
        occ_op = sim._occ_operator(0)
        assert occ_op.shape == (49, 49)

    def test_occ_operator_trace(self):
        """Occupation operator trace should be 2*7=14 (both atoms, 7 levels each)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False)
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
        """avg_fidelity with TO strategy should return a float."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        infid = sim.avg_fidelity(x)
        assert isinstance(infid, (float, np.floating))

    def test_fidelity_AR_returns_float(self):
        """avg_fidelity with AR strategy should return a float."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="AR")
        x = [1.0, 0.1, 0.0, 0.05, 0.0, 0.0, 1.0, 0.0]
        infid = sim.avg_fidelity(x)
        assert isinstance(infid, (float, np.floating))

    def test_fidelity_bounded_TO(self):
        """Infidelity should be between 0 and 1 for TO strategy."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        infid = sim.avg_fidelity(x)
        assert 0 <= infid <= 1

    def test_fidelity_bounded_AR(self):
        """Infidelity should be between 0 and 1 for AR strategy."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="AR")
        x = [1.0, 0.1, 0.0, 0.05, 0.0, 0.0, 1.0, 0.0]
        infid = sim.avg_fidelity(x)
        assert 0 <= infid <= 1

    def test_fidelity_lukin_params(self):
        """Fidelity calculation should work with lukin params."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="lukin", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        infid = sim.avg_fidelity(x)
        assert 0 <= infid <= 1


# ==================================================================
# TESTS FOR STATE EVOLUTION
# ==================================================================


class TestStateEvolution:
    """Tests for quantum state evolution methods."""

    def test_get_gate_result_TO_shape(self):
        """_get_gate_result_TO should return array of shape (49, 1000)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
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
        )
        assert result.shape == (49, 1000)

    def test_get_gate_result_AR_shape(self):
        """_get_gate_result_AR should return array of shape (49, 1000)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="AR")
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
        )
        assert result.shape == (49, 1000)

    def test_state_normalization_preserved(self):
        """State norm should be preserved during evolution (no decay)."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
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

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        result = sim.diagnose_run(x, "11")
        assert len(result) == 3
        assert all(isinstance(arr, np.ndarray) for arr in result)

    def test_diagnose_run_AR_returns_three_arrays(self):
        """diagnose_run with AR should return list of 3 arrays."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="AR")
        x = [1.0, 0.1, 0.0, 0.05, 0.0, 0.0, 1.0, 0.0]
        result = sim.diagnose_run(x, "11")
        assert len(result) == 3
        assert all(isinstance(arr, np.ndarray) for arr in result)

    def test_diagnose_run_array_shapes(self):
        """diagnose_run arrays should have length 1000."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        mid_pop, ryd_pop, ryd_garb_pop = sim.diagnose_run(x, "11")
        assert len(mid_pop) == 1000
        assert len(ryd_pop) == 1000
        assert len(ryd_garb_pop) == 1000

    def test_diagnose_run_populations_positive(self):
        """All population values should be non-negative."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        mid_pop, ryd_pop, ryd_garb_pop = sim.diagnose_run(x, "11")
        assert np.all(mid_pop >= 0)
        assert np.all(ryd_pop >= 0)
        assert np.all(ryd_garb_pop >= 0)

    def test_diagnose_run_invalid_initial_state(self):
        """diagnose_run should raise ValueError for invalid initial state."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        with pytest.raises(ValueError, match="Unsupported initial state"):
            sim.diagnose_run(x, "invalid")

    def test_diagnose_run_all_initial_states(self):
        """diagnose_run should work for all valid initial states."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
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

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
        assert sim.x_initial is None

    def test_setup_protocol_TO_stores_params(self):
        """setup_protocol should store TO parameters as x_initial."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        sim.setup_protocol(x)
        assert sim.x_initial == x

    def test_setup_protocol_AR_stores_params(self):
        """setup_protocol should store AR parameters as x_initial."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="AR")
        x = [1.0, 0.1, 0.0, 0.05, 0.0, 0.0, 1.0, 0.0]
        sim.setup_protocol(x)
        assert sim.x_initial == x

    def test_setup_protocol_TO_wrong_length_raises(self):
        """setup_protocol should raise ValueError for wrong TO parameter count."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
        with pytest.raises(ValueError, match="6 elements"):
            sim.setup_protocol([1, 2, 3])

    def test_setup_protocol_AR_wrong_length_raises(self):
        """setup_protocol should raise ValueError for wrong AR parameter count."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="AR")
        with pytest.raises(ValueError, match="8 elements"):
            sim.setup_protocol([1, 2, 3])

    def test_avg_fidelity_no_params_raises(self):
        """avg_fidelity() with no stored params should raise ValueError."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
        with pytest.raises(ValueError, match="No pulse parameters"):
            sim.avg_fidelity()

    def test_avg_fidelity_uses_stored_params(self):
        """avg_fidelity() should use stored params and match explicit call."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        sim.setup_protocol(x)
        infid_explicit = sim.avg_fidelity(x)
        infid_stored = sim.avg_fidelity()
        assert infid_explicit == infid_stored

    def test_diagnose_run_stored_params(self):
        """diagnose_run with stored params should work via keyword arg."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
        x = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        sim.setup_protocol(x)
        result = sim.diagnose_run(initial_state="11")
        assert len(result) == 3

    def test_diagnose_run_AR_sss_states(self):
        """diagnose_run with AR strategy should support SSS states."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="AR")
        x = [1.0, 0.1, 0.0, 0.05, 0.0, 0.0, 1.0, 0.0]
        for i in range(12):
            result = sim.diagnose_run(x, f"SSS-{i}")
            assert len(result) == 3

    def test_avg_fidelity_explicit_does_not_mutate(self):
        """Passing explicit x to avg_fidelity should not change x_initial."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(decayflag=False, param_set="our", strategy="TO")
        x_stored = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
        x_other = [0.2, 0.5, 0.1, 0.1, 0.1, 0.8]
        sim.setup_protocol(x_stored)
        sim.avg_fidelity(x_other)
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
