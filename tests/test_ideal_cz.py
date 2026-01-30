"""Tests for ideal_cz module - CZGateSimulator class."""

import numpy as np
import pytest

from ryd_gate.ideal_cz import CZGateSimulator


class TestCZGateSimulatorInit:
    """Tests for CZGateSimulator initialization."""

    def test_our_param_set(self):
        """Should initialize with 'our' parameter set."""
        sim = CZGateSimulator(decayflag=False, param_set='our')
        assert sim.param_set == 'our'
        assert sim.ryd_level == 70

    def test_lukin_param_set(self):
        """Should initialize with 'lukin' parameter set."""
        sim = CZGateSimulator(decayflag=False, param_set='lukin')
        assert sim.param_set == 'lukin'
        assert sim.ryd_level == 53

    def test_invalid_param_set_raises(self):
        """Should raise ValueError for invalid parameter set."""
        with pytest.raises(ValueError, match="Unknown parameter set"):
            CZGateSimulator(decayflag=False, param_set='invalid')

    def test_decay_flag_true(self):
        """Should initialize with decay enabled."""
        sim = CZGateSimulator(decayflag=True)
        assert sim is not None

    def test_decay_flag_false(self):
        """Should initialize with decay disabled."""
        sim = CZGateSimulator(decayflag=False)
        assert sim is not None

    def test_strategy_TO(self):
        """Should accept TO strategy."""
        sim = CZGateSimulator(decayflag=False, strategy='TO')
        assert sim.strategy == 'TO'

    def test_strategy_AR(self):
        """Should accept AR strategy."""
        sim = CZGateSimulator(decayflag=False, strategy='AR')
        assert sim.strategy == 'AR'

    def test_blackman_flag_true(self):
        """Should accept blackman flag True."""
        sim = CZGateSimulator(decayflag=False, blackmanflag=True)
        assert sim.blackmanflag is True

    def test_blackman_flag_false(self):
        """Should accept blackman flag False."""
        sim = CZGateSimulator(decayflag=False, blackmanflag=False)
        assert sim.blackmanflag is False


class TestCZGateSimulatorAttributes:
    """Tests for simulator attributes."""

    @pytest.fixture
    def sim_our(self):
        return CZGateSimulator(decayflag=False, param_set='our')

    @pytest.fixture
    def sim_lukin(self):
        return CZGateSimulator(decayflag=False, param_set='lukin')

    def test_rabi_eff_positive(self, sim_our):
        """Effective Rabi frequency should be positive."""
        assert sim_our.rabi_eff > 0

    def test_time_scale_defined(self, sim_our):
        """Time scale should be defined."""
        assert sim_our.time_scale > 0

    def test_hamiltonian_matrices_shape(self, sim_our):
        """Hamiltonian matrices should be 49x49."""
        assert sim_our.tq_ham_const.shape == (49, 49)
        assert sim_our.tq_ham_420.shape == (49, 49)
        assert sim_our.tq_ham_1013.shape == (49, 49)

    def test_hamiltonian_conjugates(self, sim_our):
        """Conjugate Hamiltonians should be correct."""
        assert np.allclose(sim_our.tq_ham_420_conj, sim_our.tq_ham_420.conj().T)
        assert np.allclose(sim_our.tq_ham_1013_conj, sim_our.tq_ham_1013.conj().T)

    def test_decay_rates_positive(self, sim_our):
        """Decay rates should be positive."""
        assert sim_our.mid_state_decay_rate > 0
        assert sim_our.ryd_state_decay_rate > 0

    def test_lukin_parameters_different(self, sim_our, sim_lukin):
        """Lukin parameters should differ from our parameters."""
        assert sim_our.ryd_level != sim_lukin.ryd_level
        assert sim_our.Delta != sim_lukin.Delta


class TestOccOperator:
    """Tests for occupation operator."""

    @pytest.fixture
    def sim(self):
        return CZGateSimulator(decayflag=False)

    def test_occ_operator_shape(self, sim):
        """Occupation operator should be 49x49."""
        op = sim._occ_operator(0)
        assert op.shape == (49, 49)

    def test_occ_operator_hermitian(self, sim):
        """Occupation operator should be Hermitian."""
        op = sim._occ_operator(3)
        assert np.allclose(op, op.conj().T)

    def test_occ_operator_trace(self, sim):
        """Occupation operator trace should be 2*n_levels (counts both atoms)."""
        op = sim._occ_operator(1)
        # The operator |i><i| ⊗ I + I ⊗ |i><i| has trace = 2 * n_levels = 14
        assert np.trace(op) == pytest.approx(14.0)


class TestDecayIntegrate:
    """Tests for decay integration."""

    @pytest.fixture
    def sim(self):
        return CZGateSimulator(decayflag=True)

    def test_decay_integrate_returns_array(self, sim):
        """Decay integration should return array."""
        t_list = np.linspace(0, 1e-6, 100)
        occ_list = np.exp(-t_list * 1e6)  # Decaying occupation
        
        result = sim._decay_integrate(t_list, occ_list, 1e6)
        
        assert isinstance(result, np.ndarray)

    def test_decay_integrate_increasing(self, sim):
        """Integrated decay should be increasing for positive occupation."""
        t_list = np.linspace(0, 1e-6, 100)
        occ_list = np.ones_like(t_list)  # Constant occupation
        
        result = sim._decay_integrate(t_list, occ_list, 1e6)
        
        # Result should be monotonically increasing
        assert np.all(np.diff(result.flatten()) >= 0)


class TestPublicAPIDispatch:
    """Tests for public API method dispatching."""

    def test_optimize_invalid_strategy_raises(self):
        """optimize should raise for invalid strategy."""
        sim = CZGateSimulator(decayflag=False, strategy='AR')
        sim.strategy = 'INVALID'
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            sim.optimize([0] * 8)

    def test_avg_fidelity_invalid_strategy_raises(self):
        """avg_fidelity should raise for invalid strategy."""
        sim = CZGateSimulator(decayflag=False, strategy='AR')
        sim.strategy = 'INVALID'
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            sim.avg_fidelity([0] * 8)

    def test_diagonise_plot_invalid_strategy_raises(self):
        """diagonise_plot should raise for invalid strategy."""
        sim = CZGateSimulator(decayflag=False, strategy='AR')
        sim.strategy = 'INVALID'
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            sim.diagonise_plot([0] * 8, '11')

    def test_plot_bloch_non_TO_prints_message(self, capsys):
        """plot_bloch should print message for non-TO strategy."""
        sim = CZGateSimulator(decayflag=False, strategy='AR')
        
        sim.plot_bloch([0] * 8, save=False)
        
        captured = capsys.readouterr()
        assert "only implemented for the 'TO' strategy" in captured.out


class TestDiagnoseRunInvalidInitial:
    """Tests for invalid initial state handling."""

    def test_diagonise_run_TO_invalid_initial(self):
        """TO diagnose should raise for invalid initial state."""
        sim = CZGateSimulator(decayflag=False, strategy='TO')
        x = [0.1, 1.0, 0, 0, 0.5, 1.0]
        
        with pytest.raises(ValueError, match="unsupport"):
            sim._diagonise_run_TO(x, 'invalid')

    def test_diagonise_run_AR_invalid_initial(self):
        """AR diagnose should raise for invalid initial state."""
        sim = CZGateSimulator(decayflag=False, strategy='AR')
        x = [1.0, 0.1, 0, 0.05, 0, 0, 1.0, 0.5]
        
        with pytest.raises(ValueError, match="unsupport"):
            sim._diagonise_run_AR(x, 'invalid')


class TestHamiltonianConstruction:
    """Tests for Hamiltonian construction methods."""

    def test_tq_ham_const_hermitian_no_decay(self):
        """Constant Hamiltonian without decay should be Hermitian."""
        sim = CZGateSimulator(decayflag=False)
        H = sim.tq_ham_const
        
        # Without decay, should be Hermitian
        assert np.allclose(H, H.conj().T)

    def test_tq_ham_const_with_decay(self):
        """Constant Hamiltonian with decay should have imaginary diagonal."""
        sim = CZGateSimulator(decayflag=True)
        H = sim.tq_ham_const
        
        # Some diagonal elements should have imaginary parts (decay)
        diag_imag = np.imag(np.diag(H))
        assert np.any(diag_imag < 0)  # Decay terms are negative imaginary

    def test_ham_420_our_vs_lukin(self):
        """420nm Hamiltonians should differ between parameter sets."""
        sim_our = CZGateSimulator(decayflag=False, param_set='our')
        sim_lukin = CZGateSimulator(decayflag=False, param_set='lukin')
        
        assert not np.allclose(sim_our.tq_ham_420, sim_lukin.tq_ham_420)

    def test_ham_1013_our_vs_lukin(self):
        """1013nm Hamiltonians should differ between parameter sets."""
        sim_our = CZGateSimulator(decayflag=False, param_set='our')
        sim_lukin = CZGateSimulator(decayflag=False, param_set='lukin')
        
        assert not np.allclose(sim_our.tq_ham_1013, sim_lukin.tq_ham_1013)


class TestDiagnoseRunValidInitials:
    """Tests for valid initial state handling in diagnose runs."""

    @pytest.fixture
    def sim_TO(self):
        return CZGateSimulator(decayflag=False, strategy='TO', blackmanflag=False)

    @pytest.fixture  
    def sim_AR(self):
        return CZGateSimulator(decayflag=False, strategy='AR', blackmanflag=False)

    def test_diagonise_run_TO_11(self, sim_TO):
        """TO diagnose should run for '11' initial state."""
        x = [0.1, 1.0, 0, 0, 0.5, 0.5]  # Short gate time for speed
        result = sim_TO._diagonise_run_TO(x, '11')
        
        assert len(result) == 3
        assert all(isinstance(arr, np.ndarray) for arr in result)

    def test_diagonise_run_TO_01(self, sim_TO):
        """TO diagnose should run for '01' initial state."""
        x = [0.1, 1.0, 0, 0, 0.5, 0.5]
        result = sim_TO._diagonise_run_TO(x, '01')
        
        assert len(result) == 3

    def test_diagonise_run_TO_10(self, sim_TO):
        """TO diagnose should run for '10' initial state."""
        x = [0.1, 1.0, 0, 0, 0.5, 0.5]
        result = sim_TO._diagonise_run_TO(x, '10')
        
        assert len(result) == 3

    def test_diagonise_run_AR_11(self, sim_AR):
        """AR diagnose should run for '11' initial state."""
        x = [1.0, 0.1, 0, 0.05, 0, 0, 0.5, 0.5]  # Short gate time
        result = sim_AR._diagonise_run_AR(x, '11')
        
        assert len(result) == 3

    def test_diagonise_run_AR_01(self, sim_AR):
        """AR diagnose should run for '01' initial state."""
        x = [1.0, 0.1, 0, 0.05, 0, 0, 0.5, 0.5]
        result = sim_AR._diagonise_run_AR(x, '01')
        
        assert len(result) == 3

    def test_diagonise_run_AR_10(self, sim_AR):
        """AR diagnose should run for '10' initial state."""
        x = [1.0, 0.1, 0, 0.05, 0, 0, 0.5, 0.5]
        result = sim_AR._diagonise_run_AR(x, '10')
        
        assert len(result) == 3
