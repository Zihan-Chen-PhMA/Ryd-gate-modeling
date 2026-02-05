"""Tests for noise module - PulseOptimizer class."""

import numpy as np
import pytest
from ryd_gate.noise import PulseOptimizer


class TestPulseOptimizerInit:
    """Tests for PulseOptimizer initialization."""

    def test_valid_pulse_types(self):
        """Should accept all valid pulse types."""
        for ptype in ["TO", "AR", "DR", "SSR"]:
            opt = PulseOptimizer(pulse_type=ptype)
            assert opt.pulse_type == ptype

    def test_invalid_pulse_type_raises(self):
        """Should raise ValueError for invalid pulse type."""
        with pytest.raises(ValueError, match="Invalid pulse_type"):
            PulseOptimizer(pulse_type="INVALID")

    def test_config_loaded_correctly(self):
        """Config should be loaded based on pulse type."""
        opt = PulseOptimizer(pulse_type="TO")
        assert opt.config["name"] == "Time-Optimal (TO)"
        assert opt.config["gatetime"] == 7.5

    def test_zeta_parameter_for_ssr(self):
        """SSR should store zeta parameter."""
        opt = PulseOptimizer(pulse_type="SSR", zeta=0.2)
        assert opt.config["zeta"] == 0.2

    def test_result_initially_none(self):
        """Results should be None before optimization."""
        opt = PulseOptimizer()
        assert opt.result is None
        assert opt.analytical_result is None


class TestStaticHelpers:
    """Tests for static helper methods."""

    def test_get_H0_01_shape(self):
        """H0_01 should be 2x2 matrix."""
        H = PulseOptimizer._get_H0_01(phase=0)
        assert H.shape == (2, 2)

    def test_get_H0_01_hermitian(self):
        """H0_01 should be Hermitian."""
        H = PulseOptimizer._get_H0_01(phase=0.5)
        assert np.allclose(H, H.conj().T)

    def test_get_H0_01_amplitude_scaling(self):
        """H0_01 should scale with amplitude."""
        H1 = PulseOptimizer._get_H0_01(phase=0, A=1.0)
        H2 = PulseOptimizer._get_H0_01(phase=0, A=2.0)
        assert np.allclose(H2, 2 * H1)

    def test_get_H0_11_shape(self):
        """H0_11 should be 2x2 matrix."""
        H = PulseOptimizer._get_H0_11(phase=0)
        assert H.shape == (2, 2)

    def test_get_H0_11_hermitian(self):
        """H0_11 should be Hermitian."""
        H = PulseOptimizer._get_H0_11(phase=0.5)
        assert np.allclose(H, H.conj().T)

    def test_get_H0_11_sqrt2_larger(self):
        """H0_11 should be sqrt(2) times H0_01."""
        H01 = PulseOptimizer._get_H0_01(phase=0.3)
        H11 = PulseOptimizer._get_H0_11(phase=0.3)
        assert np.allclose(H11, np.sqrt(2) * H01)

    def test_analytical_pulse_shape_basic(self):
        """Analytical pulse shape should return float."""
        result = PulseOptimizer._analytical_pulse_shape(
            t=1.0, A=0.1, B=0.05, w=1.0, p1=0, p2=0, C=0.01, D=0
        )
        assert isinstance(result, (float, np.floating))

    def test_analytical_pulse_shape_array(self):
        """Analytical pulse shape should work with arrays."""
        t = np.linspace(0, 10, 100)
        result = PulseOptimizer._analytical_pulse_shape(
            t=t, A=0.1, B=0.05, w=1.0, p1=0, p2=0, C=0.01, D=0
        )
        assert result.shape == t.shape


class TestCostFunctions:
    """Tests for cost functions."""

    @pytest.fixture
    def optimizer_TO(self):
        return PulseOptimizer(pulse_type="TO")

    @pytest.fixture
    def optimizer_AR(self):
        return PulseOptimizer(pulse_type="AR")

    @pytest.fixture
    def optimizer_DR(self):
        return PulseOptimizer(pulse_type="DR")

    @pytest.fixture
    def optimizer_SSR(self):
        return PulseOptimizer(pulse_type="SSR", zeta=0.1)

    def test_cost_function_TO_returns_scalar(self, optimizer_TO):
        """TO cost function should return scalar."""
        M = 5
        ulist = np.random.rand(M + 1) * 2 * np.pi
        args = {"gatetime": 7.5, "M": M}
        
        cost = optimizer_TO._cost_function_TO(ulist, args)
        
        assert isinstance(cost, (float, np.floating))

    def test_cost_function_TO_bounded(self, optimizer_TO):
        """TO cost should be bounded [0, 1]."""
        M = 5
        ulist = np.random.rand(M + 1) * 2 * np.pi
        args = {"gatetime": 7.5, "M": M}
        
        cost = optimizer_TO._cost_function_TO(ulist, args)
        
        assert 0 <= cost <= 1

    def test_cost_function_AR_returns_scalar(self, optimizer_AR):
        """AR cost function should return scalar."""
        M = 5
        ulist = np.random.rand(M + 1) * 2 * np.pi
        args = {"gatetime": 14.32, "M": M}
        
        cost = optimizer_AR._cost_function_AR(ulist, args)
        
        assert isinstance(cost, (float, np.floating))

    def test_cost_function_AR_nonnegative(self, optimizer_AR):
        """AR cost should be non-negative."""
        M = 5
        ulist = np.random.rand(M + 1) * 2 * np.pi
        args = {"gatetime": 14.32, "M": M}
        
        cost = optimizer_AR._cost_function_AR(ulist, args)
        
        assert cost >= 0

    def test_cost_function_DR_returns_scalar(self, optimizer_DR):
        """DR cost function should return scalar."""
        M = 5
        ulist = np.random.rand(M + 1) * 2 * np.pi
        args = {"gatetime": 8.8, "M": M}
        
        cost = optimizer_DR._cost_function_DR(ulist, args)
        
        assert isinstance(cost, (float, np.floating))

    def test_cost_function_SSR_returns_scalar(self, optimizer_SSR):
        """SSR cost function should return scalar."""
        M = 5
        ulist = np.random.rand(M + 1) * 2 * np.pi
        args = {"gatetime": 14.5, "M": M, "zeta": 0.1}
        
        cost = optimizer_SSR._cost_function_SSR(ulist, args)
        
        assert isinstance(cost, (float, np.floating))

    def test_cost_function_M_zero(self, optimizer_TO):
        """Cost function should handle M=0."""
        ulist = np.array([0.5])  # Just theta
        args = {"gatetime": 7.5, "M": 0}
        
        cost = optimizer_TO._cost_function_TO(ulist, args)
        
        assert isinstance(cost, (float, np.floating))


class TestEvolveStateDiscrete:
    """Tests for discrete state evolution."""

    def test_evolve_returns_two_states(self):
        """Should return two 2-element state vectors."""
        opt = PulseOptimizer()
        M, T = 5, 10.0
        phases = np.zeros(M)
        
        psi0, psi1 = opt._evolve_state_discrete(
            M, T, phases, opt._get_H0_01, opt._get_H0_01
        )
        
        assert psi0.shape == (2,)
        assert psi1.shape == (2,)

    def test_evolve_returns_valid_states(self):
        """Evolution should return valid state vectors."""
        opt = PulseOptimizer()
        M, T = 10, 7.5
        phases = np.random.rand(M) * np.pi
        
        psi0, psi1 = opt._evolve_state_discrete(
            M, T, phases, opt._get_H0_01, opt._get_H0_01
        )
        
        # States should have finite values
        assert np.all(np.isfinite(psi0))
        assert np.all(np.isfinite(psi1))
        # psi0 norm should be <= 1 (unitary evolution of first block)
        assert np.linalg.norm(psi0) <= 1.0 + 1e-10

    def test_evolve_M_zero_returns_initial(self):
        """M=0 should return initial state."""
        opt = PulseOptimizer()
        
        psi0, psi1 = opt._evolve_state_discrete(
            0, 10.0, [], opt._get_H0_01, opt._get_H0_01
        )
        
        assert psi0[0] == pytest.approx(1.0)
        assert psi0[1] == pytest.approx(0.0)


class TestODEEvolution:
    """Tests for ODE-based state evolution."""

    def test_evolve_ode_returns_array(self):
        """ODE evolution should return state array."""
        opt = PulseOptimizer(pulse_type="AR")
        params = (0.1, 0.05, 1.0, 0, 0, 0.01, 0)
        initial = np.array([1, 0, 0, 0], dtype=np.complex128)
        
        result = opt._evolve_state_ode(
            params, t_gate=5.0, 
            H0_func=opt._get_H0_01, 
            H1_func=opt._get_H0_01,
            initial_state=initial
        )
        
        assert result.shape == (4,)

    def test_conticost_function_AR_returns_scalar(self):
        """Continuous AR cost should return scalar."""
        opt = PulseOptimizer(pulse_type="AR")
        params_and_theta = np.array([0.1, 0.05, 1.0, 0, 0, 0.01, 0, 0.5])
        args = {"gatetime": 5.0}
        
        cost = opt._conticost_function_AR(params_and_theta, args)
        
        assert isinstance(cost, (float, np.floating))


class TestMultistepOptimization:
    """Tests for multi-step optimization (quick tests only)."""

    def test_run_with_small_M(self):
        """Should run with small M values for quick test."""
        opt = PulseOptimizer(pulse_type="TO")
        
        # Very small M for fast test
        result = opt.run_multistep_optimization(M_steps=[2, 3])
        
        assert result is not None
        assert "solution_vector" in result
        assert "cost" in result
        assert result["M"] == 3

    def test_result_stored_in_instance(self):
        """Result should be stored in optimizer instance."""
        opt = PulseOptimizer(pulse_type="TO")
        
        opt.run_multistep_optimization(M_steps=[2])
        
        assert opt.result is not None
        assert opt.result["name"] == "Time-Optimal (TO)"


class TestAnalyticalFitAndOptimization:
    """Tests for analytical fit and optimization."""

    def test_raises_without_discrete_result(self):
        """Should raise if discrete optimization not run first."""
        opt = PulseOptimizer(pulse_type="AR")
        
        with pytest.raises(RuntimeError, match="discrete optimization"):
            opt.run_analytical_fit_and_optimization()

    def test_runs_after_discrete(self):
        """Should run after discrete optimization."""
        opt = PulseOptimizer(pulse_type="AR")
        opt.run_multistep_optimization(M_steps=[3])
        
        result = opt.run_analytical_fit_and_optimization()
        
        assert result is not None
        assert "final_params" in result
        assert "theta" in result
        assert "cost" in result


class TestConstants:
    """Tests for class constants."""

    def test_sigma_matrices_shape(self):
        """Sigma matrices should be 2x2."""
        assert PulseOptimizer.SIGMA_PLUS.shape == (2, 2)
        assert PulseOptimizer.SIGMA_MINUS.shape == (2, 2)

    def test_sigma_matrices_relation(self):
        """SIGMA_MINUS should be conjugate transpose of SIGMA_PLUS."""
        assert np.allclose(
            PulseOptimizer.SIGMA_MINUS, 
            PulseOptimizer.SIGMA_PLUS.T.conj()
        )

    def test_rydberg_projector_is_projector(self):
        """Rydberg projector should satisfy P^2 = P."""
        P = PulseOptimizer.RYDBERG_PROJECTOR
        assert np.allclose(P @ P, P)

    def test_rydberg_projector_trace(self):
        """Rydberg projector should have trace 1."""
        assert np.trace(PulseOptimizer.RYDBERG_PROJECTOR) == pytest.approx(1.0)
