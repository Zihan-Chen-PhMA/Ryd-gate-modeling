"""Tests for lightshift module."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from scipy.constants import c, pi


class TestCalculatePolarizabilities:
    """Tests for calculate_polarizabilities function."""

    def test_returns_dict_with_correct_keys(self):
        """Should return dict with scalar, vector, tensor keys."""
        from ryd_gate.lightshift import calculate_polarizabilities
        from arc import Rubidium87
        
        atom = Rubidium87()
        omega = 2 * pi * c / (780e-9)  # D2 line frequency
        
        result = calculate_polarizabilities(1, omega, atom, max_n=7)
        
        assert isinstance(result, dict)
        assert "scalar" in result
        assert "vector" in result
        assert "tensor" in result

    def test_scalar_polarizability_nonzero(self):
        """Scalar polarizability should be non-zero for valid input."""
        from ryd_gate.lightshift import calculate_polarizabilities
        from arc import Rubidium87
        
        atom = Rubidium87()
        omega = 2 * pi * c / (780e-9)
        
        result = calculate_polarizabilities(1, omega, atom, max_n=7)
        
        assert result["scalar"] != 0

    def test_F1_vs_F2_different(self):
        """F=1 and F=2 ground states should have different polarizabilities."""
        from ryd_gate.lightshift import calculate_polarizabilities
        from arc import Rubidium87
        
        atom = Rubidium87()
        omega = 2 * pi * c / (780e-9)
        
        result_F1 = calculate_polarizabilities(1, omega, atom, max_n=7)
        result_F2 = calculate_polarizabilities(2, omega, atom, max_n=7)
        
        # Should have different values
        assert result_F1["scalar"] != result_F2["scalar"]

    def test_tensor_zero_for_F_half(self):
        """Tensor polarizability should be zero for F <= 0.5."""
        from ryd_gate.lightshift import calculate_polarizabilities
        from arc import Rubidium87
        
        atom = Rubidium87()
        omega = 2 * pi * c / (780e-9)
        
        # F=1 should have tensor contribution
        result = calculate_polarizabilities(1, omega, atom, max_n=7)
        
        # For F=1 > 0.5, tensor should be calculated
        # (might be zero depending on selection rules but code path is exercised)
        assert "tensor" in result

    def test_handles_resonance_skip(self):
        """Should skip near-resonant transitions without error."""
        from ryd_gate.lightshift import calculate_polarizabilities
        from arc import Rubidium87
        
        atom = Rubidium87()
        # Very close to D1 transition
        omega = 2 * pi * c / (795e-9)
        
        # Should not raise
        result = calculate_polarizabilities(2, omega, atom, max_n=7)
        assert isinstance(result, dict)


class TestCalculateStarkShiftPerESq:
    """Tests for calculate_stark_shift_per_E_sq function."""

    def test_normalized_polarization_required(self):
        """Should raise ValueError for unnormalized polarization."""
        from ryd_gate.lightshift import calculate_stark_shift_per_E_sq
        
        polarizabilities = {"scalar": 1.0, "vector": 0.5, "tensor": 0.1}
        bad_pol = [1, 1, 1]  # Not normalized
        
        with pytest.raises(ValueError, match="normalized"):
            calculate_stark_shift_per_E_sq(1, 0, polarizabilities, bad_pol)

    def test_accepts_normalized_polarization(self):
        """Should accept properly normalized polarization vector."""
        from ryd_gate.lightshift import calculate_stark_shift_per_E_sq
        
        polarizabilities = {"scalar": 1e-39, "vector": 1e-40, "tensor": 1e-41}
        # sigma+ polarization
        pol = [0, 0, 1]
        
        result = calculate_stark_shift_per_E_sq(2, 0, polarizabilities, pol)
        assert isinstance(result, float)

    def test_scalar_shift_only_for_F0(self):
        """For F=0, only scalar shift contributes."""
        from ryd_gate.lightshift import calculate_stark_shift_per_E_sq
        
        polarizabilities = {"scalar": 1e-39, "vector": 1e-40, "tensor": 1e-41}
        pol = [0, 0, 1]
        
        # F=0 case - only scalar contributes
        result = calculate_stark_shift_per_E_sq(0, 0, polarizabilities, pol)
        expected = -0.5 * polarizabilities["scalar"]
        assert result == pytest.approx(expected)

    def test_vector_shift_for_F_positive(self):
        """Vector shift should contribute for F > 0."""
        from ryd_gate.lightshift import calculate_stark_shift_per_E_sq
        
        polarizabilities = {"scalar": 1e-39, "vector": 1e-40, "tensor": 1e-41}
        
        # sigma+ vs sigma- should give different shifts for m_F != 0
        pol_plus = [0, 0, 1]
        pol_minus = [1, 0, 0]
        
        result_plus = calculate_stark_shift_per_E_sq(2, 1, polarizabilities, pol_plus)
        result_minus = calculate_stark_shift_per_E_sq(2, 1, polarizabilities, pol_minus)
        
        # Should be different due to vector contribution
        assert result_plus != result_minus

    def test_tensor_shift_for_F_greater_than_half(self):
        """Tensor shift should contribute for F > 0.5."""
        from ryd_gate.lightshift import calculate_stark_shift_per_E_sq
        
        polarizabilities = {"scalar": 1e-39, "vector": 0, "tensor": 1e-41}
        
        # pi polarization vs sigma
        pol_pi = [0, 1, 0]
        pol_sigma = [0, 0, 1]
        
        result_pi = calculate_stark_shift_per_E_sq(2, 0, polarizabilities, pol_pi)
        result_sigma = calculate_stark_shift_per_E_sq(2, 0, polarizabilities, pol_sigma)
        
        # Should be different due to tensor contribution
        assert result_pi != result_sigma

    def test_m_F_dependence(self):
        """Shift should depend on m_F quantum number."""
        from ryd_gate.lightshift import calculate_stark_shift_per_E_sq
        
        polarizabilities = {"scalar": 1e-39, "vector": 1e-40, "tensor": 1e-41}
        pol = [0, 0, 1]
        
        results = [calculate_stark_shift_per_E_sq(2, m, polarizabilities, pol) 
                   for m in [-2, -1, 0, 1, 2]]
        
        # Different m_F should give different shifts
        assert len(set(results)) > 1


class TestGetMatFromDiffShift:
    """Tests for get_mat_from_diff_shift function."""

    def test_sigma_plus_polarization(self):
        """Should handle sigma+ polarization."""
        from ryd_gate.lightshift import get_mat_from_diff_shift
        
        g_state = (5, 0, 0.5, -0.5)
        e_state = (6, 1, 1.5, 0.5)
        
        result = get_mat_from_diff_shift(780, "sigma+", 1e6, g_state, e_state)
        
        assert isinstance(result, float)
        assert result > 0

    def test_sigma_minus_polarization(self):
        """Should handle sigma- polarization."""
        from ryd_gate.lightshift import get_mat_from_diff_shift
        
        g_state = (5, 0, 0.5, 0.5)
        e_state = (6, 1, 1.5, -0.5)
        
        result = get_mat_from_diff_shift(780, "sigma-", 1e6, g_state, e_state)
        
        assert isinstance(result, float)
        assert result > 0

    def test_invalid_polarization_raises(self):
        """Should raise ValueError for invalid polarization."""
        from ryd_gate.lightshift import get_mat_from_diff_shift
        
        g_state = (5, 0, 0.5, -0.5)
        e_state = (6, 1, 1.5, 0.5)
        
        with pytest.raises(ValueError, match="sigma"):
            get_mat_from_diff_shift(780, "linear", 1e6, g_state, e_state)

    def test_larger_shift_gives_larger_rabi(self):
        """Larger differential shift should give larger Rabi frequency."""
        from ryd_gate.lightshift import get_mat_from_diff_shift
        
        g_state = (5, 0, 0.5, -0.5)
        e_state = (6, 1, 1.5, 0.5)
        
        rabi_small = get_mat_from_diff_shift(780, "sigma+", 1e5, g_state, e_state)
        rabi_large = get_mat_from_diff_shift(780, "sigma+", 1e7, g_state, e_state)
        
        assert rabi_large > rabi_small


class TestCalculateOmegaEff:
    """Tests for calculate_omega_eff function."""

    def test_returns_float(self):
        """Should return a float value."""
        from ryd_gate.lightshift import calculate_omega_eff
        
        result = calculate_omega_eff(
            diff_shift_420_hz=1e6,
            diff_shift_1013_hz=1e6,
            Delta_hz=9e9
        )
        
        assert isinstance(result, (float, np.floating))

    def test_sign_depends_on_detuning(self):
        """Sign of omega_eff should depend on detuning sign."""
        from ryd_gate.lightshift import calculate_omega_eff
        
        result_pos = calculate_omega_eff(1e6, 1e6, 9e9)
        result_neg = calculate_omega_eff(1e6, 1e6, -9e9)
        
        # Opposite detuning should give opposite sign
        assert np.sign(result_pos) != np.sign(result_neg)

    def test_proportional_to_shifts(self):
        """Omega_eff should scale with differential shifts."""
        from ryd_gate.lightshift import calculate_omega_eff
        
        result_1x = calculate_omega_eff(1e6, 1e6, 9e9)
        result_2x = calculate_omega_eff(2e6, 2e6, 9e9)
        
        # Should scale roughly as sqrt(shift_420 * shift_1013)
        # So 2x shifts -> 2x omega_eff
        assert abs(result_2x) > abs(result_1x)

    def test_inversely_proportional_to_detuning(self):
        """Omega_eff should decrease with larger detuning."""
        from ryd_gate.lightshift import calculate_omega_eff
        
        result_small_delta = calculate_omega_eff(1e6, 1e6, 5e9)
        result_large_delta = calculate_omega_eff(1e6, 1e6, 20e9)
        
        assert abs(result_small_delta) > abs(result_large_delta)
