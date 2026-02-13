"""Tests for Blackman pulse shaping utilities."""

import numpy as np
import pytest

from ryd_gate.blackman import blackman_pulse, blackman_pulse_sqrt, blackman_window


class TestBlackmanWindow:
    def test_endpoints_near_zero(self):
        """Blackman window should be close to zero at t=0."""
        assert blackman_window(0, 1.0) == pytest.approx(0.0, abs=1e-10)

    def test_peak_at_midpoint(self):
        """Blackman window should reach 1.0 at t = t_rise."""
        assert blackman_window(1.0, 1.0) == pytest.approx(1.0, abs=1e-10)

    def test_symmetry(self):
        """Window should be symmetric about t_rise."""
        t_rise = 2.0
        t = np.linspace(0, 2 * t_rise, 201)
        w = blackman_window(t, t_rise)
        np.testing.assert_allclose(w, w[::-1], atol=1e-12)

    def test_range(self):
        """Window values should be in [-0.01, 1.01] (Blackman can go slightly negative)."""
        t = np.linspace(0, 1.0, 1000)
        w = blackman_window(t, 1.0)
        assert np.all(w >= -0.01)
        assert np.all(w <= 1.01)


class TestBlackmanPulse:
    def test_flat_top(self):
        """Pulse should be ~1.0 in the flat-top region."""
        t_rise = 0.1
        t_gate = 1.0
        t_flat = np.linspace(t_rise + 0.01, t_gate - t_rise - 0.01, 100)
        vals = blackman_pulse(t_flat, t_rise, t_gate)
        np.testing.assert_allclose(vals, 1.0, atol=1e-10)

    def test_raises_on_short_gate(self):
        with pytest.raises(ValueError):
            blackman_pulse(0.5, 1.0, 1.0)

    def test_sqrt_nonnegative(self):
        t = np.linspace(0, 1.0, 500)
        vals = blackman_pulse_sqrt(t, 0.1, 1.0)
        assert np.all(vals >= 0)

    def test_sqrt_raises_on_short_gate(self):
        """blackman_pulse_sqrt should raise for t_gate < 2*t_rise."""
        with pytest.raises(ValueError):
            blackman_pulse_sqrt(0.5, 1.0, 1.0)

    def test_pulse_rise_and_fall(self):
        """Pulse should rise and fall with Blackman shape."""
        t_rise = 0.2
        t_gate = 1.0
        t = np.linspace(0, t_gate, 500)
        vals = blackman_pulse(t, t_rise, t_gate)
        
        # Should start near zero
        assert vals[0] == pytest.approx(0.0, abs=0.01)
        # Should end near zero
        assert vals[-1] == pytest.approx(0.0, abs=0.01)
        # Should have max of 1
        assert np.max(vals) == pytest.approx(1.0, abs=0.01)

    def test_sqrt_flat_top(self):
        """Sqrt pulse should be ~1.0 in flat-top region."""
        t_rise = 0.1
        t_gate = 1.0
        t_flat = np.linspace(t_rise + 0.01, t_gate - t_rise - 0.01, 100)
        vals = blackman_pulse_sqrt(t_flat, t_rise, t_gate)
        np.testing.assert_allclose(vals, 1.0, atol=1e-5)
