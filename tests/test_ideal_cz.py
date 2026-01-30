"""Smoke tests for the CZ gate simulator."""

import numpy as np
import pytest


def test_pulse_optimizer_instantiation():
    """PulseOptimizer should instantiate for all supported pulse types."""
    from ryd_gate.noise import PulseOptimizer

    for pt in ["TO", "AR", "DR", "SSR"]:
        opt = PulseOptimizer(pulse_type=pt)
        assert opt.pulse_type == pt


def test_pulse_optimizer_invalid_type():
    from ryd_gate.noise import PulseOptimizer

    with pytest.raises(ValueError):
        PulseOptimizer(pulse_type="INVALID")


def test_analytical_pulse_shape():
    """Analytical pulse shape should return correct value at t=0."""
    from ryd_gate.noise import PulseOptimizer

    val = PulseOptimizer._analytical_pulse_shape(0, A=1, B=0, w=1, p1=0, p2=0, C=0, D=5)
    assert val == pytest.approx(5.0)
