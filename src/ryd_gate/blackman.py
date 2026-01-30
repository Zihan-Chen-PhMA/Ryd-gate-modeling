"""Blackman window pulse shaping utilities for adiabatic turn-on/off."""

import numpy as np


def blackman_window(t, t_rise):
    """Evaluate the Blackman window function.

    Parameters
    ----------
    t : array_like
        Time values.
    t_rise : float
        Rise time of the window.

    Returns
    -------
    numpy.ndarray
        Window amplitude in [0, 1].
    """
    return (0.42 - 0.5 * np.cos(2 * np.pi * t / (2 * t_rise)) +
            0.08 * np.cos(4 * np.pi * t / (2 * t_rise)))


def blackman_pulse(t, t_rise, t_gate):
    """Blackman-windowed flat-top pulse.

    Parameters
    ----------
    t : array_like
        Time values.
    t_rise : float
        Rise/fall time.
    t_gate : float
        Total gate duration (must be >= 2 * t_rise).

    Returns
    -------
    numpy.ndarray
        Pulse envelope.
    """
    if t_gate < 2 * t_rise:
        raise ValueError("t_gate is too small compared to t_rise")
    ret = (blackman_window(t, t_rise) * np.heaviside(t_rise - t, 1) +
           np.heaviside(t - t_rise, 0) * np.heaviside(t_gate - t - t_rise, 0) +
           blackman_window(t_gate - t, t_rise) *
           np.heaviside(t_rise - (t_gate - t), 1))
    return ret


def blackman_pulse_sqrt(t, t_rise, t_gate):
    """Square-root of the Blackman-windowed flat-top pulse.

    Parameters
    ----------
    t : array_like
        Time values.
    t_rise : float
        Rise/fall time.
    t_gate : float
        Total gate duration (must be >= 2 * t_rise).

    Returns
    -------
    numpy.ndarray
        Square-root pulse envelope.
    """
    if t_gate < 2 * t_rise:
        raise ValueError("t_gate is too small compared to t_rise")
    ret = (blackman_window(t, t_rise) * np.heaviside(t_rise - t, 1) +
           np.heaviside(t - t_rise, 0) * np.heaviside(t_gate - t - t_rise, 0) +
           blackman_window(t_gate - t, t_rise) *
           np.heaviside(t_rise - (t_gate - t), 1))
    ret_sqrt = np.sqrt(np.maximum(ret, 0))
    return ret_sqrt
