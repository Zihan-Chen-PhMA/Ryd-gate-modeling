"""Simplified CZ gate simulator using SciPy ODE solvers with hyperfine structure.

This module provides a Schrödinger equation solver for simulating two-qubit CZ gates
in Rydberg atom systems. Unlike the JAX-based master equation solver in
`full_error_model.py`, this uses SciPy's ODE solvers for pure state evolution
in a 49-dimensional Hilbert space (7 levels × 7 levels).

Use this module for:
- Fast pulse optimization without full decoherence modeling
- Understanding ideal gate dynamics
- Comparing Time-Optimal (TO) vs Amplitude-Robust (AR) pulse strategies

For full density matrix simulations with decay channels, use `full_error_model.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from arc import Rubidium87
from arc.wigner import CG
from qutip import Bloch
from scipy import integrate, interpolate
from scipy.optimize import minimize

from ryd_gate.blackman import blackman_pulse

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Supported initial state labels for diagnostic methods
InitialStateLabel = Literal[
    "00", "01", "10", "11",
    "SSS-0", "SSS-1", "SSS-2", "SSS-3", "SSS-4", "SSS-5",
    "SSS-6", "SSS-7", "SSS-8", "SSS-9", "SSS-10", "SSS-11",
]


# ==================================================================
# MONTE CARLO RESULT DATA CLASS
# ==================================================================


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation.

    Attributes
    ----------
    mean_fidelity : float
        Average gate fidelity over all shots.
    std_fidelity : float
        Standard deviation of gate fidelity.
    mean_infidelity : float
        Average gate infidelity (1 - fidelity) over all shots.
    std_infidelity : float
        Standard deviation of gate infidelity.
    n_shots : int
        Number of Monte Carlo shots performed.
    fidelities : NDArray[np.floating]
        Array of individual fidelities for each shot.
    detuning_samples : NDArray[np.floating] | None
        Sampled detuning errors (rad/s) if T2* dephasing was enabled.
    distance_samples : NDArray[np.floating] | None
        Sampled interatomic distances (μm) if position fluctuations were enabled.
    """

    mean_fidelity: float
    std_fidelity: float
    mean_infidelity: float
    std_infidelity: float
    n_shots: int
    fidelities: "NDArray[np.floating]"
    detuning_samples: "NDArray[np.floating] | None" = None
    distance_samples: "NDArray[np.floating] | None" = None


# ==================================================================
# CZ GATE SIMULATOR CLASS
# ==================================================================


class CZGateSimulator:
    """Simplified CZ gate simulator for Rydberg atoms with hyperfine structure.

    Simulates two-atom dynamics under phase-modulated laser pulses using the
    Schrödinger equation. Each atom has 7 levels forming a 49-dimensional
    two-atom Hilbert space.

    Level Structure (per atom)
    --------------------------
    ::

        Index   Label   Description
        -----   -----   -----------
          0     |0⟩     Ground state (5S1/2, F=1)
          1     |1⟩     Ground state (5S1/2, F=2) - qubit state
          2     |e1⟩    Intermediate (6P3/2, F'=1)
          3     |e2⟩    Intermediate (6P3/2, F'=2)
          4     |e3⟩    Intermediate (6P3/2, F'=3)
          5     |r⟩     Target Rydberg state (nS1/2)
          6     |r'⟩    Unwanted Rydberg state (different mJ)

                         |r⟩ (5)     |r'⟩ (6)
                          ↑  ↑         ↑
                    1013nm│  │         │
                          │  │         │
              ┌───────────┴──┴─────────┴───────────┐
              │   |e1⟩(2)   |e2⟩(3)   |e3⟩(4)      │  6P3/2
              └───────────┬──┬─────────────────────┘
                    420nm │  │
                          ↓  ↓
              ┌───────────────────────────────────┐
              │       |0⟩(0)    |1⟩(1)            │  5S1/2
              └───────────────────────────────────┘

    Parameters
    ----------
    param_set : {'our', 'lukin'}, default='our'
        Physical parameter configuration:
        - 'our': Lab parameters with n=70 Rydberg level, Ω_eff=7 MHz
        - 'lukin': Harvard parameters with n=53 Rydberg level
    strategy : {'TO', 'AR'}, default='AR'
        Pulse optimization strategy:
        - 'TO': Time-Optimal with cosine phase modulation
        - 'AR': Amplitude-Robust with dual sine phase modulation
    blackmanflag : bool, default=True
        Whether to apply Blackman pulse envelope for smooth turn-on/off.
    enable_rydberg_decay : bool, default=False
        Include Rydberg state (|r⟩, |r'⟩) spontaneous decay as imaginary
        energy shifts on states 5, 6.
    enable_intermediate_decay : bool, default=False
        Include intermediate state (|e1⟩, |e2⟩, |e3⟩) spontaneous decay
        as imaginary energy shifts on states 2, 3, 4.
    enable_rydberg_dephasing : bool, default=False
        Gate T2* dephasing noise in Monte Carlo simulations. When False,
        ``T2_star`` parameters in MC methods are ignored.
    enable_position_error : bool, default=False
        Gate position fluctuation noise in Monte Carlo simulations. When
        False, ``temperature``/``sigma_pos`` parameters in MC methods are
        ignored.
    enable_polarization_leakage : bool, default=False
        Include coupling to the unwanted Rydberg state |r'⟩ (state 6)
        via off-polarization Clebsch-Gordan coefficients.

    Attributes
    ----------
    rabi_eff : float
        Effective two-photon Rabi frequency (rad/s).
    time_scale : float
        Natural time scale 2π/Ω_eff (seconds).
    v_ryd : float
        Rydberg-Rydberg interaction strength (rad/s).

    Examples
    --------
    Basic optimization with TO strategy (perfect gate, all error sources off):

    >>> sim = CZGateSimulator(param_set='our', strategy='TO')
    >>> x0 = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]  # Initial pulse parameters
    >>> result = sim.optimize(x0)
    >>> print(f"Optimized infidelity: {result.fun:.6f}")

    Include decay and polarization leakage:

    >>> sim = CZGateSimulator(
    ...     enable_rydberg_decay=True,
    ...     enable_intermediate_decay=True,
    ...     enable_polarization_leakage=True,
    ... )

    Population diagnostics:

    >>> sim.diagnose_plot(result.x, initial_state='11')

    Notes
    -----
    **Time-Optimal (TO) Strategy**

    Phase function: φ(t) = A·cos(ωt + φ₀) + δ·t

    Parameters x = [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale] where:
    - A: Cosine amplitude (radians)
    - ω: Modulation frequency
    - φ₀: Initial phase
    - δ: Linear chirp rate
    - θ: Single-qubit Z rotation angle
    - T: Gate time

    **Amplitude-Robust (AR) Strategy**

    Phase function: φ(t) = A₁·sin(ωt + φ₁) + A₂·sin(2ωt + φ₂) + δ·t

    Parameters x = [ω/Ω_eff, A₁, φ₁, A₂, φ₂, δ/Ω_eff, T/T_scale, θ]

    The dual-frequency modulation provides first-order robustness to
    amplitude fluctuations.
    """

    def __init__(
        self,
        param_set: Literal["our", "lukin"] = "our",
        strategy: Literal["TO", "AR"] = "AR",
        blackmanflag: bool = True,
        detuning_sign: Literal[1, -1] = 1,
        *,
        enable_rydberg_decay: bool = False,
        enable_intermediate_decay: bool = False,
        enable_rydberg_dephasing: bool = False,
        enable_position_error: bool = False,
        enable_polarization_leakage: bool = False,
        sigma_detuning: float | None = None,
        sigma_pos_xyz: tuple[float, float, float] | None = None,
        n_mc_shots: int = 100,
        mc_seed: int | None = None,
    ) -> None:
        """Initialize the CZ gate simulator with specified parameters.

        Parameters
        ----------
        param_set : {'our', 'lukin'}
            Parameter set to use.
        strategy : {'TO', 'AR'}
            Pulse optimization strategy.
        blackmanflag : bool
            Whether to use Blackman envelope.
        detuning_sign : {1, -1}
            Sign of intermediate detuning.
            +1 for blue/bright detuning, i.e., delta*Delta
            -1 for red/dark detuning.
        enable_rydberg_decay : bool
            Include Rydberg state decay as imaginary energy shifts.
        enable_intermediate_decay : bool
            Include intermediate state decay as imaginary energy shifts.
        enable_rydberg_dephasing : bool
            Enable Rydberg dephasing noise. When True, ``gate_fidelity()``
            automatically runs Monte Carlo and returns ``(mean, std)``.
        enable_position_error : bool
            Enable position fluctuation noise. When True, ``gate_fidelity()``
            automatically runs Monte Carlo and returns ``(mean, std)``.
        enable_polarization_leakage : bool
            Include coupling to the unwanted Rydberg state |r'⟩.
        sigma_detuning : float or None
            Detuning noise standard deviation in Hz (e.g. 170e3 for 170 kHz).
            Required when ``enable_rydberg_dephasing=True``.
        sigma_pos_xyz : tuple of 3 floats or None
            Position noise standard deviations ``(sigma_x, sigma_y, sigma_z)``
            in meters, where x-axis connects both atoms.
            E.g. ``(70e-6, 70e-6, 170e-6)``.
            Required when ``enable_position_error=True``.
        n_mc_shots : int, default=100
            Number of Monte Carlo shots for automatic MC in ``gate_fidelity()``.
        mc_seed : int or None
            Random seed for Monte Carlo reproducibility.
        """
        self.param_set = param_set
        self.strategy = strategy
        self.blackmanflag = blackmanflag
        self.detuning_sign = detuning_sign
        self.enable_rydberg_decay = enable_rydberg_decay
        self.enable_intermediate_decay = enable_intermediate_decay
        self.enable_rydberg_dephasing = enable_rydberg_dephasing
        self.enable_position_error = enable_position_error
        self.enable_polarization_leakage = enable_polarization_leakage
        self.sigma_detuning = sigma_detuning
        self.sigma_pos_xyz = sigma_pos_xyz
        self.n_mc_shots = n_mc_shots
        self.mc_seed = mc_seed
        self.x_initial: list[float] | None = None

        if self.enable_rydberg_dephasing and self.sigma_detuning is None:
            raise ValueError(
                "sigma_detuning must be provided when enable_rydberg_dephasing=True. "
                "Typical value: 170e3 (Hz)."
            )
        if self.enable_position_error and self.sigma_pos_xyz is None:
            raise ValueError(
                "sigma_pos_xyz must be provided when enable_position_error=True. "
                "Typical value: (70e-6, 70e-6, 170e-6) (meters)."
            )

        if param_set == "our":
            self._init_our_params()
        elif param_set == "lukin":
            self._init_lukin_params()
        else:
            raise ValueError(
                f"Unknown parameter set: '{param_set}'. Choose 'our' or 'lukin'."
            )

    # ==================================================================
    # INITIALIZATION HELPERS
    # ==================================================================

    def _init_our_params(self) -> None:
        """Initialize 'our' lab experimental parameters (n=70 Rydberg)."""
        self.atom = Rubidium87()
        self.temperature = 300 # K
        # Rydberg level and laser parameters
        self.ryd_level = 70
        # Assumes rabi_420 = rabi_1013, effective Rabi = 7 MHz
        # detuning_sign: +1 for blue/bright, -1 for red/dark
        self.Delta = self.detuning_sign * 2 * np.pi * 9.1e9  # Intermediate detuning (rad/s)
        self.rabi_420 = 2*np.pi*(491)*10**(6)
        self.rabi_1013 = 2*np.pi*(185)*10**(6)
        self.rabi_eff = self.rabi_420*self.rabi_1013/(2*abs(self.Delta)) # Effective two-photon Rabi (rad/s)
        self.time_scale = 2 * np.pi / self.rabi_eff

        # Dipole matrix element ratios for off-resonant transitions
        # Since we use sigma- polarization, we count the branch
        # (mJ = -1/2, mI = 1/2) -> (mJ = -3/2, mI = 1/2)
        # (mJ = 1/2, mI = -1/2) -> (mJ = -1/2, mI = -1/2)
        self.d_mid_ratio = self.atom.getDipoleMatrixElement(
            5, 0, 0.5, 0.5, 6, 1, 1.5, -0.5, -1
        ) / self.atom.getDipoleMatrixElement(5, 0, 0.5, -0.5, 6, 1, 1.5, -1.5, -1)
        self.d_ryd_ratio = self.atom.getDipoleMatrixElement(
            6, 1, 1.5, -0.5, self.ryd_level, 0, 0.5, 0.5, 1
        ) / self.atom.getDipoleMatrixElement(
            6, 1, 1.5, -1.5, self.ryd_level, 0, 0.5, -0.5, 1
        )
        self.rabi_420_garbage = self.rabi_420 * self.d_mid_ratio
        self.rabi_1013_garbage = self.rabi_1013 * self.d_ryd_ratio

        # Rydberg interaction and Zeeman shift
        # The real value of C_6 is C_6= h* 1337GHz*um^6 based on https://arxiv.org/pdf/1506.08463
        # In our simulation, we use  rescaled schrodinger equation $i\partial_t \psi = H/\hbar \psi$
        # So we take C_6 = h*1337GHz*um^6/(hbar) = (2 pi)*1337GHz*um^6
        self.v_ryd = 2 * np.pi * 874e9 / 3**6  # Van der Waals at ~3 μm
        self.v_ryd_garb = 2 * np.pi * 874e9 /3**6 # Suppose the garbage state has the identical van der Waals interaction
        self.ryd_zeeman_shift = 2 * np.pi * 56e6 if self.enable_polarization_leakage else  2 * np.pi * 56e9

        # Decay rate parameters
        # 6P3/2 lifetime 120.7 ± 1.2 ns, refer from https://arxiv.org/abs/physics/0409077
        self.mid_state_decay_rate = 1 / (110.7e-9)
        self.mid_garb_decay_rate = 1 / (110.7e-9)
        # refer to the data from https://arxiv.org/abs/0810.0339, Table VII, 70S1/2 @ 300 K
        self.ryd_state_decay_rate = 1 / (151.55e-6)
        # Suppose the garbage state has the identical decay rate
        self.ryd_garb_decay_rate = 1 / (151.55e-6)

        # Build Hamiltonians
        self.tq_ham_const = self._tq_ham_const()
        self.tq_ham_420 = self._tq_ham_420_our()
        self.tq_ham_1013 = self._tq_ham_1013_our()
        self.tq_ham_420_conj = self._tq_ham_420_our().conj().T
        self.tq_ham_1013_conj = self._tq_ham_1013_our().conj().T
        self.t_rise = 20e-9  # Blackman pulse rise time

    def _init_lukin_params(self) -> None:
        """Initialize 'lukin' (Harvard) experimental parameters (n=53 Rydberg)."""
        self.atom = Rubidium87()

        # Rydberg level and laser parameters
        self.ryd_level = 53
        # detuning_sign: +1 for blue/bright, -1 for red/dark
        self.Delta = self.detuning_sign * 2 * np.pi * 7.8e9  # Intermediate detuning (rad/s)
        self.rabi_420 = 2 * np.pi * 237e6
        self.rabi_1013 = 2 * np.pi * 303e6
        self.rabi_eff = self.rabi_420 * self.rabi_1013 / (2 * abs(self.Delta))
        self.time_scale = 2 * np.pi / self.rabi_eff

        # Dipole matrix element ratios (different polarization than 'our')
        self.d_mid_ratio = self.atom.getDipoleMatrixElement(
            5, 0, 0.5, -0.5, 6, 1, 1.5, 0.5, 1
        ) / self.atom.getDipoleMatrixElement(5, 0, 0.5, 0.5, 6, 1, 1.5, 1.5, 1)
        self.d_ryd_ratio = self.atom.getDipoleMatrixElement(
            6, 1, 1.5, 0.5, self.ryd_level, 0, 0.5, -0.5, -1
        ) / self.atom.getDipoleMatrixElement(
            6, 1, 1.5, 1.5, self.ryd_level, 0, 0.5, 0.5, -1
        )
        self.rabi_420_garbage = self.rabi_420 * self.d_mid_ratio
        self.rabi_1013_garbage = self.rabi_1013 * self.d_ryd_ratio

        # Rydberg interaction and Zeeman shift
        self.v_ryd = 2 * np.pi * 450e6
        self.v_ryd_garb = 2 * np.pi * 450e6
        self.ryd_zeeman_shift = 2 * np.pi * 2.4e9 if self.enable_polarization_leakage else 2 * np.pi * 2.4e12

        # Decay rate parameters
        self.mid_state_decay_rate = 1 / (110e-9)
        self.mid_garb_decay_rate = 1 / (110e-9)
        self.ryd_state_decay_rate = 1 / (88e-6)  # 53S lifetime ~88 μs
        self.ryd_garb_decay_rate = 1 / (88e-6)

        # Build Hamiltonians
        self.tq_ham_const = self._tq_ham_const()
        self.tq_ham_420 = self._tq_ham_420_lukin()
        self.tq_ham_1013 = self._tq_ham_1013_lukin()
        self.tq_ham_420_conj = self._tq_ham_420_lukin().conj().T
        self.tq_ham_1013_conj = self._tq_ham_1013_lukin().conj().T
        self.t_rise = 20e-9  # Blackman pulse rise time

    def _setup_protocol_TO(self, x: list[float]) -> None:
        """Setup the protocol for the TO strategy.

        Parameters
        ----------
        x : list of float
            TO parameters [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale].
        Usage:
        # self.phase_amp = x[0]
        # self.omega = x[1] * self.rabi_eff
        # self.phase_init = x[2]
        # self.delta = x[3] * self.rabi_eff
        # self.theta = x[4]
        # self.t_gate = x[5] * self.time_scale
        """
        if len(x) != 6:
            raise ValueError(
                f"TO parameters must be a list of 6 elements. Got {len(x)} elements."
            )
        self.x_initial = x
        print(f"TO parameters is set to: [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale] = {x}")

    def _setup_protocol_AR(self, x: list[float]) -> None:
        """Setup the protocol for the AR strategy.

        Parameters
        ----------
        x : list of float
            AR parameters [ω/Ω_eff, A₁, φ₁, A₂, φ₂, δ/Ω_eff, T/T_scale, θ].
        """
        if len(x) != 8:
            raise ValueError(
                f"AR parameters must be a list of 8 elements. Got {len(x)} elements."
            )
        self.x_initial = x
        print(f"AR parameters is set to: [ω/Ω_eff, A₁, φ₁, A₂, φ₂, δ/Ω_eff, T/T_scale, θ] = {x}")

    def _setup_protocol(self, x: list[float]) -> None:
        """Dispatch to strategy-specific setup method.

        Parameters
        ----------
        x : list of float
            Pulse parameters (format depends on strategy).
        """
        if self.strategy == "TO":
            self._setup_protocol_TO(x)
        elif self.strategy == "AR":
            self._setup_protocol_AR(x)
        else:
            raise ValueError(
                f"Unknown strategy: '{self.strategy}'. Choose 'TO' or 'AR'."
            )

    def _resolve_params(self, x: list[float] | None, caller: str = "") -> list[float]:
        """Resolve pulse parameters: use explicit x if given, else fall back to stored.

        Parameters
        ----------
        x : list of float or None
            Explicit pulse parameters. If None, uses self.x_initial.
        caller : str
            Name of calling method (for error messages).

        Returns
        -------
        list of float
            Resolved parameters.

        Raises
        ------
        ValueError
            If both x and self.x_initial are None.
        """
        if x is not None:
            return x
        if self.x_initial is not None:
            return self.x_initial
        raise ValueError(
            f"No pulse parameters available{' in ' + caller if caller else ''}. "
            "Call setup_protocol(x) first or pass x explicitly."
        )

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def setup_protocol(self, x: list[float]) -> None:
        """Store pulse parameters for subsequent method calls.

        This enables a workflow where parameters are set once and reused::

            sim.setup_protocol(x0)
            sim.optimize()        # uses stored x0 as initial guess
            sim.gate_fidelity()   # uses stored (optimized) params

        Parameters
        ----------
        x : list of float
            Pulse parameters. Format depends on strategy:
            - TO: [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale] (6 params)
            - AR: [ω/Ω_eff, A₁, φ₁, A₂, φ₂, δ/Ω_eff, T/T_scale, θ] (8 params)
        """
        self._setup_protocol(x)

    def optimize(
        self,
        x_initial=None,
        fid_type: Literal["average", "sss", "bell"] = "average",
    ) -> object:
        """Run pulse parameter optimization for the configured strategy.

        Parameters
        ----------
        x_initial : list of float
            Initial guess for pulse parameters. Format depends on strategy:
            - TO: [A, ω, φ₀, δ, θ, T] (6 parameters)
            - AR: [ω, A₁, φ₁, A₂, φ₂, δ, T, θ] (8 parameters)
        fid_type : {'average', 'sss', 'bell'}, default='average'
            Fidelity metric used as the optimization objective. See
            :meth:`gate_fidelity` for details.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Optimization result containing optimal parameters in `x` attribute
            and final infidelity in `fun` attribute.
        """
        if (self.strategy == "TO"):
            return self._optimization_TO(fid_type, x=x_initial)
        elif self.strategy == "AR":
            return self._optimization_AR(fid_type, x=x_initial)
        else:
            raise ValueError(
                f"Unknown strategy: '{self.strategy}'. Choose 'TO' or 'AR'."
            )

    def _gate_infidelity_single(
        self,
        x: list[float],
        fid_type: Literal["average", "sss", "bell"] = "average",
    ) -> float:
        """Compute single-shot gate infidelity (no MC averaging).

        This is the deterministic computation used per-shot by MC methods.
        """
        if self.strategy == "TO":
            if fid_type == "sss":
                return self.fidelity_sss(x)
            elif fid_type == "bell":
                return self.fidelity_bell(x)
            elif fid_type == "average":
                return self.fidelity_avg(x)
        elif self.strategy == "AR":
            if fid_type == "average":
                return self._avg_fidelity_AR(x)
            else:
                raise ValueError(
                    f"Fidelity type '{fid_type}' is not implemented for AR strategy. "
                    "Choose 'average'."
                )
        raise ValueError(
            f"Unknown fid_type: '{fid_type}'. Choose 'average', 'sss', or 'bell'."
        )

    def gate_fidelity(
        self,
        x: list[float] | None = None,
        fid_type: Literal["average", "sss", "bell"] = "average",
    ) -> "float | tuple[float, float]":
        """Calculate average gate infidelity for given pulse parameters.

        When ``enable_rydberg_dephasing`` or ``enable_position_error`` is True,
        this method automatically runs a Monte Carlo simulation and returns
        ``(mean_infidelity, std_infidelity)``.

        Parameters
        ----------
        x : list of float, optional
            Pulse parameters (format depends on strategy).
            If None, uses parameters stored via setup_protocol().
        fid_type : {'average', 'sss', 'bell'}, default='average'
            Method for computing average gate fidelity:

            - ``'average'`` — Nielsen closed-form formula using only |01⟩ and
              |11⟩ overlaps. 2 ODE solves per call (fastest).
            - ``'sss'`` — Average state infidelity over 12 symmetric
              superposition states. 12 ODE solves per call.
            - ``'bell'`` — Average state infidelity over 4 Bell states
              (|Φ±⟩, |Ψ±⟩). 4 ODE solves per call.

        Returns
        -------
        float or tuple[float, float]
            If no MC flags are enabled: average gate infidelity (1 - F).
            If MC flags are enabled: ``(mean_infidelity, std_infidelity)``.
        """
        x = self._resolve_params(x, "gate_fidelity")
        if self.enable_rydberg_dephasing or self.enable_position_error:
            result = self.run_monte_carlo_simulation(
                x,
                n_shots=self.n_mc_shots,
                sigma_detuning=self.sigma_detuning,
                sigma_pos_xyz=self.sigma_pos_xyz,
                seed=self.mc_seed,
            )
            return (result.mean_infidelity, result.std_infidelity)
        return self._gate_infidelity_single(x, fid_type)

    def fidelity_sss(self, x: list[float]) -> float:
        """Average state infidelity over 12 SSS states."""
        return sum(self.state_infidelity(f"SSS-{i}", x) for i in range(12)) / 12

    def fidelity_bell(self, x: list[float]) -> float:
        """Average state infidelity over 4 Bell states."""
        s0 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=complex)
        s1 = np.array([0, 1, 0, 0, 0, 0, 0], dtype=complex)
        st00, st01 = np.kron(s0, s0), np.kron(s0, s1)
        st10, st11 = np.kron(s1, s0), np.kron(s1, s1)
        bell_states = [
            (st00 + st11) / np.sqrt(2),  # Phi+
            (st00 - st11) / np.sqrt(2),  # Phi-
            (st01 + st10) / np.sqrt(2),  # Psi+
            (st01 - st10) / np.sqrt(2),  # Psi-
        ]
        return sum(self.state_infidelity(b, x) for b in bell_states) / 4

    def diagnose_plot(
        self,
        x: list[float] | None = None,
        initial_state: InitialStateLabel | None = None,
    ) -> None:
        """Generate population evolution plot for diagnostic analysis.

        Plots the time evolution of intermediate state, Rydberg state, and
        unwanted Rydberg state populations during the gate.

        Parameters
        ----------
        x : list of float, optional
            Pulse parameters (format depends on strategy).
            If None, uses parameters stored via setup_protocol().
        initial_state : str
            Two-qubit initial state label. Supports '00', '01', '10', '11'
            and 'SSS-0' through 'SSS-11'.
        """
        if initial_state is None:
            raise ValueError("initial_state is required.")
        x = self._resolve_params(x, "diagnose_plot")
        if self.strategy == "TO":
            return self._diagnose_plot_TO(x, initial_state)
        elif self.strategy == "AR":
            return self._diagnose_plot_AR(x, initial_state)
        else:
            raise ValueError(
                f"Unknown strategy: '{self.strategy}'. Choose 'TO' or 'AR'."
            )

    def diagnose_run(
        self,
        x: list[float] | None = None,
        initial_state: InitialStateLabel | None = None,
    ) -> list[NDArray[np.floating]]:
        """Run diagnostic simulation and return population arrays.

        Parameters
        ----------
        x : list of float, optional
            Pulse parameters (format depends on strategy).
            If None, uses parameters stored via setup_protocol().
        initial_state : str
            Two-qubit initial state label. Supports '00', '01', '10', '11'
            and 'SSS-0' through 'SSS-11'.

        Returns
        -------
        list of ndarray
            [mid_state_pop, ryd_state_pop, ryd_garb_pop] arrays of shape (1000,).
        """
        if initial_state is None:
            raise ValueError("initial_state is required.")
        x = self._resolve_params(x, "diagnose_run")
        if self.strategy == "TO":
            return self._diagnose_run_TO(x, initial_state)
        elif self.strategy == "AR":
            return self._diagnose_run_AR(x, initial_state)
        else:
            raise ValueError(
                f"Unknown strategy: '{self.strategy}'. Choose 'TO' or 'AR'."
            )

    def plot_bloch(self, x: list[float] | None = None, save: bool = True) -> None:
        """Generate Bloch sphere visualization for key transitions.

        Plots trajectories for |01⟩ → |0r⟩ and |11⟩ → |W⟩ transitions on
        separate Bloch spheres. Only implemented for TO strategy.

        Parameters
        ----------
        x : list of float, optional
            TO pulse parameters [A, ω, φ₀, δ, θ, T].
            If None, uses parameters stored via setup_protocol().
        save : bool, default=True
            Whether to save plots as PNG files.
        """
        x = self._resolve_params(x, "plot_bloch")
        if self.strategy == "TO":
            return self._plotBloch_TO(x, save)
        else:
            print("Bloch sphere plot is only implemented for the 'TO' strategy.")
            return

    def state_infidelity(
        self,
        initial_state: InitialStateLabel | NDArray[np.complexfloating],
        x: list[float] | None = None,
    ) -> float:
        """Compute state infidelity for a specific initial state.

        Evolves the given initial state under the pulse protocol and compares
        the result to the ideal CZ gate output (with local Rz corrections).

        The ideal CZ gate with local Rz corrections transforms::

            |00⟩ → |00⟩
            |01⟩ → exp(+iθ) |01⟩
            |10⟩ → exp(+iθ) |10⟩
            |11⟩ → exp(+i(2θ+π)) |11⟩ = −exp(+2iθ) |11⟩

        Parameters
        ----------
        initial_state : str or ndarray
            Either a state label ('00', '01', '10', '11', 'SSS-0' through
            'SSS-11') or a state vector of shape (49,).
        x : list of float, optional
            Pulse parameters (format depends on strategy).
            If None, uses parameters stored via setup_protocol().

        Returns
        -------
        float
            State infidelity (1 - F), where F = |⟨ψ_ideal|ψ_actual⟩|².
        """
        x = self._resolve_params(x, "state_infidelity")

        # Resolve initial state: string label → vector
        if isinstance(initial_state, str):
            sss_states = self._build_sss_state_map()
            if initial_state not in sss_states:
                raise ValueError(f"Unsupported initial state: '{initial_state}'")
            ini_state = sss_states[initial_state]
        else:
            ini_state = np.asarray(initial_state, dtype=complex)

        # Extract theta (single-qubit Z rotation angle)
        if self.strategy == "TO":
            theta = x[4]
        elif self.strategy == "AR":
            theta = x[-1]
        else:
            raise ValueError(
                f"Unknown strategy: '{self.strategy}'. Choose 'TO' or 'AR'."
            )

        # Computational basis states
        s0 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=complex)
        s1 = np.array([0, 1, 0, 0, 0, 0, 0], dtype=complex)
        state_00 = np.kron(s0, s0)
        state_01 = np.kron(s0, s1)
        state_10 = np.kron(s1, s0)
        state_11 = np.kron(s1, s1)

        # Evolve and take final state
        res = self.get_gate_result(ini_state, x)

        # Build ideal final state: (Rz⊗Rz) · CZ · |ψ₀⟩
        c00 = np.vdot(state_00, ini_state)
        c01 = np.vdot(state_01, ini_state)
        c10 = np.vdot(state_10, ini_state)
        c11 = np.vdot(state_11, ini_state)

        psi_ideal = (c00 * state_00 +
                     c01 * np.exp(+1j * theta) * state_01 +
                     c10 * np.exp(+1j * theta) * state_10 +
                     c11 * np.exp(+1j * (2*theta + np.pi)) * state_11)

        fid = np.abs(np.vdot(res, psi_ideal)) ** 2
        return 1.0 - fid

    def get_gate_result(
        self,
        state_mat: NDArray[np.complexfloating],
        x: list[float] | None = None,
        t_eval: NDArray[np.floating] | None = None,
    ) -> NDArray[np.complexfloating]:
        """Evolve a quantum state under the configured pulse protocol.

        Parameters
        ----------
        state_mat : ndarray
            Initial state vector of shape (49,).
        x : list of float, optional
            Pulse parameters (format depends on strategy).
            If None, uses parameters stored via setup_protocol().
        t_eval : ndarray or None, optional
            Times at which to store the solution. If None, only the final
            state is returned as a 1-D array of shape (49,).

        Returns
        -------
        ndarray
            If t_eval is provided: shape (49, len(t_eval)).
            If t_eval is None: shape (49,) — the final state only.
        """
        x = self._resolve_params(x, "get_gate_result")
        if self.strategy == "TO":
            return self._get_gate_result_TO(
                phase_amp=x[0],
                omega=x[1] * self.rabi_eff,
                phase_init=x[2],
                delta=x[3] * self.rabi_eff,
                t_gate=x[5] * self.time_scale,
                state_mat=state_mat,
                t_eval=t_eval,
            )
        elif self.strategy == "AR":
            return self._get_gate_result_AR(
                omega=x[0] * self.rabi_eff,
                phase_amp1=x[1],
                phase_init1=x[2],
                phase_amp2=x[3],
                phase_init2=x[4],
                delta=x[5] * self.rabi_eff,
                t_gate=x[6] * self.time_scale,
                state_mat=state_mat,
                t_eval=t_eval,
            )
        else:
            raise ValueError(
                f"Unknown strategy: '{self.strategy}'. Choose 'TO' or 'AR'."
            )

    # ==================================================================
    # MONTE CARLO SIMULATION
    # ==================================================================

    def run_monte_carlo_simulation(
        self,
        x: list[float],
        n_shots: int = 1000,
        sigma_detuning: float | None = None,
        sigma_pos_xyz: tuple[float, float, float] | None = None,
        seed: int | None = None,
    ) -> MonteCarloResult:
        """Run quasi-static Monte Carlo simulation for error budget analysis.

        Simulates physical error mechanisms via statistical averaging over many
        shots, following the methodology from arXiv:2305.03406 (Lukin group).

        This implements two key error sources:
        1. **Doppler Dephasing**: Gaussian-distributed detuning noise from
           ground-Rydberg coherence limitations.
        2. **Interaction Fluctuations**: 3D position spread causes
           shot-to-shot variations in V_Ryd = C_6/R^6.

        Parameters
        ----------
        x : list of float
            Pulse parameters (format depends on strategy).
        n_shots : int, default=1000
            Number of Monte Carlo shots to perform.
        sigma_detuning : float or None
            Detuning noise standard deviation in Hz (e.g. 170e3 for 170 kHz).
            Set to None to disable dephasing.
        sigma_pos_xyz : tuple of 3 floats or None
            Position noise standard deviations ``(sigma_x, sigma_y, sigma_z)``
            in meters, where x-axis connects both atoms.
            Set to None to disable position fluctuations.
        seed : int or None, default=None
            Random seed for reproducibility.

        Returns
        -------
        MonteCarloResult
            Dataclass containing mean/std fidelity and infidelity,
            plus per-shot samples.
        """
        # Initialize RNG
        rng = np.random.default_rng(seed)

        # Compute noise parameters (gated by enable flags)
        sigma_delta_rad = (
            2 * np.pi * sigma_detuning
            if self.enable_rydberg_dephasing and sigma_detuning is not None
            else None
        )
        use_position = (
            self.enable_position_error and sigma_pos_xyz is not None
        )

        # Store original Hamiltonian
        original_ham_const = self.tq_ham_const.copy()
        original_v_ryd = self.v_ryd

        # Pre-compute Rydberg interaction operator for efficient updates
        ham_vdw_unit = self._build_vdw_unit_operator()

        # Convert position sigmas to μm (internal distance unit)
        if use_position:
            sx_um, sy_um, sz_um = [s * 1e6 for s in sigma_pos_xyz]

        # Storage for results
        fidelities = np.zeros(n_shots)
        detuning_samples = np.zeros(n_shots) if sigma_delta_rad else None
        distance_samples = np.zeros(n_shots) if use_position else None

        # Run Monte Carlo loop
        for shot in range(n_shots):
            # Reset Hamiltonian to original
            self.tq_ham_const = original_ham_const.copy()
            self.v_ryd = original_v_ryd

            # Apply dephasing (detuning noise)
            if sigma_delta_rad is not None:
                delta_err = rng.normal(0, sigma_delta_rad)
                detuning_samples[shot] = delta_err
                self._apply_detuning_perturbation(delta_err)

            # Apply 3D position fluctuations
            if use_position:
                dx1 = rng.normal(0, sx_um)
                dy1 = rng.normal(0, sy_um)
                dz1 = rng.normal(0, sz_um)
                dx2 = rng.normal(0, sx_um)
                dy2 = rng.normal(0, sy_um)
                dz2 = rng.normal(0, sz_um)
                R0 = self._get_nominal_distance()  # μm
                d_new = np.sqrt(
                    (R0 + dx1 - dx2) ** 2
                    + (dy1 - dy2) ** 2
                    + (dz1 - dz2) ** 2
                )
                # Clamp to avoid pathological cases
                d_new = max(d_new, 0.1)
                distance_samples[shot] = d_new
                self._apply_interaction_perturbation(d_new, ham_vdw_unit)

            # Compute fidelity for this shot
            infidelity = self._gate_infidelity_single(x)
            fidelities[shot] = 1.0 - infidelity

        # Restore original state
        self.tq_ham_const = original_ham_const
        self.v_ryd = original_v_ryd

        # Compute statistics
        mean_fid = float(np.mean(fidelities))
        std_fid = float(np.std(fidelities))
        infidelities = 1.0 - fidelities

        return MonteCarloResult(
            mean_fidelity=mean_fid,
            std_fidelity=std_fid,
            mean_infidelity=float(np.mean(infidelities)),
            std_infidelity=float(np.std(infidelities)),
            n_shots=n_shots,
            fidelities=fidelities,
            detuning_samples=detuning_samples,
            distance_samples=distance_samples,
        )

    def run_monte_carlo_jax(
        self,
        x: list[float],
        n_shots: int = 1000,
        sigma_detuning: float | None = None,
        sigma_pos_xyz: tuple[float, float, float] | None = None,
        seed: int = 0,
    ) -> MonteCarloResult:
        """GPU-accelerated Monte Carlo simulation using JAX.

        Functionally equivalent to :meth:`run_monte_carlo_simulation` but uses
        JAX ``odeint`` + ``vmap`` + ``jit`` to run all Monte Carlo shots in
        parallel on GPU (or CPU). Typically 10-100x faster for large *n_shots*.

        Parameters
        ----------
        x : list of float
            Pulse parameters (TO strategy only).
        n_shots : int, default=1000
            Number of Monte Carlo shots.
        sigma_detuning : float or None
            Detuning noise standard deviation in Hz. ``None`` disables dephasing.
        sigma_pos_xyz : tuple of 3 floats or None
            Position noise ``(sigma_x, sigma_y, sigma_z)`` in meters.
        seed : int, default=0
            PRNG seed for reproducibility.

        Returns
        -------
        MonteCarloResult
            Same format as :meth:`run_monte_carlo_simulation`.
        """
        import jax
        import jax.numpy as jnp
        from jax.experimental.ode import odeint

        jax.config.update("jax_enable_x64", True)

        if self.strategy != "TO":
            raise NotImplementedError(
                "run_monte_carlo_jax only supports TO strategy."
            )

        # --- Extract pulse parameters ---
        phase_amp = x[0]
        omega = x[1] * self.rabi_eff
        phase_init = x[2]
        delta = x[3] * self.rabi_eff
        theta = x[4]
        t_gate = x[5] * self.time_scale

        # --- Convert Hamiltonians to JAX arrays ---
        H_const_base = jnp.array(self.tq_ham_const)
        H_420 = jnp.array(self.tq_ham_420)
        H_420_conj = jnp.array(self.tq_ham_420_conj)
        H_1013 = jnp.array(self.tq_ham_1013)
        H_1013_conj = jnp.array(self.tq_ham_1013_conj)

        # Static coupling sum (time-independent)
        H_1013_sum = H_1013 + H_1013_conj

        # --- Build detuning perturbation mask ---
        pert_sq = np.zeros((7, 7), dtype=np.complex128)
        pert_sq[5, 5] = 1.0
        pert_sq[6, 6] = 1.0
        detuning_diag = jnp.array(
            np.diag(np.kron(np.eye(7), pert_sq) + np.kron(pert_sq, np.eye(7)))
        )  # shape (49,)

        # --- Build vdW perturbation unit operator ---
        ham_vdw_unit = jnp.array(self._build_vdw_unit_operator())
        d_nominal = self._get_nominal_distance()
        C_6 = self.v_ryd * d_nominal**6

        # --- Compute noise parameters (gated by enable flags) ---
        sigma_delta_rad = (
            2 * np.pi * sigma_detuning
            if self.enable_rydberg_dephasing and sigma_detuning is not None
            else None
        )
        use_position = (
            self.enable_position_error and sigma_pos_xyz is not None
        )

        # --- Sample perturbations ---
        key = jax.random.PRNGKey(seed)

        if sigma_delta_rad is not None:
            key, key1 = jax.random.split(key)
            delta_errs = jax.random.normal(key1, (n_shots,)) * sigma_delta_rad
        else:
            delta_errs = jnp.zeros(n_shots)

        if use_position:
            sx_um, sy_um, sz_um = [s * 1e6 for s in sigma_pos_xyz]
            keys = jax.random.split(key, 7)
            dx1 = jax.random.normal(keys[0], (n_shots,)) * sx_um
            dy1 = jax.random.normal(keys[1], (n_shots,)) * sy_um
            dz1 = jax.random.normal(keys[2], (n_shots,)) * sz_um
            dx2 = jax.random.normal(keys[3], (n_shots,)) * sx_um
            dy2 = jax.random.normal(keys[4], (n_shots,)) * sy_um
            dz2 = jax.random.normal(keys[5], (n_shots,)) * sz_um
            d_new = jnp.sqrt(
                (d_nominal + dx1 - dx2) ** 2
                + (dy1 - dy2) ** 2
                + (dz1 - dz2) ** 2
            )
            d_new = jnp.maximum(d_new, 0.1)
            delta_v = C_6 / d_new**6 - self.v_ryd
        else:
            d_new = jnp.full(n_shots, d_nominal)
            delta_v = jnp.zeros(n_shots)

        # --- Build batched H_const: shape (n_shots, 49, 49) ---
        H_const_batch = (
            H_const_base[None, :, :]
            + delta_errs[:, None, None] * jnp.diag(detuning_diag)[None, :, :]
            + delta_v[:, None, None] * ham_vdw_unit[None, :, :]
        )

        # --- Initial states ---
        y0_01 = jnp.array(
            np.kron([1 + 0j, 0, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0])
        )
        y0_11 = jnp.array(
            np.kron([0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0])
        )

        # --- Time grid ---
        t_eval = jnp.linspace(0.0, t_gate, 1000)

        # --- Blackman pulse support ---
        use_blackman = self.blackmanflag
        t_rise = self.t_rise

        # --- RHS of Schrödinger equation ---
        def rhs(y, t, H_c):
            phase_420 = jnp.exp(
                -1j * (phase_amp * jnp.cos(omega * t + phase_init) + delta * t)
            )
            if use_blackman:
                # JAX-compatible Blackman pulse
                bw = lambda tt, tr: (
                    0.42 - 0.5 * jnp.cos(2 * jnp.pi * tt / (2 * tr))
                    + 0.08 * jnp.cos(4 * jnp.pi * tt / (2 * tr))
                )
                amp_rise = bw(t, t_rise) * (t < t_rise)
                amp_fall = bw(t_gate - t, t_rise) * ((t_gate - t) < t_rise)
                amp_flat = ((t >= t_rise) & (t <= t_gate - t_rise)).astype(y.dtype)
                amplitude = amp_rise + amp_flat + amp_fall
            else:
                amplitude = 1.0
            H = (
                H_c
                + amplitude * phase_420 * H_420
                + amplitude * jnp.conj(phase_420) * H_420_conj
                + H_1013_sum
            )
            return -1j * (H @ y)

        # --- Single-shot: evolve |01⟩ and |11⟩, compute infidelity ---
        def single_shot_infidelity(H_c):
            sol_01 = odeint(rhs, y0_01, t_eval, H_c, rtol=1e-8, atol=1e-12)
            psi_01 = sol_01[-1]  # final state

            sol_11 = odeint(rhs, y0_11, t_eval, H_c, rtol=1e-8, atol=1e-12)
            psi_11 = sol_11[-1]

            # Overlaps
            a01 = jnp.exp(-1.0j * theta) * jnp.vdot(y0_01, psi_01)
            a11 = jnp.exp(-2.0j * theta - 1.0j * jnp.pi) * jnp.vdot(y0_11, psi_11)

            # Average gate fidelity
            avg_F = (1.0 / 20.0) * (
                jnp.abs(1.0 + 2.0 * a01 + a11) ** 2
                + 1.0
                + 2.0 * jnp.abs(a01) ** 2
                + jnp.abs(a11) ** 2
            )
            return 1.0 - avg_F.real

        # --- Vectorize + JIT ---
        batched_infidelity = jax.jit(jax.vmap(single_shot_infidelity))
        infidelities_jax = batched_infidelity(H_const_batch)

        # --- Convert to numpy ---
        infidelities = np.asarray(infidelities_jax)
        fidelities = 1.0 - infidelities

        mean_fid = float(np.mean(fidelities))
        std_fid = float(np.std(fidelities))

        return MonteCarloResult(
            mean_fidelity=mean_fid,
            std_fidelity=std_fid,
            mean_infidelity=float(np.mean(infidelities)),
            std_infidelity=float(np.std(infidelities)),
            n_shots=n_shots,
            fidelities=fidelities,
            detuning_samples=np.asarray(delta_errs) if sigma_delta_rad is not None else None,
            distance_samples=np.asarray(d_new) if use_position else None,
        )

    def _get_nominal_distance(self) -> float:
        """Get the nominal interatomic distance in μm.

        Returns
        -------
        float
            Interatomic distance in μm.
        """
        # For 'our' param_set, d=3μm is implicit in v_ryd calculation
        # v_ryd = 2π × 874e9 / 3^6
        # We need to extract d from C_6 / v_ryd
        if self.param_set == "our":
            return 3.0  # μm
        elif self.param_set == "lukin":
            # v_ryd = 2π × 450e6 for lukin
            # From paper, typical distance is ~4-5 μm
            # C_6 for n=53 is smaller, let's compute from v_ryd
            # Assume C_6 = 2π × 50e9 μm^6 for n=53 (approximate)
            return 3.0  # Approximate, adjust as needed
        return 3.0

    def _build_vdw_unit_operator(self) -> NDArray[np.complexfloating]:
        """Build the unit van der Waals interaction operator.

        Returns the operator that, when multiplied by V_ryd, gives the
        Rydberg-Rydberg interaction Hamiltonian.

        Returns
        -------
        ndarray
            Unit vdW operator of shape (49, 49).
        """
        ham_vdw_mat = np.zeros((7, 7))
        ham_vdw_mat[5][5] = 1
        ham_vdw_mat_garb = np.zeros((7, 7))
        ham_vdw_mat_garb[6][6] = 1
        return np.kron(
            ham_vdw_mat + ham_vdw_mat_garb, ham_vdw_mat + ham_vdw_mat_garb
        )


    def _apply_detuning_perturbation(self, delta_err: float) -> None:
        """Apply detuning perturbation to Rydberg states.

        Adds delta_err to the diagonal elements of the Hamiltonian
        corresponding to Rydberg states (indices 5 and 6 in single-atom basis).

        Parameters
        ----------
        delta_err : float
            Detuning error in rad/s.
        """
        # Build single-atom perturbation
        perturbation_sq = np.zeros((7, 7), dtype=np.complex128)
        perturbation_sq[5, 5] = delta_err  # |r⟩
        perturbation_sq[6, 6] = delta_err  # |r'⟩

        # Extend to two-atom space
        perturbation_tq = np.kron(np.eye(7), perturbation_sq) + np.kron(
            perturbation_sq, np.eye(7)
        )

        # Apply to Hamiltonian
        self.tq_ham_const = self.tq_ham_const + perturbation_tq

    def _apply_interaction_perturbation(
        self, d_new: float, ham_vdw_unit: NDArray[np.complexfloating]
    ) -> None:
        """Apply interaction strength perturbation from distance change.

        Updates the Rydberg-Rydberg interaction based on the new distance.

        Parameters
        ----------
        d_new : float
            New interatomic distance in μm.
        ham_vdw_unit : ndarray
            Unit van der Waals operator.
        """
        # Compute C_6 coefficient from original v_ryd and nominal distance
        d_nominal = self._get_nominal_distance()
        C_6 = self.v_ryd * d_nominal**6

        # Compute new v_ryd
        v_ryd_new = C_6 / d_new**6

        # Compute change in interaction
        delta_v = v_ryd_new - self.v_ryd

        # Apply perturbation
        self.tq_ham_const = self.tq_ham_const + delta_v * ham_vdw_unit
        self.v_ryd = v_ryd_new

    def get_error_budget(
        self,
        x: list[float],
        n_shots: int = 1000,
        sigma_detuning: float | None = None,
        sigma_pos_xyz: tuple[float, float, float] | None = None,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Compute error budget breakdown for the CZ gate.

        Runs separate Monte Carlo simulations to isolate contributions
        from different error sources, similar to Extended Data Fig. 4
        from arXiv:2305.03406.

        Parameters
        ----------
        x : list of float
            Pulse parameters.
        n_shots : int, default=1000
            Number of Monte Carlo shots per error source.
        sigma_detuning : float or None
            Detuning noise standard deviation in Hz (e.g. 170e3).
        sigma_pos_xyz : tuple of 3 floats or None
            Position noise ``(sigma_x, sigma_y, sigma_z)`` in meters.
        seed : int or None, default=None
            Random seed for reproducibility.

        Returns
        -------
        dict
            Error budget with keys:
            - 'ideal_infidelity': Infidelity without noise
            - 'dephasing_infidelity': Infidelity with only dephasing
            - 'position_infidelity': Infidelity with only position fluctuations
            - 'total_infidelity': Infidelity with both error sources
            - 'dephasing_contribution': Isolated dephasing contribution
            - 'position_contribution': Isolated position contribution
        """
        # Ideal (no noise)
        ideal_infidelity = self._gate_infidelity_single(x)

        # Dephasing only
        result_deph = self.run_monte_carlo_simulation(
            x,
            n_shots=n_shots,
            sigma_detuning=sigma_detuning,
            sigma_pos_xyz=None,
            seed=seed,
        )

        # Position fluctuations only
        result_pos = self.run_monte_carlo_simulation(
            x,
            n_shots=n_shots,
            sigma_detuning=None,
            sigma_pos_xyz=sigma_pos_xyz,
            seed=seed + 1 if seed else None,
        )

        # Both error sources
        result_both = self.run_monte_carlo_simulation(
            x,
            n_shots=n_shots,
            sigma_detuning=sigma_detuning,
            sigma_pos_xyz=sigma_pos_xyz,
            seed=seed + 2 if seed else None,
        )

        return {
            "ideal_infidelity": ideal_infidelity,
            "dephasing_infidelity": result_deph.mean_infidelity,
            "dephasing_std": result_deph.std_infidelity,
            "position_infidelity": result_pos.mean_infidelity,
            "position_std": result_pos.std_infidelity,
            "total_infidelity": result_both.mean_infidelity,
            "total_std": result_both.std_infidelity,
            "dephasing_contribution": result_deph.mean_infidelity - ideal_infidelity,
            "position_contribution": result_pos.mean_infidelity - ideal_infidelity,
        }

    # ==================================================================
    # HAMILTONIAN CONSTRUCTION
    # ==================================================================

    def _tq_ham_const(self) -> NDArray[np.complexfloating]:
        """Build the time-independent two-atom Hamiltonian.

        Includes intermediate state detunings, Rydberg detunings, decay rates
        (as imaginary energy), and Rydberg-Rydberg van der Waals interaction.
        Decay and polarization leakage are controlled by ``self.enable_*`` flags.

        Returns
        -------
        ndarray
            Complex Hamiltonian matrix of shape (49, 49).
        """
        ham_tq_mat = np.zeros((49, 49), dtype=np.complex128)
        ham_sq_mat = np.zeros((7, 7), dtype=np.complex128)
        delta = 0
        middecay = self.mid_state_decay_rate if self.enable_intermediate_decay else 0
        ryddecay = self.ryd_state_decay_rate if self.enable_rydberg_decay else 0

        # Intermediate state energies with hyperfine splitting
        ham_sq_mat[2][2] = self.Delta - 2 * np.pi * 51e6 - 1j * middecay / 2
        ham_sq_mat[3][3] = self.Delta - 1j * middecay / 2
        ham_sq_mat[4][4] = self.Delta + 2 * np.pi * 87e6 - 1j * middecay / 2

        # Rydberg state energies
        ham_sq_mat[5][5] = delta - 1j * ryddecay / 2
        ham_sq_mat[6][6] = delta + self.ryd_zeeman_shift - 1j * ryddecay / 2

        # Two-atom Hamiltonian via Kronecker products
        ham_tq_mat = ham_tq_mat + np.kron(np.eye(7), ham_sq_mat)
        ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat, np.eye(7))

        # Rydberg-Rydberg van der Waals interaction
        ham_vdw_mat = np.zeros((7, 7))
        ham_vdw_mat[5][5] = 1
        ham_vdw_mat_garb = np.zeros((7, 7))
        ham_vdw_mat_garb[6][6] = 1
        ham_tq_mat = ham_tq_mat + self.v_ryd * np.kron(
            ham_vdw_mat + ham_vdw_mat_garb, ham_vdw_mat + ham_vdw_mat_garb
        )
        return ham_tq_mat

    def _occ_operator(self, index: int) -> NDArray[np.complexfloating]:
        """Build occupation number operator for level `index`.

        Creates the operator |i⟩⟨i| ⊗ I + I ⊗ |i⟩⟨i| for measuring total
        population in level i across both atoms.

        Parameters
        ----------
        index : int
            Single-atom level index (0-6).

        Returns
        -------
        ndarray
            Occupation operator of shape (49, 49).
        """
        oper_tq = np.zeros((49, 49), dtype=np.complex128)
        oper_sq = np.zeros((7, 7), dtype=np.complex128)
        oper_sq[index][index] = 1
        oper_tq = oper_tq + np.kron(np.eye(7), oper_sq)
        oper_tq = oper_tq + np.kron(oper_sq, np.eye(7))
        return oper_tq

    def _tq_ham_420_our(self) -> NDArray[np.complexfloating]:
        """Build 420nm laser coupling Hamiltonian for 'our' parameter set.

        Couples ground state |1⟩ to intermediate states |e1⟩, |e2⟩, |e3⟩
        with Clebsch-Gordan coefficients for σ⁻ polarization.

        Returns
        -------
        ndarray
            Coupling Hamiltonian of shape (49, 49).
        """
        ham_tq_mat = np.zeros((49, 49), dtype=np.complex128)
        ham_sq_mat = np.zeros((7, 7), dtype=np.complex128)

        # |1⟩ → |e1⟩, |e2⟩, |e3⟩ transitions
        ham_sq_mat[2][1] = (
            self.rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, 1, -1)
            + self.rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, 1, -1)
        ) / 2
        ham_sq_mat[3][1] = (
            self.rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, 2, -1)
            + self.rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, 2, -1)
        ) / 2
        ham_sq_mat[4][1] = (
            self.rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, 3, -1)
            + self.rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, 3, -1)
        ) / 2

        ham_tq_mat = ham_tq_mat + np.kron(np.eye(7), ham_sq_mat)
        ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat, np.eye(7))
        return ham_tq_mat

    def _tq_ham_1013_our(self) -> NDArray[np.complexfloating]:
        """Build 1013nm laser coupling Hamiltonian for 'our' parameter set.

        Couples intermediate states to Rydberg states |r⟩ and |r'⟩.

        Returns
        -------
        ndarray
            Coupling Hamiltonian of shape (49, 49).
        """
        ham_tq_mat = np.zeros((49, 49), dtype=np.complex128)
        ham_sq_mat = np.zeros((7, 7), dtype=np.complex128)

        # |e⟩ → |r⟩ transitions
        ham_sq_mat[5][2] = (self.rabi_1013 / 2) * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, 1, -1)
        ham_sq_mat[5][3] = (self.rabi_1013 / 2) * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, 2, -1)
        ham_sq_mat[5][4] = (self.rabi_1013 / 2) * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, 3, -1)

        # |e⟩ → |r'⟩ (garbage) transitions
        ham_sq_mat[6][2] = (self.rabi_1013_garbage / 2) * CG(
            3 / 2, -1 / 2, 3 / 2, -1 / 2, 1, -1
        )
        ham_sq_mat[6][3] = (self.rabi_1013_garbage / 2) * CG(
            3 / 2, -1 / 2, 3 / 2, -1 / 2, 2, -1
        )
        ham_sq_mat[6][4] = (self.rabi_1013_garbage / 2) * CG(
            3 / 2, -1 / 2, 3 / 2, -1 / 2, 3, -1
        )

        ham_tq_mat = ham_tq_mat + np.kron(np.eye(7), ham_sq_mat)
        ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat, np.eye(7))
        return ham_tq_mat

    def _tq_ham_420_lukin(self) -> NDArray[np.complexfloating]:
        """Build 420nm laser coupling Hamiltonian for 'lukin' parameter set.

        Uses σ⁺ polarization appropriate for Harvard experiment geometry.

        Returns
        -------
        ndarray
            Coupling Hamiltonian of shape (49, 49).
        """
        ham_tq_mat = np.zeros((49, 49), dtype=np.complex128)
        ham_sq_mat = np.zeros((7, 7), dtype=np.complex128)

        ham_sq_mat[2][1] = (
            self.rabi_420 * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, 1, 1)
            + self.rabi_420_garbage * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, 1, 1)
        ) / 2
        ham_sq_mat[3][1] = (
            self.rabi_420 * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, 2, 1)
            + self.rabi_420_garbage * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, 2, 1)
        ) / 2
        ham_sq_mat[4][1] = (
            self.rabi_420 * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, 3, 1)
            + self.rabi_420_garbage * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, 3, 1)
        ) / 2

        ham_tq_mat = ham_tq_mat + np.kron(np.eye(7), ham_sq_mat)
        ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat, np.eye(7))
        return ham_tq_mat

    def _tq_ham_1013_lukin(self) -> NDArray[np.complexfloating]:
        """Build 1013nm laser coupling Hamiltonian for 'lukin' parameter set.

        Returns
        -------
        ndarray
            Coupling Hamiltonian of shape (49, 49).
        """
        ham_tq_mat = np.zeros((49, 49), dtype=np.complex128)
        ham_sq_mat = np.zeros((7, 7), dtype=np.complex128)

        ham_sq_mat[5][2] = (self.rabi_1013 / 2) * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, 1, 1)
        ham_sq_mat[5][3] = (self.rabi_1013 / 2) * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, 2, 1)
        ham_sq_mat[5][4] = (self.rabi_1013 / 2) * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, 3, 1)

        ham_sq_mat[6][2] = (self.rabi_1013_garbage / 2) * CG(
            3 / 2, 1 / 2, 3 / 2, 1 / 2, 1, 1
        )
        ham_sq_mat[6][3] = (self.rabi_1013_garbage / 2) * CG(
            3 / 2, 1 / 2, 3 / 2, 1 / 2, 2, 1
        )
        ham_sq_mat[6][4] = (self.rabi_1013_garbage / 2) * CG(
            3 / 2, 1 / 2, 3 / 2, 1 / 2, 3, 1
        )

        ham_tq_mat = ham_tq_mat + np.kron(np.eye(7), ham_sq_mat)
        ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat, np.eye(7))
        return ham_tq_mat

    def _decay_integrate(
        self,
        t_list: NDArray[np.floating],
        occ_list: NDArray[np.floating],
        decay_rate: float,
    ) -> NDArray[np.floating]:
        """Integrate decay probability given time-dependent occupation.

        Parameters
        ----------
        t_list : ndarray
            Time points array.
        occ_list : ndarray
            Occupation probability at each time point.
        decay_rate : float
            Spontaneous emission rate (1/s).

        Returns
        -------
        ndarray
            Cumulative decay probability at each time point.
        """
        poly_interpolation = interpolate.CubicSpline(t_list, occ_list)
        args = (poly_interpolation, decay_rate)

        def fun(
            t: float,
            _y: NDArray,
            poly_interpolation: interpolate.CubicSpline,
            decay_rate: float,
        ) -> NDArray:
            diff = decay_rate * poly_interpolation.__call__(t)
            return np.array([diff])

        t_span = [0, t_list[-1]]
        result = integrate.solve_ivp(
            fun,
            t_span,
            np.array([0]),
            t_eval=t_list,
            args=args,
            method="DOP853",
            rtol=1e-8,
            atol=1e-12,
        )
        return np.array(result.y)

    @staticmethod
    def _build_sss_state_map() -> dict[str, NDArray[np.complexfloating]]:
        """Build mapping from state labels to 49-dimensional state vectors.

        Returns
        -------
        dict
            Mapping from state label strings to complex state vectors of shape (49,).
        """
        s0 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=complex)
        s1 = np.array([0, 1, 0, 0, 0, 0, 0], dtype=complex)
        state_00 = np.kron(s0, s0)
        state_01 = np.kron(s0, s1)
        state_10 = np.kron(s1, s0)
        state_11 = np.kron(s1, s1)
        return {
            "00": state_00,
            "01": state_01,
            "10": state_10,
            "11": state_11,
            "SSS-0": 0.5*state_00 + 0.5*state_01 + 0.5*state_10 + 0.5*state_11,
            "SSS-1": 0.5*state_00 - 0.5*state_01 - 0.5*state_10 + 0.5*state_11,
            "SSS-2": 0.5*state_00 + 0.5j*state_01 + 0.5j*state_10 - 0.5*state_11,
            "SSS-3": 0.5*state_00 - 0.5j*state_01 - 0.5j*state_10 - 0.5*state_11,
            "SSS-4": state_00,
            "SSS-5": state_11,
            "SSS-6": 0.5*state_00 + 0.5*state_01 + 0.5*state_10 - 0.5*state_11,
            "SSS-7": 0.5*state_00 - 0.5*state_01 - 0.5*state_10 - 0.5*state_11,
            "SSS-8": 0.5*state_00 + 0.5j*state_01 + 0.5j*state_10 + 0.5*state_11,
            "SSS-9": 0.5*state_00 - 0.5j*state_01 - 0.5j*state_10 + 0.5*state_11,
            "SSS-10": state_00/np.sqrt(2) + 1j*state_11/np.sqrt(2),
            "SSS-11": state_00/np.sqrt(2) - 1j*state_11/np.sqrt(2),
        }

    # ==================================================================
    # TIME-OPTIMAL (TO) STRATEGY - INTERNAL
    # ==================================================================

    def _get_gate_result_TO(
        self,
        phase_amp: float,
        omega: float,
        phase_init: float,
        delta: float,
        t_gate: float,
        state_mat: NDArray[np.complexfloating],
        t_eval: NDArray[np.floating] | None = None,
    ) -> NDArray[np.complexfloating]:
        """Evolve quantum state under Time-Optimal pulse protocol.

        Integrates the Schrödinger equation with phase-modulated 420nm laser:
        φ(t) = phase_amp · cos(omega·t + phase_init) + delta·t

        Parameters
        ----------
        phase_amp : float
            Amplitude of cosine phase modulation (radians).
        omega : float
            Angular frequency of phase modulation (rad/s).
        phase_init : float
            Initial phase of cosine modulation (radians).
        delta : float
            Linear phase chirp rate (rad/s).
        t_gate : float
            Total gate duration (seconds).
        state_mat : ndarray
            Initial state vector of shape (49,).
        t_eval : ndarray or None, optional
            Times at which to store the solution. If None, only the final
            state is returned as a 1-D array of shape (49,).

        Returns
        -------
        ndarray
            If t_eval is provided: shape (49, len(t_eval)).
            If t_eval is None: shape (49,) — the final state only.
        """
        # Precompute static Hamiltonian (folds 1013 into const)
        ham_static = self.tq_ham_const + self.tq_ham_1013 + self.tq_ham_1013_conj

        def fun(
            t: float,
            y: NDArray,
            phase_init: float,
            omega: float,
            phase_amp: float,
            delta: float,
            t_gate: float,
        ) -> NDArray:
            phase_420 = np.exp(
                -1j * (phase_amp * np.cos(omega * t + phase_init) + delta * t)
            )
            phase_420_conj = np.conjugate(phase_420)
            amplitude = blackman_pulse(t, self.t_rise, t_gate) if self.blackmanflag else 1

            ham_tq_mat = (
                ham_static
                + amplitude * phase_420 * self.tq_ham_420
                + amplitude * phase_420_conj * self.tq_ham_420_conj
            )
            return -1j * ham_tq_mat @ y

        args = (phase_init, omega, phase_amp, delta, t_gate)
        t_span = [0, t_gate]
        solve_kwargs = dict(
            method="DOP853",
            rtol=1e-8,
            atol=1e-12,
        )
        if t_eval is not None:
            solve_kwargs["t_eval"] = t_eval
        result = integrate.solve_ivp(
            fun,
            t_span,
            state_mat,
            args=args,
            **solve_kwargs,
        )
        if t_eval is not None:
            return np.array(result.y)
        return np.array(result.y[:, -1])

    def fidelity_avg(self, x: list[float]) -> float:
        """Calculate average gate infidelity for TO parameters.

        Computes overlap with ideal CZ gate for |01⟩ and |11⟩ initial states
        and returns the average infidelity.

        Parameters
        ----------
        x : list of float
            TO parameters [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale].

        Returns
        -------
        float
            Average gate infidelity (1 - F).
        """
        theta = x[4]

        # Compute |01⟩ → |01⟩ overlap
        ini_state = np.kron(
            [1 + 0j, 0, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
        )
        res = self._get_gate_result_TO(
            phase_amp=x[0],
            omega=x[1] * self.rabi_eff,
            phase_init=x[2],
            delta=x[3] * self.rabi_eff,
            t_gate=x[5] * self.time_scale,
            state_mat=ini_state,
        )
        a01 = np.exp(-1.0j * theta) * ini_state.conj().dot(res.T)

        # Compute |11⟩ → -|11⟩ overlap (CZ applies π phase)
        ini_state = np.kron(
            [0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
        )
        res = self._get_gate_result_TO(
            phase_amp=x[0],
            omega=x[1] * self.rabi_eff,
            phase_init=x[2],
            delta=x[3] * self.rabi_eff,
            t_gate=x[5] * self.time_scale,
            state_mat=ini_state,
        )
        a11 = np.exp(-2.0j * theta - 1.0j * np.pi) * ini_state.conj().dot(res.T)

        # Average gate fidelity formula
        avg_F = (1 / 20) * (abs(1 + 2 * a01 + a11) ** 2 + 1 + 2 * abs(a01) ** 2 + abs(a11) ** 2)
        print(f"Average Fidelity: {avg_F}")
        return 1 - avg_F

    def _optimization_TO(self, fid_type, x: list[float] = None) -> object:
        """Run Nelder-Mead optimization for TO pulse parameters.

        Parameters
        ----------
        x : list of float
            Initial guess [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale].
            If None, use default parameters.
        fid_type : str
            Fidelity metric passed to :meth:`gate_fidelity`.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Optimization result.
        """
        if x is None:
            x = self.x_initial
        if fid_type == "average":
            raw_objective = self.fidelity_avg
        if fid_type == "bell":
            raw_objective = self.fidelity_bell
        if fid_type == "sss":
            raw_objective = self.fidelity_sss

        cache = {}

        def objective(x):
            val = raw_objective(x)
            cache['last_val'] = val
            return val

        def callback_func(x: list[float], saveflag: bool = False) -> None:
            if saveflag:
                with open("opt_hf_new.txt", "a") as f:
                    for var in x:
                        f.write("{:.9f},".format(var))
                    f.write("\n")
            print("Current iteration parameters:", x, "Infidelity:", cache.get('last_val', '?'))

        bounds = (
            (-np.pi, np.pi),
            (-10, 10),
            (-np.pi, np.pi),
            (-2, 2),
            (-np.inf, np.inf),
            (-np.pi, np.pi),
        )
        optimres = minimize(
            fun=objective,
            x0=x,
            method="Nelder-Mead",
            options={"disp": True, "fatol": 1e-9},
            bounds=bounds,
            callback=callback_func,
        )
        print(f"The final optimized protocol is: {optimres.x.tolist()}, with infidelity: {optimres.fun}")
        return optimres

    def _diagnose_run_TO(
        self, x: list[float], initial: InitialStateLabel,
    ) -> list[NDArray[np.floating]]:
        """Run TO simulation and return population time series.

        Parameters
        ----------
        x : list of float
            TO parameters [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale].
        initial : str
            Initial two-qubit state. Supports computational basis states
            ('00', '01', '10', '11') and SSS states ('SSS-0' through 'SSS-11').

        Returns
        -------
        list of ndarray
            [mid_state_pop, ryd_state_pop, ryd_garb_pop] arrays.
        """
        sss_states = self._build_sss_state_map()
        if initial not in sss_states:
            raise ValueError(f"Unsupported initial state: '{initial}'")
        ini_state = sss_states[initial]

        t_gate = x[5] * self.time_scale
        # res_list[:, t] stores state at time t_gate * t/999
        res_list = self._get_gate_result_TO(
            phase_amp=x[0],
            omega=x[1] * self.rabi_eff,
            phase_init=x[2],
            delta=x[3] * self.rabi_eff,
            t_gate=t_gate,
            state_mat=ini_state,
            t_eval=np.linspace(0, t_gate, 1000),
        )

        mid_state_occ_oper = (
            self._occ_operator(2) + self._occ_operator(3) + self._occ_operator(4)
        )
        ryd_state_occ_oper = self._occ_operator(5)
        ryd_garb_state_occ_oper = self._occ_operator(6)

        mid_state_list = []
        ryd_state_list = []
        ryd_garb_list = []

        for col in range(len(res_list[0, :])):
            state_temp = res_list[:, col]
            mid_state_occ_temp = np.dot(
                np.conjugate(state_temp),
                np.reshape(
                    np.matmul(mid_state_occ_oper, np.reshape(state_temp, (-1, 1))), (-1)
                ),
            )
            ryd_state_occ_temp = np.dot(
                np.conjugate(state_temp),
                np.reshape(
                    np.matmul(ryd_state_occ_oper, np.reshape(state_temp, (-1, 1))), (-1)
                ),
            )
            ryd_garb_occ_temp = np.dot(
                np.conjugate(state_temp),
                np.reshape(
                    np.matmul(ryd_garb_state_occ_oper, np.reshape(state_temp, (-1, 1))),
                    (-1),
                ),
            )
            mid_state_list.append(np.abs(mid_state_occ_temp))
            ryd_state_list.append(np.abs(ryd_state_occ_temp))
            ryd_garb_list.append(np.abs(ryd_garb_occ_temp))

        return [
            np.array(mid_state_list),
            np.array(ryd_state_list),
            np.array(ryd_garb_list),
        ]

    def _diagnose_plot_TO(
        self, x: list[float], initial: InitialStateLabel,
    ) -> None:
        """Generate population evolution plot for TO strategy.

        Parameters
        ----------
        x : list of float
            TO parameters.
        initial : str
            Initial state label.
        """
        population_evolution = self._diagnose_run_TO(x, initial)
        t_gate = x[5] * self.time_scale
        time_axis_ns = np.linspace(0, t_gate * 1e9, len(population_evolution[0]))

        plt.style.use("seaborn-v0_8-whitegrid")
        _fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            time_axis_ns,
            population_evolution[0],
            label=f"Intermediate states population for |{initial}⟩ state",
            lw=2,
        )
        ax.plot(
            time_axis_ns,
            population_evolution[1],
            label="Rydberg state |r⟩ population",
            linestyle="--",
            lw=2,
        )
        ax.plot(
            time_axis_ns,
            population_evolution[2],
            label="Unwanted Rydberg |r'⟩ population",
            linestyle=":",
            lw=2,
        )
        ax.set_title("Population Evolution During CPHASE Gate (TO)", fontsize=16)
        ax.set_xlabel("Time (ns)", fontsize=12)
        ax.set_ylabel("Population Probability", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"population_TO_{initial}.png")
        plt.show()

    def _plotBloch_TO(self, x: list[float], saveflag: bool = True) -> None:
        """Generate Bloch sphere plots for TO strategy.

        Parameters
        ----------
        x : list of float
            TO parameters.
        saveflag : bool
            Whether to save plots.
        """
        basis_01 = np.kron([1 + 0j, 0, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0])
        basis_0r = np.kron([1 + 0j, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1 + 0j, 0])

        basis_01_mat = np.reshape(basis_01, (-1, 1))
        basis_0r_mat = np.reshape(basis_0r, (-1, 1))

        # First Bloch sphere: |01⟩ ↔ |0r⟩
        sigmaz = basis_01_mat.dot(basis_01_mat.T) - basis_0r_mat.dot(basis_0r_mat.T)
        sigmax = basis_0r_mat.dot(basis_01_mat.T) + basis_01_mat.dot(basis_0r_mat.T)
        sigmay = -1j * basis_01_mat.dot(basis_0r_mat.T) + 1j * basis_0r_mat.dot(
            basis_01_mat.T
        )

        zlist = []
        xlist = []
        ylist = []

        ini_state = basis_01
        t_gate = x[5] * self.time_scale
        res = self._get_gate_result_TO(
            phase_amp=x[0],
            omega=x[1] * self.rabi_eff,
            phase_init=x[2],
            delta=x[3] * self.rabi_eff,
            t_gate=t_gate,
            state_mat=ini_state,
            t_eval=np.linspace(0, t_gate, 1000),
        )

        for t in range(len(res[0, :])):
            state = res[:, t]
            zlist.append(
                np.matmul(state.reshape(-1).conj(), sigmaz).dot(state.reshape(-1, 1))[0]
            )
            xlist.append(
                np.matmul(state.reshape(-1).conj(), sigmax).dot(state.reshape(-1, 1))[0]
            )
            ylist.append(
                np.matmul(state.reshape(-1).conj(), sigmay).dot(state.reshape(-1, 1))[0]
            )

        b = Bloch()
        b.zlabel = [r"$ |01 \rangle$ ", r"$|0r\rangle$"]
        pnts = [np.array(xlist), np.array(ylist), np.array(zlist)]
        b.point_color = ["#CC6600"]
        b.point_size = [5]
        b.point_marker = ["^"]
        b.add_points(pnts)
        b.make_sphere()

        # Second Bloch sphere: |11⟩ ↔ |W⟩
        basis_11 = np.kron([0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0])
        basis_W = np.sqrt(1 / 2) * (
            np.kron([0, 1 + 0j, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1 + 0j, 0])
            + np.kron([0, 0, 0, 0, 0, 1 + 0j, 0], [0, 1 + 0j, 0, 0, 0, 0, 0])
        )
        basis_11_mat = np.reshape(basis_11, (-1, 1))
        basis_W_mat = np.reshape(basis_W, (-1, 1))

        sigmaz = basis_11_mat.dot(basis_11_mat.T) - basis_W_mat.dot(basis_W_mat.T)
        sigmax = basis_W_mat.dot(basis_11_mat.T) + basis_11_mat.dot(basis_W_mat.T)
        sigmay = -1j * basis_11_mat.dot(basis_W_mat.T) + 1j * basis_W_mat.dot(
            basis_11_mat.T
        )

        zlist = []
        xlist = []
        ylist = []

        ini_state = basis_11
        res = self._get_gate_result_TO(
            phase_amp=x[0],
            omega=x[1] * self.rabi_eff,
            phase_init=x[2],
            delta=x[3] * self.rabi_eff,
            t_gate=t_gate,
            state_mat=ini_state,
            t_eval=np.linspace(0, t_gate, 1000),
        )

        for t in range(len(res[0, :])):
            state = res[:, t]
            zlist.append(
                np.matmul(state.reshape(-1).conj(), sigmaz).dot(state.reshape(-1, 1))[0]
            )
            xlist.append(
                np.matmul(state.reshape(-1).conj(), sigmax).dot(state.reshape(-1, 1))[0]
            )
            ylist.append(
                np.matmul(state.reshape(-1).conj(), sigmay).dot(state.reshape(-1, 1))[0]
            )

        b2 = Bloch()
        b2.zlabel = [r"$ |11 \rangle$ ", r"$|W\rangle$"]
        pnts = [np.array(xlist), np.array(ylist), np.array(zlist)]
        b2.point_color = ["r"]
        b2.point_size = [5]
        b2.point_marker = ["^"]
        b2.add_points(pnts)
        b2.make_sphere()

        if saveflag:
            b.save("10-r0_Bloch")
            b2.save("11-W_Bloch")
        plt.show()

    # ==================================================================
    # AMPLITUDE-ROBUST (AR) STRATEGY - INTERNAL
    # ==================================================================

    def _get_gate_result_AR(
        self,
        omega: float,
        phase_amp1: float,
        phase_init1: float,
        phase_amp2: float,
        phase_init2: float,
        delta: float,
        t_gate: float,
        state_mat: NDArray[np.complexfloating],
        t_eval: NDArray[np.floating] | None = None,
    ) -> NDArray[np.complexfloating]:
        """Evolve quantum state under Amplitude-Robust pulse protocol.

        Integrates the Schrödinger equation with dual-frequency phase modulation:
        φ(t) = A₁·sin(ωt + φ₁) + A₂·sin(2ωt + φ₂) + δ·t

        Parameters
        ----------
        omega : float
            Base angular frequency of phase modulation (rad/s).
        phase_amp1 : float
            Amplitude of first sine term (radians).
        phase_init1 : float
            Phase offset of first sine term (radians).
        phase_amp2 : float
            Amplitude of second (2ω) sine term (radians).
        phase_init2 : float
            Phase offset of second sine term (radians).
        delta : float
            Linear phase chirp rate (rad/s).
        t_gate : float
            Total gate duration (seconds).
        state_mat : ndarray
            Initial state vector of shape (49,).
        t_eval : ndarray or None, optional
            Times at which to store the solution. If None, only the final
            state is returned as a 1-D array of shape (49,).

        Returns
        -------
        ndarray
            If t_eval is provided: shape (49, len(t_eval)).
            If t_eval is None: shape (49,) — the final state only.
        """
        # Precompute static Hamiltonian (folds 1013 into const)
        ham_static = self.tq_ham_const + self.tq_ham_1013 + self.tq_ham_1013_conj

        def fun(
            t: float,
            y: NDArray,
            omega: float,
            phase_init1: float,
            phase_amp1: float,
            phase_init2: float,
            phase_amp2: float,
            delta: float,
            t_gate: float,
        ) -> NDArray:
            phase_420 = np.exp(
                -1j
                * (
                    phase_amp1 * np.sin(omega * t + phase_init1)
                    + phase_amp2 * np.sin(2 * omega * t + phase_init2)
                    + delta * t
                )
            )
            phase_420_conj = np.conjugate(phase_420)
            amplitude = blackman_pulse(t, self.t_rise, t_gate) if self.blackmanflag else 1

            ham_tq_mat = (
                ham_static
                + amplitude * phase_420 * self.tq_ham_420
                + amplitude * phase_420_conj * self.tq_ham_420_conj
            )
            return -1j * ham_tq_mat @ y

        args = (omega, phase_init1, phase_amp1, phase_init2, phase_amp2, delta, t_gate)
        t_span = [0, t_gate]
        solve_kwargs = dict(
            method="DOP853",
            rtol=1e-8,
            atol=1e-12,
        )
        if t_eval is not None:
            solve_kwargs["t_eval"] = t_eval
        result = integrate.solve_ivp(
            fun,
            t_span,
            state_mat,
            args=args,
            **solve_kwargs,
        )
        if t_eval is not None:
            return np.array(result.y)
        return np.array(result.y[:, -1])

    def _avg_fidelity_AR(self, x: list[float]) -> float:
        """Calculate average gate infidelity for AR parameters.

        Parameters
        ----------
        x : list of float
            AR parameters [ω/Ω_eff, A₁, φ₁, A₂, φ₂, δ/Ω_eff, T/T_scale, θ].

        Returns
        -------
        float
            Average gate infidelity (1 - F).
        """
        theta = x[-1]

        # Compute |01⟩ overlap
        ini_state = np.kron(
            [1 + 0j, 0, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
        )
        res = self._get_gate_result_AR(
            omega=x[0] * self.rabi_eff,
            phase_amp1=x[1],
            phase_init1=x[2],
            phase_amp2=x[3],
            phase_init2=x[4],
            delta=x[5] * self.rabi_eff,
            t_gate=x[6] * self.time_scale,
            state_mat=ini_state,
        )
        a01 = np.exp(-1.0j * theta) * ini_state.conj().dot(res.T)

        # Compute |11⟩ overlap
        ini_state = np.kron(
            [0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
        )
        res = self._get_gate_result_AR(
            omega=x[0] * self.rabi_eff,
            phase_amp1=x[1],
            phase_init1=x[2],
            phase_amp2=x[3],
            phase_init2=x[4],
            delta=x[5] * self.rabi_eff,
            t_gate=x[6] * self.time_scale,
            state_mat=ini_state,
        )
        a11 = np.exp(-2.0j * theta - 1.0j * np.pi) * ini_state.conj().dot(res.T)

        # Average gate fidelity
        avg_F = (1 / 20) * (
            abs(1 + 2 * a01 + a11) ** 2 + 1 + 2 * abs(a01) ** 2 + abs(a11) ** 2
        )
        return 1 - avg_F

    def _optimization_AR(self, x: list[float] | None = None, fid_type: str = "average") -> object:
        """Run Nelder-Mead optimization for AR pulse parameters.

        Parameters
        ----------
        x : list of float, optional
            Initial guess [ω/Ω_eff, A₁, φ₁, A₂, φ₂, δ/Ω_eff, T/T_scale, θ].
            If None, uses stored parameters.
        fid_type : str
            Fidelity metric passed to :meth:`gate_fidelity`.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Optimization result.
        """
        if x is None:
            x = self.x_initial
        else:
            print(f"AR parameters is overwritten to: [ω/Ω_eff, A₁, φ₁, A₂, φ₂, δ/Ω_eff, T/T_scale, θ] = {x}")

        cache = {}

        def objective(x):
            val = self._gate_infidelity_single(x, fid_type=fid_type)
            cache['last_val'] = val
            return val

        def callback_func(x: list[float], saveflag: bool = False) -> None:
            if saveflag:
                with open("opt_hf_new.txt", "a") as f:
                    for var in x:
                        f.write("{:.9f},".format(var))
                    f.write("\n")
            print("Current iteration parameters:", x, "Infidelity:", cache.get('last_val', '?'))
            print(f"overwrite protocol from {self.x_initial} to {x}")
            self._setup_protocol_AR(x)

        bounds = (
            (-10, 10),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-2, 2),
            (-np.inf, np.inf),
            (-np.pi, np.pi),
        )
        optimres = minimize(
            fun=objective,
            x0=x,
            method="Nelder-Mead",
            options={"disp": True, "fatol": 1e-9},
            bounds=bounds,
            callback=callback_func,
        )
        print(f"The final optimized protocol is: {optimres.x.tolist()}, with infidelity: {optimres.fun}")
        return optimres

    def _diagnose_run_AR(
        self, x: list[float], initial: InitialStateLabel,
    ) -> list[NDArray[np.floating]]:
        """Run AR simulation and return population time series.

        Parameters
        ----------
        x : list of float
            AR parameters [ω/Ω_eff, A₁, φ₁, A₂, φ₂, δ/Ω_eff, T/T_scale, θ].
        initial : str
            Initial two-qubit state. Supports computational basis states
            ('00', '01', '10', '11') and SSS states ('SSS-0' through 'SSS-11').

        Returns
        -------
        list of ndarray
            [mid_state_pop, ryd_state_pop, ryd_garb_pop] arrays.
        """
        sss_states = self._build_sss_state_map()
        if initial not in sss_states:
            raise ValueError(f"Unsupported initial state: '{initial}'")
        ini_state = sss_states[initial]

        t_gate = x[6] * self.time_scale
        # res_list[:, t] stores state at time t_gate * t/999
        res_list = self._get_gate_result_AR(
            omega=x[0] * self.rabi_eff,
            phase_amp1=x[1],
            phase_init1=x[2],
            phase_amp2=x[3],
            phase_init2=x[4],
            delta=x[5] * self.rabi_eff,
            t_gate=t_gate,
            state_mat=ini_state,
            t_eval=np.linspace(0, t_gate, 1000),
        )

        mid_state_occ_oper = (
            self._occ_operator(2) + self._occ_operator(3) + self._occ_operator(4)
        )
        ryd_state_occ_oper = self._occ_operator(5)
        ryd_garb_state_occ_oper = self._occ_operator(6)

        mid_state_list = []
        ryd_state_list = []
        ryd_garb_list = []

        for col in range(len(res_list[0, :])):
            state_temp = res_list[:, col]
            mid_state_occ_temp = np.dot(
                np.conjugate(state_temp),
                np.reshape(
                    np.matmul(mid_state_occ_oper, np.reshape(state_temp, (-1, 1))), (-1)
                ),
            )
            ryd_state_occ_temp = np.dot(
                np.conjugate(state_temp),
                np.reshape(
                    np.matmul(ryd_state_occ_oper, np.reshape(state_temp, (-1, 1))), (-1)
                ),
            )
            ryd_garb_occ_temp = np.dot(
                np.conjugate(state_temp),
                np.reshape(
                    np.matmul(ryd_garb_state_occ_oper, np.reshape(state_temp, (-1, 1))),
                    (-1),
                ),
            )
            mid_state_list.append(np.abs(mid_state_occ_temp))
            ryd_state_list.append(np.abs(ryd_state_occ_temp))
            ryd_garb_list.append(np.abs(ryd_garb_occ_temp))

        return [
            np.array(mid_state_list),
            np.array(ryd_state_list),
            np.array(ryd_garb_list),
        ]

    def _diagnose_plot_AR(
        self, x: list[float], initial: InitialStateLabel,
    ) -> None:
        """Generate population evolution plot for AR strategy.

        Parameters
        ----------
        x : list of float
            AR parameters.
        initial : str
            Initial state label. Supports '00', '01', '10', '11'
            and 'SSS-0' through 'SSS-11'.
        """
        population_evolution = self._diagnose_run_AR(x, initial)
        t_gate = x[6] * self.time_scale
        time_axis_ns = np.linspace(0, t_gate * 1e9, len(population_evolution[0]))

        plt.style.use("seaborn-v0_8-whitegrid")
        _fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            time_axis_ns,
            population_evolution[0],
            label=f"Intermediate states population for |{initial}⟩ state",
            lw=2,
        )
        ax.plot(
            time_axis_ns,
            population_evolution[1],
            label="Rydberg state |r⟩ population",
            linestyle="--",
            lw=2,
        )
        ax.plot(
            time_axis_ns,
            population_evolution[2],
            label="Unwanted Rydberg |r'⟩ population",
            linestyle=":",
            lw=2,
        )
        ax.set_title("Population Evolution During CPHASE Gate (AR)", fontsize=16)
        ax.set_xlabel("Time (ns)", fontsize=12)
        ax.set_ylabel("Population Probability", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"population_AR_{initial}.png")
        plt.show()
