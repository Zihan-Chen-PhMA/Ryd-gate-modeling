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
from scipy.constants import k as kB
from scipy.optimize import minimize

from ryd_gate.blackman import blackman_pulse

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    decayflag : bool
        Whether to include spontaneous emission as imaginary energy shifts.
        When True, Rydberg and intermediate state decay rates are added to
        the diagonal Hamiltonian elements.
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
    Basic optimization with TO strategy:

    >>> sim = CZGateSimulator(decayflag=False, param_set='our', strategy='TO')
    >>> x0 = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]  # Initial pulse parameters
    >>> result = sim.optimize(x0)
    >>> print(f"Optimized infidelity: {result.fun:.6f}")

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
        decayflag: bool,
        param_set: Literal["our", "lukin"] = "our",
        strategy: Literal["TO", "AR"] = "AR",
        blackmanflag: bool = True,
    ) -> None:
        """Initialize the CZ gate simulator with specified parameters."""
        self.param_set = param_set
        self.strategy = strategy
        self.blackmanflag = blackmanflag

        if param_set == "our":
            self._init_our_params(decayflag)
        elif param_set == "lukin":
            self._init_lukin_params(decayflag)
        else:
            raise ValueError(
                f"Unknown parameter set: '{param_set}'. Choose 'our' or 'lukin'."
            )

    # ==================================================================
    # INITIALIZATION HELPERS
    # ==================================================================

    def _init_our_params(self, decayflag: bool) -> None:
        """Initialize 'our' lab experimental parameters (n=70 Rydberg).

        Parameters
        ----------
        decayflag : bool
            Whether to include decay rates in the Hamiltonian.
        """
        self.atom = Rubidium87()
        self.temperature = 300 # K
        # Rydberg level and laser parameters
        self.ryd_level = 70
        # Assumes rabi_420 = rabi_1013, effective Rabi = 7 MHz
        self.Delta = 2 * np.pi * 9.1e9  # Intermediate detuning (rad/s)
        self.rabi_420 = 2*np.pi*(491)*10**(6)
        self.rabi_1013 = 2*np.pi*(185)*10**(6)
        self.rabi_eff = self.rabi_420*self.rabi_1013/(2*self.Delta) # Effective two-photon Rabi (rad/s)
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
        self.ryd_zeeman_shift = -2 * np.pi * 56e6

        # Decay rate parameters
        # 6P3/2 lifetime 120.7 ± 1.2 ns, refer from https://arxiv.org/abs/physics/0409077
        self.mid_state_decay_rate = 1 / (110.7e-9)  
        self.mid_garb_decay_rate = 1 / (110.7e-9)
        # refer to the data from https://arxiv.org/abs/0810.0339, Table VII, 70S1/2 @ 300 K
        self.ryd_state_decay_rate = 1 / (151.55e-6)  
        # Suppose the garbage state has the identical decay rate
        self.ryd_garb_decay_rate = 1 / (151.55e-6)

        # Build Hamiltonians
        self.tq_ham_const = self._tq_ham_const(decayflag)
        self.tq_ham_420 = self._tq_ham_420_our()
        self.tq_ham_1013 = self._tq_ham_1013_our()
        self.tq_ham_420_conj = self._tq_ham_420_our().conj().T
        self.tq_ham_1013_conj = self._tq_ham_1013_our().conj().T
        self.t_rise = 20e-9  # Blackman pulse rise time

    def _init_lukin_params(self, decayflag: bool) -> None:
        """Initialize 'lukin' (Harvard) experimental parameters (n=53 Rydberg).

        Parameters
        ----------
        decayflag : bool
            Whether to include decay rates in the Hamiltonian.
        """
        self.atom = Rubidium87()

        # Rydberg level and laser parameters
        self.ryd_level = 53
        self.Delta = 2 * np.pi * 7.8e9  # Intermediate detuning (rad/s)
        self.rabi_420 = 2 * np.pi * 237e6
        self.rabi_1013 = 2 * np.pi * 303e6
        self.rabi_eff = self.rabi_420 * self.rabi_1013 / (2 * self.Delta)
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
        self.ryd_zeeman_shift = 2 * np.pi * 2.4e9

        # Decay rate parameters
        self.mid_state_decay_rate = 1 / (110e-9)
        self.mid_garb_decay_rate = 1 / (110e-9)
        self.ryd_state_decay_rate = 1 / (88e-6)  # 53S lifetime ~88 μs
        self.ryd_garb_decay_rate = 1 / (88e-6)

        # Build Hamiltonians
        self.tq_ham_const = self._tq_ham_const(decayflag)
        self.tq_ham_420 = self._tq_ham_420_lukin()
        self.tq_ham_1013 = self._tq_ham_1013_lukin()
        self.tq_ham_420_conj = self._tq_ham_420_lukin().conj().T
        self.tq_ham_1013_conj = self._tq_ham_1013_lukin().conj().T
        self.t_rise = 20e-9  # Blackman pulse rise time

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def optimize(self, x_initial: list[float]) -> object:
        """Run pulse parameter optimization for the configured strategy.

        Parameters
        ----------
        x_initial : list of float
            Initial guess for pulse parameters. Format depends on strategy:
            - TO: [A, ω, φ₀, δ, θ, T] (6 parameters)
            - AR: [ω, A₁, φ₁, A₂, φ₂, δ, T, θ] (8 parameters)

        Returns
        -------
        scipy.optimize.OptimizeResult
            Optimization result containing optimal parameters in `x` attribute
            and final infidelity in `fun` attribute.
        """
        if self.strategy == "TO":
            return self._optimization_TO(x_initial)
        elif self.strategy == "AR":
            return self._optimization_AR(x_initial)
        else:
            raise ValueError(
                f"Unknown strategy: '{self.strategy}'. Choose 'TO' or 'AR'."
            )

    def avg_fidelity(self, x: list[float]) -> float:
        """Calculate average gate infidelity for given pulse parameters.

        Parameters
        ----------
        x : list of float
            Pulse parameters (format depends on strategy).

        Returns
        -------
        float
            Average gate infidelity (1 - F), where F is the fidelity.
            Value is bounded between 0 and 1.
        """
        if self.strategy == "TO":
            return self._avg_fidelity_TO(x)
        elif self.strategy == "AR":
            return self._avg_fidelity_AR(x)
        else:
            raise ValueError(
                f"Unknown strategy: '{self.strategy}'. Choose 'TO' or 'AR'."
            )

    def diagnose_plot(
        self, x: list[float], initial_state: Literal["00", "01", "10", "11"]
    ) -> None:
        """Generate population evolution plot for diagnostic analysis.

        Plots the time evolution of intermediate state, Rydberg state, and
        unwanted Rydberg state populations during the gate.

        Parameters
        ----------
        x : list of float
            Pulse parameters (format depends on strategy).
        initial_state : {'00', '01', '10', '11'}
            Two-qubit initial state label.
        """
        if self.strategy == "TO":
            return self._diagnose_plot_TO(x, initial_state)
        elif self.strategy == "AR":
            return self._diagnose_plot_AR(x, initial_state)
        else:
            raise ValueError(
                f"Unknown strategy: '{self.strategy}'. Choose 'TO' or 'AR'."
            )

    def diagnose_run(
        self, x: list[float], initial_state: Literal["00", "01", "10", "11"]
    ) -> list[NDArray[np.floating]]:
        """Run diagnostic simulation and return population arrays.

        Parameters
        ----------
        x : list of float
            Pulse parameters (format depends on strategy).
        initial_state : {'00', '01', '10', '11'}
            Two-qubit initial state label.

        Returns
        -------
        list of ndarray
            [mid_state_pop, ryd_state_pop, ryd_garb_pop] arrays of shape (1000,).
        """
        if self.strategy == "TO":
            return self._diagnose_run_TO(x, initial_state)
        elif self.strategy == "AR":
            return self._diagnose_run_AR(x, initial_state)
        else:
            raise ValueError(
                f"Unknown strategy: '{self.strategy}'. Choose 'TO' or 'AR'."
            )

    def plot_bloch(self, x: list[float], save: bool = True) -> None:
        """Generate Bloch sphere visualization for key transitions.

        Plots trajectories for |01⟩ → |0r⟩ and |11⟩ → |W⟩ transitions on
        separate Bloch spheres. Only implemented for TO strategy.

        Parameters
        ----------
        x : list of float
            TO pulse parameters [A, ω, φ₀, δ, θ, T].
        save : bool, default=True
            Whether to save plots as PNG files.
        """
        if self.strategy == "TO":
            return self._plotBloch_TO(x, save)
        else:
            print("Bloch sphere plot is only implemented for the 'TO' strategy.")
            return

    # ==================================================================
    # MONTE CARLO SIMULATION
    # ==================================================================

    def run_monte_carlo_simulation(
        self,
        x: list[float],
        n_shots: int = 1000,
        T2_star: float | None = 3e-6,
        temperature: float | None = None,
        trap_freq: float | None = None,
        sigma_pos: float | None = None,
        seed: int | None = None,
    ) -> MonteCarloResult:
        """Run quasi-static Monte Carlo simulation for error budget analysis.

        Simulates physical error mechanisms via statistical averaging over many
        shots, following the methodology from arXiv:2305.03406 (Lukin group).

        This implements two key error sources:
        1. **Doppler Dephasing (T2*)**: Gaussian-distributed detuning noise from
           ground-Rydberg coherence limitations due to Doppler shifts and laser
           phase noise.
        2. **Interaction Fluctuations**: Position spread from finite atomic
           temperature causes shot-to-shot variations in V_Ryd = C_6/R^6.

        Parameters
        ----------
        x : list of float
            Pulse parameters (format depends on strategy).
        n_shots : int, default=1000
            Number of Monte Carlo shots to perform.
        T2_star : float or None, default=3e-6
            Ground-Rydberg coherence time in seconds. Set to None to disable
            Doppler dephasing. The default 3 μs matches the paper.
            Detuning noise σ_δ ≈ √2/T2* (Gaussian decay profile).
        temperature : float or None, default=None
            Atomic temperature in μK. Set to None to disable position
            fluctuations. Typical experimental value is 10-20 μK.
        trap_freq : float or None, default=None
            Trap frequency in kHz. Required if temperature is specified and
            sigma_pos is not directly provided.
        sigma_pos : float or None, default=None
            Position spread standard deviation in μm. If provided, overrides
            the calculation from temperature and trap_freq.
        seed : int or None, default=None
            Random seed for reproducibility.

        Returns
        -------
        MonteCarloResult
            Dataclass containing:
            - mean_fidelity: Average gate fidelity
            - std_fidelity: Standard deviation of fidelity
            - mean_infidelity: Average gate infidelity (1 - F)
            - std_infidelity: Standard deviation of infidelity
            - n_shots: Number of shots performed
            - fidelities: Array of individual fidelities
            - detuning_samples: Sampled detuning errors (if T2* enabled)
            - distance_samples: Sampled distances (if position enabled)

        Notes
        -----
        **Doppler Dephasing Model**:
        Assumes a Gaussian decay profile exp(-(t/T2*)²) for the coherence,
        which corresponds to Gaussian-distributed detuning noise with
        σ_δ = √2 / T2* (in rad/s). For each shot, a random detuning
        δ_err ~ N(0, σ_δ²) is added to the Rydberg state diagonal elements.

        **Position Fluctuation Model**:
        Atoms in a harmonic trap at temperature T have position spread
        σ_pos = √(k_B T / (m ω²)). For each shot, independent displacements
        δr_1, δr_2 ~ N(0, σ_pos²) are sampled, and the effective distance
        R_new = R_0 + δr_1 - δr_2 is used to compute V_Ryd = C_6/R_new^6.

        **Efficiency**:
        The base Hamiltonian is reused; only shot-specific perturbations
        (detuning and interaction strength) are recomputed per shot.

        Examples
        --------
        Run with T2* dephasing only:

        >>> sim = CZGateSimulator(decayflag=False, strategy='TO')
        >>> result = sim.run_monte_carlo_simulation(x_opt, n_shots=500, T2_star=3e-6)
        >>> print(f"Mean fidelity: {result.mean_fidelity:.6f} ± {result.std_fidelity:.6f}")

        Run with both error sources:

        >>> result = sim.run_monte_carlo_simulation(
        ...     x_opt, n_shots=1000, T2_star=3e-6, temperature=15.0, trap_freq=50.0
        ... )

        Disable T2* to isolate position fluctuation contribution:

        >>> result = sim.run_monte_carlo_simulation(
        ...     x_opt, n_shots=1000, T2_star=None, temperature=15.0, trap_freq=50.0
        ... )
        """
        # Validate inputs
        if temperature is not None and trap_freq is None and sigma_pos is None:
            raise ValueError(
                "Either trap_freq or sigma_pos must be provided when temperature is set."
            )

        # Initialize RNG
        rng = np.random.default_rng(seed)

        # Compute noise parameters
        sigma_delta = self._compute_detuning_sigma(T2_star) if T2_star else None
        sigma_position = self._compute_position_sigma(temperature, trap_freq, sigma_pos)

        # Store original Hamiltonian
        original_ham_const = self.tq_ham_const.copy()
        original_v_ryd = self.v_ryd

        # Pre-compute Rydberg interaction operator for efficient updates
        ham_vdw_unit = self._build_vdw_unit_operator()

        # Storage for results
        fidelities = np.zeros(n_shots)
        detuning_samples = np.zeros(n_shots) if sigma_delta else None
        distance_samples = np.zeros(n_shots) if sigma_position else None

        # Run Monte Carlo loop
        for shot in range(n_shots):
            # Reset Hamiltonian to original
            self.tq_ham_const = original_ham_const.copy()
            self.v_ryd = original_v_ryd

            # Apply T2* dephasing (detuning noise)
            if sigma_delta is not None:
                delta_err = rng.normal(0, sigma_delta)
                detuning_samples[shot] = delta_err
                self._apply_detuning_perturbation(delta_err)

            # Apply position fluctuations
            if sigma_position is not None:
                delta_r1 = rng.normal(0, sigma_position)
                delta_r2 = rng.normal(0, sigma_position)
                d_new = self._get_nominal_distance() + delta_r1 - delta_r2
                # Clamp to avoid pathological cases
                d_new = max(d_new, 0.1)
                distance_samples[shot] = d_new
                self._apply_interaction_perturbation(d_new, ham_vdw_unit)

            # Compute fidelity for this shot
            infidelity = self.avg_fidelity(x)
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

    def _compute_detuning_sigma(self, T2_star: float) -> float:
        """Compute detuning noise standard deviation from T2* coherence time.

        For a Gaussian decay profile exp(-(t/T2*)²), the corresponding
        detuning noise has σ_δ = √2 / T2* in rad/s.

        Parameters
        ----------
        T2_star : float
            Coherence time in seconds.

        Returns
        -------
        float
            Detuning standard deviation in rad/s.
        """
        return np.sqrt(2) / T2_star

    def _compute_position_sigma(
        self,
        temperature: float | None,
        trap_freq: float | None,
        sigma_pos: float | None,
    ) -> float | None:
        """Compute position spread from thermal parameters.

        Parameters
        ----------
        temperature : float or None
            Atomic temperature in μK.
        trap_freq : float or None
            Trap frequency in kHz.
        sigma_pos : float or None
            Direct position spread in μm (overrides calculation).

        Returns
        -------
        float or None
            Position standard deviation in μm, or None if disabled.
        """
        if sigma_pos is not None:
            return sigma_pos
        if temperature is None:
            return None
        if trap_freq is None or trap_freq <= 0:
            raise ValueError("trap_freq must be positive when temperature is specified.")

        # Convert units
        T_K = temperature * 1e-6  # μK to K
        omega = 2 * np.pi * trap_freq * 1e3  # kHz to rad/s
        m_Rb = 87 * 1.66054e-27  # 87Rb mass in kg

        # σ_pos = sqrt(k_B T / (m ω²))
        sigma_m = np.sqrt(kB * T_K / (m_Rb * omega**2))
        return sigma_m * 1e6  # Convert to μm

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
        ham_vdw_mat_garb = np.zeros((7, 7))
        ham_vdw_mat[5][5] = 1
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
        T2_star: float = 3e-6,
        temperature: float = 15.0,
        trap_freq: float = 50.0,
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
        T2_star : float, default=3e-6
            Ground-Rydberg coherence time in seconds.
        temperature : float, default=15.0
            Atomic temperature in μK.
        trap_freq : float, default=50.0
            Trap frequency in kHz.
        seed : int or None, default=None
            Random seed for reproducibility.

        Returns
        -------
        dict
            Error budget with keys:
            - 'ideal_infidelity': Infidelity without noise
            - 'T2_star_infidelity': Infidelity with only T2* dephasing
            - 'position_infidelity': Infidelity with only position fluctuations
            - 'total_infidelity': Infidelity with both error sources
            - 'T2_star_contribution': Isolated T2* contribution
            - 'position_contribution': Isolated position contribution
        """
        # Ideal (no noise)
        ideal_infidelity = self.avg_fidelity(x)

        # T2* only
        result_t2 = self.run_monte_carlo_simulation(
            x,
            n_shots=n_shots,
            T2_star=T2_star,
            temperature=None,
            seed=seed,
        )

        # Position fluctuations only
        result_pos = self.run_monte_carlo_simulation(
            x,
            n_shots=n_shots,
            T2_star=None,
            temperature=temperature,
            trap_freq=trap_freq,
            seed=seed + 1 if seed else None,
        )

        # Both error sources
        result_both = self.run_monte_carlo_simulation(
            x,
            n_shots=n_shots,
            T2_star=T2_star,
            temperature=temperature,
            trap_freq=trap_freq,
            seed=seed + 2 if seed else None,
        )

        return {
            "ideal_infidelity": ideal_infidelity,
            "T2_star_infidelity": result_t2.mean_infidelity,
            "T2_star_std": result_t2.std_infidelity,
            "position_infidelity": result_pos.mean_infidelity,
            "position_std": result_pos.std_infidelity,
            "total_infidelity": result_both.mean_infidelity,
            "total_std": result_both.std_infidelity,
            "T2_star_contribution": result_t2.mean_infidelity - ideal_infidelity,
            "position_contribution": result_pos.mean_infidelity - ideal_infidelity,
        }

    # ==================================================================
    # HAMILTONIAN CONSTRUCTION
    # ==================================================================

    def _tq_ham_const(self, decayflag: bool) -> NDArray[np.complexfloating]:
        """Build the time-independent two-atom Hamiltonian.

        Includes intermediate state detunings, Rydberg detunings, decay rates
        (as imaginary energy), and Rydberg-Rydberg van der Waals interaction.

        Parameters
        ----------
        decayflag : bool
            Whether to include decay as imaginary energy shifts.

        Returns
        -------
        ndarray
            Complex Hamiltonian matrix of shape (49, 49).
        """
        ham_tq_mat = np.zeros((49, 49), dtype=np.complex128)
        ham_sq_mat = np.zeros((7, 7), dtype=np.complex128)
        delta = 0
        middecay = self.mid_state_decay_rate if decayflag else 0
        ryddecay = self.ryd_state_decay_rate if decayflag else 0

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
        ham_vdw_mat_garb = np.zeros((7, 7))
        ham_vdw_mat[5][5] = 1
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

        Returns
        -------
        ndarray
            State evolution array of shape (49, 1000) where column t
            contains the state at time t_gate * t/999.
        """

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
                self.tq_ham_const
                + amplitude * phase_420 * self.tq_ham_420
                + amplitude * phase_420_conj * self.tq_ham_420_conj
                + self.tq_ham_1013
                + self.tq_ham_1013_conj
            )
            diff = -1j * ham_tq_mat
            y_arr = np.reshape(np.array(y), (-1, 1))
            return np.reshape(np.matmul(diff, y_arr), (-1))

        args = (phase_init, omega, phase_amp, delta, t_gate)
        t_span = [0, t_gate]
        t_eval = np.linspace(0, t_gate, 1000)
        result = integrate.solve_ivp(
            fun,
            t_span,
            state_mat,
            t_eval=t_eval,
            args=args,
            method="DOP853",
            rtol=1e-8,
            atol=1e-12,
        )
        return np.array(result.y)

    def _avg_fidelity_TO(self, x: list[float]) -> float:
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
        )[:, -1]
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
        )[:, -1]
        a11 = np.exp(-2.0j * theta - 1.0j * np.pi) * ini_state.conj().dot(res.T)

        # Average gate fidelity formula
        avg_F = (1 / 20) * (abs(1 + 2 * a01 + a11) ** 2 + 1 + 2 * abs(a01) ** 2 + abs(a11) ** 2)
        return 1 - avg_F

    def _optimization_TO(self, x: list[float]) -> object:
        """Run Nelder-Mead optimization for TO pulse parameters.

        Parameters
        ----------
        x : list of float
            Initial guess [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale].

        Returns
        -------
        scipy.optimize.OptimizeResult
            Optimization result.
        """

        def callback_func(x: list[float]) -> None:
            with open("opt_hf_new.txt", "a") as f:
                for var in x:
                    f.write("{:.9f},".format(var))
                f.write("\n")
            print("Current iteration parameters:", x)

        bounds = (
            (-np.pi, np.pi),
            (-10, 10),
            (-np.pi, np.pi),
            (-2, 2),
            (-np.inf, np.inf),
            (-np.pi, np.pi),
        )
        optimres = minimize(
            fun=self._avg_fidelity_TO,
            x0=x,
            method="Nelder-Mead",
            options={"disp": True, "fatol": 1e-9},
            bounds=bounds,
            callback=callback_func,
        )
        return optimres

    def _diagnose_run_TO(
        self, x: list[float],
        initial: Literal["00", "01", "10", "11",
                         "SSS-0", "SSS-1", "SSS-2", "SSS-3", "SSS-4", "SSS-5",
                         "SSS-6", "SSS-7", "SSS-8", "SSS-9", "SSS-10", "SSS-11"]
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
        # Build basis states
        s0 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=complex)
        s1 = np.array([0, 1, 0, 0, 0, 0, 0], dtype=complex)
        state_00 = np.kron(s0, s0)
        state_01 = np.kron(s0, s1)
        state_10 = np.kron(s1, s0)
        state_11 = np.kron(s1, s1)

        # Map labels to state vectors
        sss_states = {
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

        if initial not in sss_states:
            raise ValueError(f"Unsupported initial state: '{initial}'")
        ini_state = sss_states[initial]

        # res_list[:, t] stores state at time t_gate * t/999
        res_list = self._get_gate_result_TO(
            phase_amp=x[0],
            omega=x[1] * self.rabi_eff,
            phase_init=x[2],
            delta=x[3] * self.rabi_eff,
            t_gate=x[5] * self.time_scale,
            state_mat=ini_state,
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
        self, x: list[float],
        initial: Literal["00", "01", "10", "11",
                         "SSS-0", "SSS-1", "SSS-2", "SSS-3", "SSS-4", "SSS-5",
                         "SSS-6", "SSS-7", "SSS-8", "SSS-9", "SSS-10", "SSS-11"]
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
        res = self._get_gate_result_TO(
            phase_amp=x[0],
            omega=x[1] * self.rabi_eff,
            phase_init=x[2],
            delta=x[3] * self.rabi_eff,
            t_gate=x[5] * self.time_scale,
            state_mat=ini_state,
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
            t_gate=x[5] * self.time_scale,
            state_mat=ini_state,
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

        Returns
        -------
        ndarray
            State evolution array of shape (49, 1000).
        """

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
                self.tq_ham_const
                + amplitude * phase_420 * self.tq_ham_420
                + amplitude * phase_420_conj * self.tq_ham_420_conj
                + self.tq_ham_1013
                + self.tq_ham_1013_conj
            )
            diff = -1j * ham_tq_mat
            y_arr = np.reshape(np.array(y), (-1, 1))
            return np.reshape(np.matmul(diff, y_arr), (-1))

        args = (omega, phase_init1, phase_amp1, phase_init2, phase_amp2, delta, t_gate)
        t_span = [0, t_gate]
        t_eval = np.linspace(0, t_gate, 1000)
        result = integrate.solve_ivp(
            fun,
            t_span,
            state_mat,
            t_eval=t_eval,
            args=args,
            method="DOP853",
            rtol=1e-8,
            atol=1e-12,
        )
        return np.array(result.y)

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
        )[:, -1]
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
        )[:, -1]
        a11 = np.exp(-2.0j * theta - 1.0j * np.pi) * ini_state.conj().dot(res.T)

        # Average gate fidelity
        avg_F = (1 / 20) * (
            abs(1 + 2 * a01 + a11) ** 2 + 1 + 2 * abs(a01) ** 2 + abs(a11) ** 2
        )
        return 1 - avg_F

    def _optimization_AR(self, x: list[float]) -> object:
        """Run Nelder-Mead optimization for AR pulse parameters.

        Parameters
        ----------
        x : list of float
            Initial guess [ω/Ω_eff, A₁, φ₁, A₂, φ₂, δ/Ω_eff, T/T_scale, θ].

        Returns
        -------
        scipy.optimize.OptimizeResult
            Optimization result.
        """

        def callback_func(x: list[float]) -> None:
            with open("opt_hf_new.txt", "a") as f:
                for var in x:
                    f.write("{:.9f},".format(var))
                f.write("\n")
            print("parameters:", x, "Infidelity:", self._avg_fidelity_AR(x))

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
            fun=self._avg_fidelity_AR,
            x0=x,
            method="Nelder-Mead",
            options={"disp": True, "fatol": 1e-9},
            bounds=bounds,
            callback=callback_func,
        )
        return optimres

    def _diagnose_run_AR(
        self, x: list[float], initial: Literal["00", "01", "10", "11"]
    ) -> list[NDArray[np.floating]]:
        """Run AR simulation and return population time series.

        Parameters
        ----------
        x : list of float
            AR parameters [ω/Ω_eff, A₁, φ₁, A₂, φ₂, δ/Ω_eff, T/T_scale, θ].
        initial : {'00', '01', '10', '11'}
            Initial two-qubit state.

        Returns
        -------
        list of ndarray
            [mid_state_pop, ryd_state_pop, ryd_garb_pop] arrays.
        """
        if initial == "11":
            ini_state = np.kron(
                [0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
            )
        elif initial == "10":
            ini_state = np.kron(
                [0, 1 + 0j, 0, 0, 0, 0, 0], [1 + 0j, 0, 0, 0, 0, 0, 0]
            )
        elif initial == "01":
            ini_state = np.kron(
                [1 + 0j, 0, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
            )
        elif initial == "00":
            ini_state = np.kron(
                [1 + 0j, 0, 0, 0, 0, 0, 0], [1 + 0j, 0, 0, 0, 0, 0, 0]
            )
        else:
            raise ValueError(f"Unsupported initial state: '{initial}'")

        # res_list[:, t] stores state at time t_gate * t/999
        res_list = self._get_gate_result_AR(
            omega=x[0] * self.rabi_eff,
            phase_amp1=x[1],
            phase_init1=x[2],
            phase_amp2=x[3],
            phase_init2=x[4],
            delta=x[5] * self.rabi_eff,
            t_gate=x[6] * self.time_scale,
            state_mat=ini_state,
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
        self, x: list[float], initial: Literal["00", "01", "10", "11"]
    ) -> None:
        """Generate population evolution plot for AR strategy.

        Parameters
        ----------
        x : list of float
            AR parameters.
        initial : str
            Initial state label.
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
