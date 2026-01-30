"""Discrete and analytical pulse optimisation for robust CZ gates.

Supports Time-Optimal (TO), Amplitude-Robust (AR), Doppler-Robust (DR),
and Stark-Shift-Robust (SSR) protocols.
"""

import time
from functools import partial

import numpy as np
from scipy import integrate
from scipy.linalg import expm
from scipy.optimize import curve_fit, minimize


class PulseOptimizer:
    """Find and optimise robust quantum gate control pulses.

    Supports multi-step discrete optimisation followed by an optional
    high-precision optimisation of a fitted analytical pulse shape using an
    ODE solver.
    """

    SIGMA_PLUS = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    SIGMA_MINUS = SIGMA_PLUS.T.conj()
    RYDBERG_PROJECTOR = np.array([[0, 0], [0, 1]], dtype=np.complex128)

    def __init__(self, pulse_type="AR", zeta=0.1):
        self.pulse_type = pulse_type
        self._all_configs = {
            "TO": {"name": "Time-Optimal (TO)", "gatetime": 7.5},
            "AR": {"name": "Amplitude-Robust (AR)", "gatetime": 14.32},
            "DR": {"name": "Doppler-Robust (DR)", "gatetime": 8.8},
            "SSR": {"name": "Stark-Shift-Robust (SSR)", "gatetime": 14.5, "zeta": zeta},
        }

        if pulse_type not in self._all_configs:
            raise ValueError(f"Invalid pulse_type. Choose from {list(self._all_configs.keys())}")

        self.config = self._all_configs[pulse_type]
        self.config["cost_func"] = self._get_cost_function()
        self.result = None
        self.analytical_result = None

    # --- Static helpers ---
    @staticmethod
    def _get_H0_01(phase, A=1.0):
        return 0.5 * A * (
            np.exp(1.0j * phase) * PulseOptimizer.SIGMA_PLUS + np.exp(-1.0j * phase) * PulseOptimizer.SIGMA_MINUS
        )

    @staticmethod
    def _get_H0_11(phase, A=1.0):
        return 0.5 * np.sqrt(2) * A * (
            np.exp(1.0j * phase) * PulseOptimizer.SIGMA_PLUS + np.exp(-1.0j * phase) * PulseOptimizer.SIGMA_MINUS
        )

    @staticmethod
    def _analytical_pulse_shape(t, A, B, w, p1, p2, C, D):
        """Analytical pulse shape with two harmonic terms and a linear ramp."""
        return A * np.sin(w * t + p1) + B * np.sin(2 * w * t + p2) + C * t + D

    # --- Private core logic ---
    def _callback(self, x, args, start_time):
        elapsed = time.time() - start_time
        cost = args["cost_func"](x, args)
        print(f"Elapsed: {elapsed:.1f}s | Discrete Cost (J): {cost:.6f}")

    def _get_cost_function(self):
        return {
            "TO": self._cost_function_TO,
            "AR": self._cost_function_AR,
            "DR": self._cost_function_DR,
            "SSR": self._cost_function_SSR,
        }[self.pulse_type]

    def _evolve_state_discrete(self, M, T, phases, H0_func, H1_func):
        dt = T / M if M > 0 else 0
        psi_combined = np.array([1, 0, 0, 0], dtype=np.complex128)
        if M == 0:
            return psi_combined[:2], psi_combined[2:]
        for i in range(M):
            phase = phases[i]
            H0, H1 = H0_func(phase), H1_func(phase)
            H_block = np.block([[H0, np.zeros_like(H0)], [H1, H0]])
            psi_combined = expm(-1j * H_block * dt) @ psi_combined
        return psi_combined[:2], psi_combined[2:]

    def _cost_function_TO(self, ulist, args):
        T, M, theta = args["gatetime"], args["M"], ulist[-1]
        phases = ulist[:-1]
        dt = T / M if M > 0 else 0
        psi_final_01 = np.array([1, 0], dtype=np.complex128)
        psi_final_11 = np.array([1, 0], dtype=np.complex128)
        if M > 0:
            for i in range(M):
                psi_final_01 = expm(-1j * self._get_H0_01(phases[i]) * dt) @ psi_final_01
            for i in range(M):
                psi_final_11 = expm(-1j * self._get_H0_11(phases[i]) * dt) @ psi_final_11
        a01 = np.exp(-1.0j * theta) * psi_final_01[0]
        a11 = np.exp(-2.0j * theta - 1.0j * np.pi) * psi_final_11[0]
        F = (1 / 16) * np.abs(1 + 2 * a01 + a11) ** 2
        return 1 - F

    def _cost_function_AR(self, ulist, args):
        T, M, theta = args["gatetime"], args["M"], ulist[-1]
        psi0_01, psi1_01 = self._evolve_state_discrete(M, T, ulist[:-1], self._get_H0_01, self._get_H0_01)
        psi0_11, psi1_11 = self._evolve_state_discrete(M, T, ulist[:-1], self._get_H0_11, self._get_H0_11)
        a01 = np.exp(-1.0j * theta) * psi0_01[0]
        a11 = np.exp(-2.0j * theta - 1.0j * np.pi) * psi0_11[0]
        F = (1 / 16) * np.abs(1 + 2 * a01 + a11) ** 2
        return (1 - F) + 2 * np.linalg.norm(psi1_01) ** 2 + np.linalg.norm(psi1_11) ** 2

    def _cost_function_DR(self, ulist, args):
        T, M, theta = args["gatetime"], args["M"], ulist[-1]
        H1_detuning = lambda phase: self.RYDBERG_PROJECTOR
        psi0_01, psi1_01 = self._evolve_state_discrete(M, T, ulist[:-1], self._get_H0_01, H1_detuning)
        psi0_11, psi1_11 = self._evolve_state_discrete(M, T, ulist[:-1], self._get_H0_11, H1_detuning)
        a01 = np.exp(-1.0j * theta) * psi0_01[0]
        a11 = np.exp(-2.0j * theta - 1.0j * np.pi / 2.0) * psi0_11[0]
        F_half = (1 / 16) * np.abs(1 + 2 * a01 + a11) ** 2
        return (1 - F_half) + 2 * np.abs(psi1_01[1]) ** 2 + np.abs(psi1_11[1]) ** 2

    def _cost_function_SSR(self, ulist, args):
        T, M, theta, zeta = args["gatetime"], args["M"], ulist[-1], args["zeta"]
        H1_ssr_01 = lambda p: self._get_H0_01(p) + zeta * self.RYDBERG_PROJECTOR
        H1_ssr_11 = lambda p: self._get_H0_11(p) + zeta * self.RYDBERG_PROJECTOR
        psi0_01, psi1_01 = self._evolve_state_discrete(M, T, ulist[:-1], self._get_H0_01, H1_ssr_01)
        psi0_11, psi1_11 = self._evolve_state_discrete(M, T, ulist[:-1], self._get_H0_11, H1_ssr_11)
        a01 = np.exp(-1.0j * theta) * psi0_01[0]
        a11 = np.exp(-2.0j * theta - 1.0j * np.pi) * psi0_11[0]
        F = (1 / 16) * np.abs(1 + 2 * a01 + a11) ** 2
        return (1 - F) + 2 * np.linalg.norm(psi1_01) ** 2 + np.linalg.norm(psi1_11) ** 2

    # --- ODE-based evolution ---
    def _evolve_state_ode(self, params, t_gate, H0_func, H1_func, initial_state):
        def schrodinger_rhs(t, y, params, H0_func, H1_func):
            phase = self._analytical_pulse_shape(t, *params)
            H0, H1 = H0_func(phase), H1_func(phase)
            H_block = np.block([[H0, np.zeros_like(H0)], [H1, H0]])
            diff = -1j * H_block
            y_arr = np.reshape(np.array(y), (-1, 1))
            return np.reshape(np.matmul(diff, y_arr), (-1))

        sol = integrate.solve_ivp(
            schrodinger_rhs,
            [0, t_gate],
            initial_state,
            args=(params, H0_func, H1_func),
            method="DOP853",
            rtol=1e-8,
            atol=1e-12,
        )
        return sol.y[:, -1]

    def _conticost_function_AR(self, analytical_params_and_theta, args):
        analytical_params = analytical_params_and_theta[:-1]
        theta = analytical_params_and_theta[-1]
        t_gate = args["gatetime"]

        initial_state = np.array([1, 0, 0, 0], dtype=np.complex128)

        final_state_01 = self._evolve_state_ode(analytical_params, t_gate, self._get_H0_01, self._get_H0_01, initial_state)
        psi0_01, psi1_01 = final_state_01[:2], final_state_01[2:]

        final_state_11 = self._evolve_state_ode(analytical_params, t_gate, self._get_H0_11, self._get_H0_11, initial_state)
        psi0_11, psi1_11 = final_state_11[:2], final_state_11[2:]

        a01 = np.exp(-1.0j * theta) * psi0_01[0]
        a11 = np.exp(-2.0j * theta - 1.0j * np.pi) * psi0_11[0]
        F = (1 / 16) * np.abs(1 + 2 * a01 + a11) ** 2

        return (1 - F) + 2 * np.linalg.norm(psi1_01) ** 2 + np.linalg.norm(psi1_11) ** 2

    # --- Public API ---
    def run_multistep_optimization(self, M_steps):
        """Run multi-step discrete pulse optimisation with increasing resolution."""
        print(f"\n{'=' * 20} Starting Multi-Step Optimization for {self.config['name']} Pulse {'=' * 20}")

        def _optimize_single_step(M, initial_guess=None):
            step_config = self.config.copy()
            step_config["M"] = M
            print(f"--- Running Optimization for M={M} ---")
            start_time = time.time()

            if initial_guess is None:
                initial_guess = np.random.rand(M + 1) * 2 * np.pi

            callback_with_args = partial(self._callback, args=step_config, start_time=start_time)
            optimres = minimize(
                fun=self.config["cost_func"],
                x0=initial_guess,
                args=step_config,
                method="L-BFGS-B",
                callback=callback_with_args,
                options={"maxiter": 1000, "ftol": 1e-12, "disp": True},
            )
            print(f"--- Optimization Finished in {time.time() - start_time:.1f}s ---")
            print(f"Final Cost (J): {optimres['fun']:.6f}\n")
            return {"solution_vector": optimres["x"], "M": M, "cost": optimres["fun"]}

        result = _optimize_single_step(M=M_steps[0])
        for M_next in M_steps[1:]:
            prev_sol, prev_M = result["solution_vector"], result["M"]
            prev_phases, prev_theta = prev_sol[:-1], prev_sol[-1]
            prev_x, next_x = np.linspace(0, 1, prev_M), np.linspace(0, 1, M_next)
            next_phases_guess = np.interp(next_x, prev_x, np.unwrap(prev_phases))
            next_initial_guess = np.append(next_phases_guess, prev_theta)
            result = _optimize_single_step(M=M_next, initial_guess=next_initial_guess)

        self.result = result
        self.result["name"] = self.config["name"]
        self.result["gate_time"] = self.config["gatetime"]
        return self.result

    def run_analytical_fit_and_optimization(self):
        """Fit discrete result to an analytical pulse shape and refine with ODE solver."""
        if self.result is None:
            raise RuntimeError("Run discrete optimization first.")

        phases_disc = self.result["solution_vector"][:-1]
        theta_disc = self.result["solution_vector"][-1]
        M, T = self.result["M"], self.result["gate_time"]
        tlist = np.linspace(0, T, M)
        phases_unwrapped = np.unwrap(phases_disc)

        try:
            popt, _ = curve_fit(self._analytical_pulse_shape, tlist, phases_unwrapped, maxfev=10000)
        except Exception:
            popt = np.ones(7)

        self.analytical_result = {"initial_fit_params": popt}

        analytical_guess = np.append(popt, theta_disc)

        res = minimize(
            fun=self._conticost_function_AR,
            x0=analytical_guess,
            args=self.config,
            method="Nelder-Mead",
            options={"maxiter": 500, "fatol": 1e-9, "disp": True},
        )

        self.analytical_result.update(
            {"final_params": res["x"][:-1], "theta": res["x"][-1], "cost": res["fun"], "name": self.config["name"]}
        )
        return self.analytical_result
