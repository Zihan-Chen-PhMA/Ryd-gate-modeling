# This script plots the population of the intermediate states over time for the TO strategy.

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ryd_gate.ideal_cz import CZGateSimulator

# **Time-Optimal (TO) Strategy**

# Phase function: φ(t) = A·cos(ωt + φ₀) + δ·t

# Parameters x = [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale] where:
# - A: Cosine amplitude (radians)
# - ω: Modulation frequency
# - φ₀: Initial phase
# - δ: Linear chirp rate
# - θ: Single-qubit Z rotation angle
# - T: Gate time

X_TO_OUR_DARK = [
    -0.6990251940088914, 1.0294930712188455, 0.37642793463018853, 1.5710847832834478, 1.4454415553284314, 1.340639491094446
]
X_TO_OUR_BRIGHT =[
0.6246672641243727, 1.2369507331752663, -0.470787497434612, 1.6547386752699043, 3.41960305947842, 1.3338111168065905
]  

sim = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_0_scattering=False,
        enable_polarization_leakage=False,
    )   

INITIAL_STATE = "11"

mid_bright, _ryd_bright, _rg_bright = sim.diagnose_run(
    X_TO_OUR_BRIGHT, initial_state=INITIAL_STATE
)
mid_dark, _ryd_dark, _rg_dark = sim.diagnose_run(
    X_TO_OUR_DARK, initial_state=INITIAL_STATE
)

t_gate_bright = X_TO_OUR_BRIGHT[5] * sim.time_scale
t_gate_dark = X_TO_OUR_DARK[5] * sim.time_scale
time_bright_ns = np.linspace(0, t_gate_bright * 1e9, len(mid_bright))
time_dark_ns = np.linspace(0, t_gate_dark * 1e9, len(mid_dark))

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_dark_ns, mid_dark, label="Dark detuning")
ax.plot(time_bright_ns, mid_bright, label="Bright detuning")
ax.set_xlabel("Gate time (ns)")
ax.set_ylabel("Intermediate state population")
ax.set_title(f"Mid-state population vs time (initial |{INITIAL_STATE}⟩)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_path = "mid_state_population_bright_vs_dark.png"
fig.savefig(out_path, dpi=200)
print(f"Saved plot to {out_path}")
