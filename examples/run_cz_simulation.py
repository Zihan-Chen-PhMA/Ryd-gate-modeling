"""Example: run a CZ gate simulation using the Schrödinger solver.

This script demonstrates a basic fidelity calculation and diagnostics
using the CZGateSimulator from ideal_cz.py.
"""

from ryd_gate.ideal_cz import CZGateSimulator

# --- Time-Optimal (TO) strategy ---
sim = CZGateSimulator(param_set='our', strategy='TO')

# Known good TO pulse parameters: [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale]
X_TO = [0.1122, 1.0431, -0.72565603, 0.0, 0.452, 1.219096]

infidelity = sim.gate_fidelity(X_TO)
print(f"TO gate infidelity: {infidelity:.2e}")

# --- Amplitude-Robust (AR) strategy ---
sim_AR = CZGateSimulator(param_set='our', strategy='AR')

# Known good AR pulse parameters
X_AR = [0.85973359, 0.39146974, 0.99181418, 0.1924498,
        -1.17123748, -0.00826712, 1.67429728, 0.28527346]

infidelity_AR = sim_AR.gate_fidelity(X_AR)
print(f"AR gate infidelity: {infidelity_AR:.2e}")

# --- Population diagnostics ---
print("\nPopulation diagnostics for |11⟩ (TO):")
mid_pop, ryd_pop, ryd_garb_pop = sim.diagnose_run(X_TO, '11')
print(f"  Peak intermediate population: {mid_pop.max():.4f}")
print(f"  Peak Rydberg population:      {ryd_pop.max():.4f}")
