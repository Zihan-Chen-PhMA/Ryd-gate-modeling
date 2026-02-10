# Find the optimal parameters for the dark detuning CZ gate
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
   -0.9509172186259588, 1.105272315809505, 0.383911389220584,
   1.2848721417313045, 1.3035218398648376, 1.246566016566724
]
sim_dark = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_polarization_leakage=False,
    ) 
sim_dark.optimize(x_initial=X_TO_OUR_DARK, fid_type="average")