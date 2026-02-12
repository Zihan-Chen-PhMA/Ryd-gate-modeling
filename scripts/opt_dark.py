# Find the optimal parameters for the dark detuning CZ gate
from ryd_gate.ideal_cz import CZGateSimulator
# **Time-Optimal (TO) Strategy**

# Phase function: φ(t) = A·cos(ωt + φ₀) + δ·t
# H = e^{iφ(t)} |00⟩⟨00| + e^{-iφ(t)} |11⟩⟨11|
# Given H(t,a) and df/dt = -iH(t,a)*f,  try to find a' such that the ODE dg/dt = iH(t,a') g 's solution satisfied: if f(0) = g(0) then f(t) = g(t)
# Parameters x = [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale] where:
# - A: Cosine amplitude (radians)
# - ω: Modulation frequency
# - φ₀: Initial phase
# - δ: Linear chirp rate
# - θ: Single-qubit Z rotation angle
# - T: Gate time

X_TO_OUR_DARK = [
-0.6989301339711643, 1.0296229082590798, 0.3759232324550267, 1.5710180991068543, 1.4454279613697887, 1.3406239758422793
]


sim_dark = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_polarization_leakage=False,
    ) 

res = sim_dark.gate_fidelity(X_TO_OUR_DARK)
print(res)
sim_dark.optimize(x_initial=X_TO_OUR_DARK, fid_type="average")