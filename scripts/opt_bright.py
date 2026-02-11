# Find the optimal parameters for the bright detuning CZ gate
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

X_TO_OUR_BRIGHT =[
-0.6989301339711643 - 3.1415926535, 1.0296229082590798, 0.3759232324550267, 1.5710180991068543, 1.4454279613697887, 1.3406239758422793
]   
# [-0.57427882  1.02548636  0.37090274  1.40236909  3.58690842  1.31468954] 
# [1.9792338119200394, 0.8805904859802784, -0.38840789469712256, -1.040280411129939, 3.275909805658351, 1.3583544348458325]
sim_perfect = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=-1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_polarization_leakage=False,
    )
res = sim_perfect.gate_fidelity(X_TO_OUR_BRIGHT)
print(res)
sim_perfect.optimize(x_initial=X_TO_OUR_BRIGHT, fid_type="average")
# 