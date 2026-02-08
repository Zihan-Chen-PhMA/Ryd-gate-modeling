# Find the optimal parameters for the bright detuning CZ gate
from ryd_gate.ideal_cz import CZGateSimulator

X_TO_OUR_BRIGHT = [
   -0.9509172186259588, 1.105272315809505, 0.383911389220584,
   1.2848721417313045, 1.3035218398648376, 1.246566016566724
]
sim_perfect = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_polarization_leakage=False,
    )
sim_perfect.optimize(x_initial=X_TO_OUR_BRIGHT, fid_type="average")
# 