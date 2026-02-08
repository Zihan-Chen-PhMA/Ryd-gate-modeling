# Find the optimal parameters for the dark detuning CZ gate
from ryd_gate.ideal_cz import CZGateSimulator

X_TO_OUR_BRIGHT = [
   -1.60692534, 0.83165467, 2.22419665, 0.46423769, 0.90466714, 1.24153599
]
# [-0.57427882  1.02548636  0.37090274  1.40236909  3.58690842  1.31468954]
sim_perfect = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=-1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_polarization_leakage=False,
    )
sim_perfect.optimize(x_initial=X_TO_OUR_BRIGHT, fid_type="average")
# 