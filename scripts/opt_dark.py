# Find the optimal parameters for the dark detuning CZ gate
from ryd_gate.ideal_cz import CZGateSimulator

X_TO_OUR_BRIGHT = [
    -0.6918786926901699, 1.0385195543731935, 0.34079994362678945, 1.5661611471642423, 2.803412458711804, 1.3399024260140027
]
sim_perfect = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=-1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_polarization_leakage=False,
    )
sim_perfect.optimize(x_initial=X_TO_OUR_BRIGHT, fid_type="average")
# 