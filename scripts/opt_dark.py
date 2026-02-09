# Find the optimal parameters for the dark detuning CZ gate
from ryd_gate.ideal_cz import CZGateSimulator

X_TO_OUR_DARK = [
   -1.7370398295694707, 0.7988774460188806, 2.3116588890406224, 0.5186261498956248, 0.900066116155231, 1.2415235064066774
]   
# [-0.57427882  1.02548636  0.37090274  1.40236909  3.58690842  1.31468954]
sim_dark = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=-1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_polarization_leakage=False,
    ) 
sim_dark.optimize(x_initial=X_TO_OUR_DARK, fid_type="average")