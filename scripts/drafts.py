from ryd_gate.ideal_cz import CZGateSimulator

X_TO_OUR_BRIGHT = [
   -0.6918786926901699, 1.0385195543731935, 0.34079994362678945, 1.5661611471642423, 2.803412458711804, 1.3399024260140027
]


N_SSS = 12
SSS_LABELS = [f"SSS-{i}" for i in range(N_SSS)]

sim_nodecay = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_polarization_leakage=False,
    )
print("Running perfect gate...")
inf_perfect = sim_nodecay.gate_fidelity(X_TO_OUR_BRIGHT)
print(f"Infidelity of perfect gate: {inf_perfect:.6f}")
########################################################

sim_with_rydberg_decay = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=True, enable_intermediate_decay=False,
        enable_polarization_leakage=False,
    )

print("Running with decay...")
inf_with_rydberg_decay = sim_with_rydberg_decay.gate_fidelity(X_TO_OUR_BRIGHT)   # Without decay
print(f"Infidelity with Rydberg decay: {inf_with_rydberg_decay:.6f}")

########################################################

sim_with_intermediate_decay = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=False, enable_intermediate_decay=True,
        enable_polarization_leakage=False,
    )

print("Running with intermediate decay...")
inf_with_intermediate_decay = sim_with_intermediate_decay.gate_fidelity(X_TO_OUR_BRIGHT)   # Without decay
print(f"Infidelity with intermediate decay: {inf_with_intermediate_decay:.6f}")

########################################################

sim_with_polarization_leakage = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_polarization_leakage=True,
    )

print("Running with polarization leakage...")
inf_with_polarization_leakage = sim_with_polarization_leakage.gate_fidelity(X_TO_OUR_BRIGHT)   # Without decay
print(f"Infidelity with polarization leakage: {inf_with_polarization_leakage:.6f}")

########################################################

sim_with_all_errors = CZGateSimulator(
            param_set="our", strategy="TO",
            blackmanflag=True, detuning_sign=1,
            enable_rydberg_decay=True, enable_intermediate_decay=True,
            enable_polarization_leakage=True,
        )

print("Running with all errors...")
inf_with_all_errors = sim_with_all_errors.gate_fidelity(X_TO_OUR_BRIGHT)   # Without decay
print(f"Infidelity with all errors: {inf_with_all_errors:.6f}")
########################################################
