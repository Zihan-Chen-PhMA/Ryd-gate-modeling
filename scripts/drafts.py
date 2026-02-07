from ryd_gate.ideal_cz import CZGateSimulator

X_TO_OUR_BRIGHT = [
    -0.64168872, 1.14372811, 0.35715965, 1.51843443, 2.96448688, 1.21214853
]
N_SSS = 12
SSS_LABELS = [f"SSS-{i}" for i in range(N_SSS)]

sim_nodecay = CZGateSimulator(
    decayflag=False, param_set="our", strategy="TO",
        blackmanflag=False, detuning_sign=1,
    )
sim_decay = CZGateSimulator(
        decayflag=True, param_set="our", strategy="TO",
        blackmanflag=False, detuning_sign=1,
    )
print("Running without decay...")
inf_nodecay = sim_nodecay.avg_fidelity(X_TO_OUR_BRIGHT)  # Note: with decay
print("Running with decay...")
inf_withdecay = sim_decay.avg_fidelity(X_TO_OUR_BRIGHT)   # Without decay
print(f"Infidelity without decay: {inf_nodecay:.6f}")
print(f"Infidelity with decay: {inf_withdecay:.6f}")
print(f"Decay contribution: {inf_withdecay - inf_nodecay:.6f}")