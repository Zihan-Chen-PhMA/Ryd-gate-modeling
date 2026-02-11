from ryd_gate.ideal_cz import CZGateSimulator

X_TO_OUR_DARK = [
   -0.9509172186259588, 1.105272315809505, 0.383911389220584,
   1.2848721417313045, 1.3035218398648376, 1.246566016566724
]
sim = CZGateSimulator(
    param_set="our", strategy="TO",
    blackmanflag=True, detuning_sign=+1
)
res2 = sim.gate_fidelity(X_TO_OUR_DARK, fid_type="average")
print(res2)

X_TO_OUR_DARK2 = [
   -0.9509172186259588, 1.105272315809505, 0.783911389220584,
   1.2848721417313045, 1.3035218398648376, 1.246566016566724
]
sim2 = CZGateSimulator(
    param_set="our", strategy="TO",
    blackmanflag=True, detuning_sign=+1
)
res2 = sim2.gate_fidelity(X_TO_OUR_DARK2, fid_type="average")
print(res2)