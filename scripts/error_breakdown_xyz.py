"""Error type breakdown (XYZ / AL / LG) using branching ratios.

Computes the error budget decomposed into Pauli (XYZ), atom loss (AL),
and leakage (LG) channels for bright detuning.
"""

from ryd_gate.ideal_cz import CZGateSimulator

X_TO_OUR_BRIGHT = [
   -0.9509172186259588, 1.105272315809505, 0.383911389220584,
   1.2848721417313045, 1.3035218398648376, 1.246566016566724
]

SSS_LABELS = [f"SSS-{i}" for i in range(12)]

# ==================== Decay error budget ====================

sim_budget = CZGateSimulator(
    param_set="our", strategy="TO",
    blackmanflag=True, detuning_sign=1,
    enable_rydberg_decay=True, enable_intermediate_decay=True,
    enable_polarization_leakage=False,
)

print("Computing error budget with branching ratios (12 SSS states)...")
budget = sim_budget.error_budget(X_TO_OUR_BRIGHT, initial_states=SSS_LABELS)

# ==================== Non-decay contributions ====================

# Polarization leakage → all LG
sim_perfect = CZGateSimulator(
    param_set="our", strategy="TO",
    blackmanflag=True, detuning_sign=1,
)
inf_perfect = sim_perfect.gate_fidelity(X_TO_OUR_BRIGHT)

sim_pol = CZGateSimulator(
    param_set="our", strategy="TO",
    blackmanflag=True, detuning_sign=1,
    enable_polarization_leakage=True,
)
inf_pol = sim_pol.gate_fidelity(X_TO_OUR_BRIGHT)

pol_contribution = inf_pol - inf_perfect
budget["polarization_leakage"] = {
    "total": pol_contribution,
    "XYZ": 0.0,
    "AL": 0.0,
    "LG": pol_contribution,
}

# ==================== Print ====================

print(f"\n{'Error source':<25} {'Total':>10} {'XYZ':>10} {'AL':>10} {'LG':>10}")
print("-" * 80)
total_xyz = 0.0
total_al = 0.0
total_lg = 0.0
for source, errors in budget.items():
    print(
        f"{source:<25} {errors['total']:>10.2e} "
        f"{errors['XYZ']:>10.2e} {errors['AL']:>10.2e} {errors['LG']:>10.2e}"
    )
    total_xyz += errors["XYZ"]
    total_al += errors["AL"]
    total_lg += errors["LG"]
grand_total = total_xyz + total_al + total_lg
print("-" * 80)
print(
    f"{'TOTAL':<25} {grand_total:>10.2e} "
    f"{total_xyz:>10.2e} {total_al:>10.2e} {total_lg:>10.2e}"
)
print("=" * 80)
