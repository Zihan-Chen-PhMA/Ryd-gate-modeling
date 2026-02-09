"""Monte Carlo error sources: dephasing, position error, and all-errors combined.

Requires long runtime due to MC sampling. Each MC error source returns
(mean_infidelity, std_infidelity).
"""

from ryd_gate.ideal_cz import CZGateSimulator

X_TO_OUR_BRIGHT = [
   -0.9509172186259588, 1.105272315809505, 0.383911389220584,
   1.2848721417313045, 1.3035218398648376, 1.246566016566724
]

N_MC = 1000

# ==================== Rydberg dephasing ====================

sim_with_dephasing = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_dephasing=True,
        sigma_detuning=130e3,  # 130 kHz
        n_mc_shots=N_MC,
        mc_seed=42,
    )

print(f"Running with Rydberg dephasing (sigma_detuning=130 kHz, {N_MC} MC shots)...")
inf_dephasing = sim_with_dephasing.gate_fidelity(X_TO_OUR_BRIGHT)
print(f"Infidelity with dephasing: {inf_dephasing[0]:.6f} +/- {inf_dephasing[1]:.6f}")
print()

# ==================== Position error ====================

# NOTE: sigma_pos_xyz in meters. (70e-9, 70e-9, 170e-9) = (70, 70, 170) nm
# Typical position spread at 15 uK / 50 kHz trap is ~120 nm.
sigma_pos = (70e-9, 70e-9, 170e-9)  # meters

sim_with_position = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_position_error=True,
        sigma_pos_xyz=sigma_pos,
        n_mc_shots=N_MC,
        mc_seed=42,
    )

print(f"Running with position error (sigma_xyz=(70,70,170) nm, {N_MC} MC shots)...")
inf_position = sim_with_position.gate_fidelity(X_TO_OUR_BRIGHT)
print(f"Infidelity with position error: {inf_position[0]:.6f} +/- {inf_position[1]:.6f}")
print()

# ==================== All errors combined ====================

sim_with_all_errors = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=True, enable_intermediate_decay=True,
        enable_polarization_leakage=True,
        enable_rydberg_dephasing=True,
        enable_position_error=True,
        sigma_detuning=130e3,  # 130 kHz
        sigma_pos_xyz=sigma_pos,
        n_mc_shots=N_MC,
        mc_seed=42,
    )

print(f"Running with ALL errors (deterministic + MC, {N_MC} shots)...")
inf_all = sim_with_all_errors.gate_fidelity(X_TO_OUR_BRIGHT)
print(f"Infidelity with ALL errors: {inf_all[0]:.6f} +/- {inf_all[1]:.6f}")
print()

# ==================== Summary ====================

print("=" * 65)
print("MONTE CARLO ERROR BUDGET")
print("=" * 65)
print(f"{'Dephasing (130 kHz)':<30} {inf_dephasing[0]:>12.6f} +/- {inf_dephasing[1]:.6f}")
print(f"{'Position (70,70,170 nm)':<30} {inf_position[0]:>12.6f} +/- {inf_position[1]:.6f}")
print(f"{'ALL errors':<30} {inf_all[0]:>12.6f} +/- {inf_all[1]:.6f}")
print("=" * 65)
