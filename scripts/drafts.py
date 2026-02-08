"""Error budget analysis: one-line API for each error source.

Demonstrates the CZGateSimulator API where each error source is
independently toggled via constructor flags. Monte Carlo errors
(dephasing, position) return (mean_infidelity, std_infidelity).
"""

from ryd_gate.ideal_cz import CZGateSimulator

X_TO_OUR_BRIGHT = [
   -0.9509172186259588, 1.105272315809505, 0.383911389220584,
   1.2848721417313045, 1.3035218398648376, 1.246566016566724
]

# ==================== Deterministic error sources ====================

sim_nodecay = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_polarization_leakage=False,
    )
print("Running perfect gate...")
inf_perfect = sim_nodecay.gate_fidelity(X_TO_OUR_BRIGHT)
print(f"Infidelity of perfect gate: {inf_perfect:.6f}")
print()

########################################################

sim_with_rydberg_decay = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=True, enable_intermediate_decay=False,
        enable_polarization_leakage=False,
    )

print("Running with Rydberg decay...")
inf_with_rydberg_decay = sim_with_rydberg_decay.gate_fidelity(X_TO_OUR_BRIGHT)
print(f"Infidelity with Rydberg decay: {inf_with_rydberg_decay:.6f}")
print()

########################################################

sim_with_intermediate_decay = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=False, enable_intermediate_decay=True,
        enable_polarization_leakage=False,
    )

print("Running with intermediate decay...")
inf_with_intermediate_decay = sim_with_intermediate_decay.gate_fidelity(X_TO_OUR_BRIGHT)
print(f"Infidelity with intermediate decay: {inf_with_intermediate_decay:.6f}")
print()

########################################################

sim_with_polarization_leakage = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_polarization_leakage=True,
    )

print("Running with polarization leakage...")
inf_with_polarization_leakage = sim_with_polarization_leakage.gate_fidelity(X_TO_OUR_BRIGHT)
print(f"Infidelity with polarization leakage: {inf_with_polarization_leakage:.6f}")
print()

########################################################

sim_with_zero_scattering = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_polarization_leakage=False,
        enable_zero_state_scattering=True,
    )

print("Running with zero-state scattering...")
inf_zero_scatter = sim_with_zero_scattering.gate_fidelity(X_TO_OUR_BRIGHT)
print(f"Infidelity with zero-state scattering: {inf_zero_scatter:.6f}")
print()

# ==================== Monte Carlo error sources ====================

N_MC = 1000

########################################################

sim_with_dephasing = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_dephasing=True,
        sigma_detuning=130e3,  # 170 kHz
        n_mc_shots=N_MC,
        mc_seed=42,
    )

print(f"Running with Rydberg dephasing (sigma_detuning=170 kHz, {N_MC} MC shots)...")
inf_dephasing = sim_with_dephasing.gate_fidelity(X_TO_OUR_BRIGHT)
print(f"Infidelity with dephasing: {inf_dephasing[0]:.6f} +/- {inf_dephasing[1]:.6f}")
print()

########################################################

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

########################################################

sim_with_all_deterministic = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=True, enable_intermediate_decay=True,
        enable_polarization_leakage=True,
        enable_zero_state_scattering=True,
    )

print("Running with all deterministic errors...")
inf_all_det = sim_with_all_deterministic.gate_fidelity(X_TO_OUR_BRIGHT)
print(f"Infidelity with all deterministic errors: {inf_all_det:.6f}")
print()

########################################################

sim_with_all_errors = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=True, enable_intermediate_decay=True,
        enable_polarization_leakage=True,
        enable_zero_state_scattering=True,
        enable_rydberg_dephasing=True,
        enable_position_error=True,
        sigma_detuning=170e3,  # 170 kHz
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
print("ERROR BUDGET SUMMARY")
print("=" * 65)
print(f"{'Error source':<30} {'Infidelity':>12} {'Contribution':>14}")
print("-" * 65)
print(f"{'Perfect gate':<30} {inf_perfect:>12.6f} {'(baseline)':>14}")
print(f"{'Rydberg decay':<30} {inf_with_rydberg_decay:>12.6f} {inf_with_rydberg_decay - inf_perfect:>+14.6f}")
print(f"{'Intermediate decay':<30} {inf_with_intermediate_decay:>12.6f} {inf_with_intermediate_decay - inf_perfect:>+14.6f}")
print(f"{'Polarization leakage':<30} {inf_with_polarization_leakage:>12.6f} {inf_with_polarization_leakage - inf_perfect:>+14.6f}")
print(f"{'Zero-state scattering':<30} {inf_zero_scatter:>12.6f} {inf_zero_scatter - inf_perfect:>+14.6f}")
print(f"{'Dephasing (170 kHz)':<30} {inf_dephasing[0]:>12.6f} {inf_dephasing[0] - inf_perfect:>+14.6f}")
print(f"{'Position (70,70,170 nm)':<30} {inf_position[0]:>12.6f} {inf_position[0] - inf_perfect:>+14.6f}")
print("-" * 65)
print(f"{'All deterministic':<30} {inf_all_det:>12.6f} {inf_all_det - inf_perfect:>+14.6f}")
print(f"{'ALL errors':<30} {inf_all[0]:>12.6f} {inf_all[0] - inf_perfect:>+14.6f}")
print("=" * 65)
