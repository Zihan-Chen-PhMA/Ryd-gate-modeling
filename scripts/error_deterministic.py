"""Deterministic error budget: each error source toggled independently.

Computes infidelity for bright/dark detuning with individual error sources:
- Perfect gate (baseline)
- Rydberg decay
- Intermediate decay (with |0⟩ vs |1⟩ scattering decomposition)
- Polarization leakage
- All deterministic combined
"""

from ryd_gate.ideal_cz import CZGateSimulator

X_TO_OUR_BRIGHT = [
   -0.9509172186259588, 1.105272315809505, 0.383911389220584,
   1.2848721417313045, 1.3035218398648376, 1.246566016566724
]
X_TO_OUR_DARK = [
   -1.7370398295694707, 0.7988774460188806, 2.3116588890406224, 0.5186261498956248, 0.900066116155231, 1.2415235064066774
]

# ==================== Perfect gate ====================

sim_nodecay_bright = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_polarization_leakage=False,
    )
print("Running perfect gate (bright)...")
inf_perfect_bright = sim_nodecay_bright.gate_fidelity(X_TO_OUR_BRIGHT)
print(f"Infidelity of perfect gate (bright): {inf_perfect_bright:.6f}")


sim_nodecay_dark = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=-1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_polarization_leakage=False,
    )
print("Running perfect gate (dark)...")
inf_perfect_dark = sim_nodecay_dark.gate_fidelity(X_TO_OUR_DARK)
print(f"Infidelity of perfect gate (dark): {inf_perfect_dark:.6f}")

# ==================== Rydberg decay ====================

sim_with_rydberg_decay_bright = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=True, enable_intermediate_decay=False,
        enable_polarization_leakage=False,
    )

print("Running with Rydberg decay (bright)...")
inf_with_rydberg_decay_bright = sim_with_rydberg_decay_bright.gate_fidelity(X_TO_OUR_BRIGHT)
print(f"Infidelity with Rydberg decay: {inf_with_rydberg_decay_bright:.6f}")

sim_with_rydberg_decay_dark = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=-1,
        enable_rydberg_decay=True, enable_intermediate_decay=False,
        enable_polarization_leakage=False,
    )

print("Running with Rydberg decay (dark)...")
inf_with_rydberg_decay_dark = sim_with_rydberg_decay_dark.gate_fidelity(X_TO_OUR_DARK)
print(f"Infidelity with Rydberg decay (dark): {inf_with_rydberg_decay_dark:.6f}")

# ==================== Intermediate decay + scattering decomposition ====================

sim_with_intermediate_decay_bright = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=False, enable_intermediate_decay=True,
        enable_polarization_leakage=False,
    )

print("Running with intermediate decay (bright)...")
inf_with_intermediate_decay_bright = sim_with_intermediate_decay_bright.gate_fidelity(X_TO_OUR_BRIGHT)
print(f"Infidelity with intermediate decay (bright): {inf_with_intermediate_decay_bright:.6f}")

# Scattering decomposition: |0⟩ vs |1⟩ contributions
print("Computing scattering decomposition (|0⟩ vs |1⟩)...")
budget_00_bright = sim_with_intermediate_decay_bright.error_budget(X_TO_OUR_BRIGHT, initial_states=["00"])
budget_11_bright = sim_with_intermediate_decay_bright.error_budget(X_TO_OUR_BRIGHT, initial_states=["11"])
inf_scatter_0_bright = budget_00_bright["intermediate_decay"]["total"]
inf_scatter_1_bright = budget_11_bright["intermediate_decay"]["total"]
print(f"Scattering* |0⟩ (from |00⟩): {inf_scatter_0_bright:.6e}")
print(f"Scattering |1⟩ (from |11⟩):  {inf_scatter_1_bright:.6e}")


sim_with_intermediate_decay_dark = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=-1,
        enable_rydberg_decay=False, enable_intermediate_decay=True,
        enable_polarization_leakage=False,
    )

print("Running with intermediate decay (dark)...")
inf_with_intermediate_decay_dark = sim_with_intermediate_decay_dark.gate_fidelity(X_TO_OUR_DARK)
print(f"Infidelity with intermediate decay (dark): {inf_with_intermediate_decay_dark:.6f}")

print("Computing scattering decomposition (|0⟩ vs |1⟩)...")
budget_00_dark = sim_with_intermediate_decay_dark.error_budget(X_TO_OUR_DARK, initial_states=["00"])
budget_11_dark = sim_with_intermediate_decay_dark.error_budget(X_TO_OUR_DARK, initial_states=["11"])
inf_scatter_0_dark = budget_00_dark["intermediate_decay"]["total"]
inf_scatter_1_dark = budget_11_dark["intermediate_decay"]["total"]
print(f"Scattering* |0⟩ (from |00⟩): {inf_scatter_0_dark:.6e}")
print(f"Scattering |1⟩ (from |11⟩):  {inf_scatter_1_dark:.6e}")

# ==================== Polarization leakage ====================

sim_with_polarization_leakage_bright = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_polarization_leakage=True,
    )

print("Running with polarization leakage (bright)...")
inf_with_polarization_leakage_bright = sim_with_polarization_leakage_bright.gate_fidelity(X_TO_OUR_BRIGHT)
print(f"Infidelity with polarization leakage (bright): {inf_with_polarization_leakage_bright:.6f}")

sim_with_polarization_leakage_dark = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=-1,
        enable_rydberg_decay=False, enable_intermediate_decay=False,
        enable_polarization_leakage=True,
    )

print("Running with polarization leakage (dark)...")
inf_with_polarization_leakage_dark = sim_with_polarization_leakage_dark.gate_fidelity(X_TO_OUR_DARK)
print(f"Infidelity with polarization leakage (dark): {inf_with_polarization_leakage_dark:.6f}")

# ==================== All deterministic ====================

sim_with_all_deterministic = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=True, enable_intermediate_decay=True,
        enable_polarization_leakage=True,
    )

print("Running with all deterministic errors...")
inf_all_det = sim_with_all_deterministic.gate_fidelity(X_TO_OUR_BRIGHT)
print(f"Infidelity with all deterministic errors: {inf_all_det:.6f}")
print()

# ==================== Summary ====================

print("=" * 65)
print("DETERMINISTIC ERROR BUDGET SUMMARY")
print("=" * 65)
print(f"{'Error source':<30} {'Infidelity':>12} {'Contribution':>14}")
print("-" * 65)
print(f"{'Perfect gate (bright)':<30} {inf_perfect_bright:>12.6f} {'(baseline)':>14}")
print(f"{'Perfect gate (dark)':<30} {inf_perfect_dark:>12.6f} {'(baseline)':>14}")
print(f"{'Rydberg decay (bright)':<30} {inf_with_rydberg_decay_bright:>12.6f} {inf_with_rydberg_decay_bright - inf_perfect_bright:>+14.6f}")
print(f"{'Rydberg decay (dark)':<30} {inf_with_rydberg_decay_dark:>12.6f} {inf_with_rydberg_decay_dark - inf_perfect_dark:>+14.6f}")
print(f"{'Intermediate decay (bright)':<30} {inf_with_intermediate_decay_bright:>12.6f} {inf_with_intermediate_decay_bright - inf_perfect_bright:>+14.6f}")
print(f"{'Intermediate decay (dark)':<30} {inf_with_intermediate_decay_dark:>12.6f} {inf_with_intermediate_decay_dark - inf_perfect_dark:>+14.6f}")
print(f"{'  Scattering* |0⟩ bright':<30} {inf_scatter_0_bright:>12.6e}")
print(f"{'  Scattering |1⟩ bright':<30} {inf_scatter_1_bright:>12.6e}")
print(f"{'  Scattering* |0⟩ dark':<30} {inf_scatter_0_dark:>12.6e}")
print(f"{'  Scattering |1⟩ dark':<30} {inf_scatter_1_dark:>12.6e}")
print(f"{'Polarization leak (bright)':<30} {inf_with_polarization_leakage_bright:>12.6f} {inf_with_polarization_leakage_bright - inf_perfect_bright:>+14.6f}")
print(f"{'Polarization leak (dark)':<30} {inf_with_polarization_leakage_dark:>12.6f} {inf_with_polarization_leakage_dark - inf_perfect_dark:>+14.6f}")
print("-" * 65)
print(f"{'All deterministic (bright)':<30} {inf_all_det:>12.6f} {inf_all_det - inf_perfect_bright:>+14.6f}")
print("=" * 65)
