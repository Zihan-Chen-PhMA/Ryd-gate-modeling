"""Deterministic error budget: each error source toggled independently.

Computes SSS-averaged infidelity for bright/dark detuning with individual
error sources and XYZ/AL/LG branching decomposition:
- Perfect gate (baseline)
- Rydberg decay + branching
- Intermediate decay + branching (with |0⟩ vs |1⟩ scattering decomposition)
- Polarization leakage
- All deterministic combined + branching
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

from ryd_gate.ideal_cz import CZGateSimulator

SSS_12_STATES = [f"SSS-{i}" for i in range(12)]

# **Time-Optimal (TO) Strategy**

# Phase function: φ(t) = A·cos(ωt + φ₀) + δ·t

# Parameters x = [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale] where:
# - A: Cosine amplitude (radians)
# - ω: Modulation frequency
# - φ₀: Initial phase
# - δ: Linear chirp rate
# - θ: Single-qubit Z rotation angle
# - T: Gate time

X_TO_OUR_DARK = [
-0.6989301339711643, 1.0296229082590798, 0.3759232324550267, 1.5710180991068543, 1.4454279613697887, 1.3406239758422793
]
X_TO_OUR_BRIGHT = [
0.6246672641243727, 1.2369507331752663, -0.470787497434612, 1.6547386752699043, 3.41960305947842, 1.3338111168065905
]


def run_error_source(label, detuning_sign, x, **sim_kwargs):
    """Run a single error source: compute SSS infidelity and error_budget.

    Returns (sss_infidelity, budget_dict_or_None).
    """
    sim = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=detuning_sign,
        **sim_kwargs,
    )
    print(f"  Running {label}...")
    infid = sim.gate_fidelity(x, fid_type="sss")
    print(f"    Infidelity: {infid:.6e}")

    # error_budget meaningful when any decay/leakage is enabled
    has_decay = sim_kwargs.get("enable_rydberg_decay", False) or \
                sim_kwargs.get("enable_intermediate_decay", False) or \
                sim_kwargs.get("enable_polarization_leakage", False)
    budget = None
    if has_decay:
        budget = sim.error_budget(x, initial_states=SSS_12_STATES)
        for source, vals in budget.items():
            print(f"    {source}: total={vals['total']:.6e}  "
                  f"XYZ={vals['XYZ']:.6e}  AL={vals['AL']:.6e}  LG={vals['LG']:.6e}")

    return infid, budget


def main():
    results = {}

    for sign, sign_label, x in [
        # (-1, "bright", X_TO_OUR_BRIGHT),
        (1, "dark", X_TO_OUR_DARK),
                                  ]:
        print(f"\n{'='*70}")
        print(f"  Detuning: {sign_label}")
        print(f"{'='*70}")

        # ==================== Perfect gate ====================
        inf_perfect, _ = run_error_source(
            f"perfect gate ({sign_label})", sign, x,
            enable_rydberg_decay=False, enable_intermediate_decay=False,
            enable_polarization_leakage=False,
        )

        # ==================== Rydberg decay ====================
        inf_ryd, budget_ryd = run_error_source(
            f"Rydberg decay ({sign_label})", sign, x,
            enable_rydberg_decay=True, enable_intermediate_decay=False,
            enable_polarization_leakage=False,
        )

        # ==================== Intermediate decay ====================
        inf_mid, budget_mid = run_error_source(
            f"intermediate decay ({sign_label})", sign, x,
            enable_rydberg_decay=False, enable_intermediate_decay=True,
            enable_polarization_leakage=False,
        )

        # Scattering decomposition (|0⟩ vs |1⟩)
        inf_mid_no_scat, budget_mid_no_scat = run_error_source(
            f"intermediate decay no |0⟩ scattering ({sign_label})", sign, x,
            enable_rydberg_decay=False, enable_intermediate_decay=True,
            enable_0_scattering=False,
            enable_polarization_leakage=False,
        )

        # ==================== Polarization leakage ====================
        inf_pol, budget_pol = run_error_source(
            f"polarization leakage ({sign_label})", sign, x,
            enable_rydberg_decay=False, enable_intermediate_decay=False,
            enable_polarization_leakage=True,
        )

        # ==================== All deterministic ====================
        inf_all, budget_all = run_error_source(
            f"all deterministic ({sign_label})", sign, x,
            enable_rydberg_decay=True, enable_intermediate_decay=True,
            enable_polarization_leakage=True,
        )

        results[sign_label] = {
            "perfect": inf_perfect,
            "rydberg": inf_ryd,
            "budget_ryd": budget_ryd,
            "intermediate": inf_mid,
            "budget_mid": budget_mid,
            "mid_no_scat": inf_mid_no_scat,
            "budget_mid_no_scat": budget_mid_no_scat,
            "polarization": inf_pol,
            "budget_pol": budget_pol,
            "all_det": inf_all,
            "budget_all": budget_all,
        }

    # ==================== Summary ====================
    w = 100
    print(f"\n{'='*w}")
    print("DETERMINISTIC ERROR BUDGET SUMMARY")
    print(f"{'='*w}")
    print(f"{'Error source':<35} {'Infidelity':>12} {'Contrib.':>12}"
          f" {'XYZ':>12} {'AL':>12} {'LG':>12}")
    print(f"{'-'*w}")

    for sl in results:
        r = results[sl]
        baseline = r["perfect"]

        print(f"{'Perfect gate (' + sl + ')':<35} {baseline:>12.6e} {'(baseline)':>12}")

        # Rydberg decay
        contrib = r["rydberg"] - baseline
        b = r["budget_ryd"]["rydberg_decay"]
        print(f"{'Rydberg decay (' + sl + ')':<35} {r['rydberg']:>12.6e} {contrib:>+12.6e}"
              f" {b['XYZ']:>12.6e} {b['AL']:>12.6e} {b['LG']:>12.6e}")

        # Intermediate decay (full 0+1 scattering)
        contrib = r["intermediate"] - baseline
        b = r["budget_mid"]["intermediate_decay"]
        print(f"{'Intermediate decay (' + sl + ')':<35} {r['intermediate']:>12.6e} {contrib:>+12.6e}"
              f" {b['XYZ']:>12.6e} {b['AL']:>12.6e} {b['LG']:>12.6e}")
        # |0⟩ contribution: extra infidelity only (no valid XYZ/AL/LG decomposition)
        scat_0 = r["intermediate"] - r["mid_no_scat"]
        print(f"{'  (|0> contrib.)':<35} {scat_0:>12.6e}")

        # Polarization leakage
        contrib = r["polarization"] - baseline
        b_pol = r["budget_pol"]["polarization_leakage"]
        print(f"{'Polarization leak (' + sl + ')':<35} {r['polarization']:>12.6e} {contrib:>+12.6e}"
              f" {b_pol['XYZ']:>12.6e} {b_pol['AL']:>12.6e} {b_pol['LG']:>12.6e}")

        # All deterministic
        contrib = r["all_det"] - baseline
        ba = r["budget_all"]
        b_ryd = ba["rydberg_decay"]
        b_mid_a = ba["intermediate_decay"]
        b_pol_a = ba["polarization_leakage"]
        xyz_total = b_ryd["XYZ"] + b_mid_a["XYZ"] + b_pol_a["XYZ"]
        al_total = b_ryd["AL"] + b_mid_a["AL"] + b_pol_a["AL"]
        lg_total = b_ryd["LG"] + b_mid_a["LG"] + b_pol_a["LG"]
        print(f"{'All deterministic (' + sl + ')':<35} {r['all_det']:>12.6e} {contrib:>+12.6e}"
              f" {xyz_total:>12.6e} {al_total:>12.6e} {lg_total:>12.6e}")

        print(f"{'-'*w}")

    print(f"{'='*w}")


if __name__ == "__main__":
    main()
