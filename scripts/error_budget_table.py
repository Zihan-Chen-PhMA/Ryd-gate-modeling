#!/usr/bin/env python3
"""Simulated error budget table for CZ gate — Extended Data Fig. 4 reproduction.

Decomposes CZ gate errors by physical source and error type (XYZ/LG/AL),
following the methodology of Evered et al. (arXiv:2304.05420).

Error types:
  - XYZ: Decay/dephasing into computational qubit states |0⟩, |1⟩
  - LG:  Leakage to non-qubit mF states
  - AL:  Atom loss (BBR ionization from Rydberg)

Usage:
    uv run python scripts/error_budget_table.py [OPTIONS]

Options:
    --param-set {our,lukin,both}   Parameter set (default: both)
    --t2-star FLOAT                T2* in microseconds (default: 4.0)
    --mc-shots INT                 Monte Carlo shots for T2* (default: 1000)
    --optimize                     Run optimization for param sets without cached params
    --output-dir DIR               Output directory (default: docs)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ryd_gate.ideal_cz import CZGateSimulator


# ======================================================================
# Optimized TO pulse parameters (param_set='our')
# ======================================================================
X_TO_OUR_BRIGHT = [
    -0.64168872, 1.14372811, 0.35715965, 1.51843443, 2.96448688, 1.21214853
]
X_TO_OUR_DARK = [
    -0.62169911, -1.3591053, 0.50639069, -1.70318155, 1.17181594, 1.22294773
]

N_SSS = 12
SSS_LABELS = [f"SSS-{i}" for i in range(N_SSS)]


# ======================================================================
# Output formatting
# ======================================================================

def format_table(budget, t2_xyz, label):
    """Format error budget as a markdown table."""
    lines = []
    lines.append(f"\n### {label}")
    lines.append("")
    lines.append("| Error source            | XYZ (%)  | LG (%)   | AL (%)   | Total (%) |")
    lines.append("|-------------------------|----------|----------|----------|-----------|")

    # Rydberg T1
    lines.append(
        f"| Rydberg T1              "
        f"| {budget['ryd_T1_XYZ']*100:8.4f} "
        f"| {budget['ryd_T1_LG']*100:8.4f} "
        f"| {budget['ryd_T1_AL']*100:8.4f} "
        f"| {budget['ryd_T1_total']*100:9.4f} |"
    )

    # Intermediate scattering
    lines.append(
        f"| Intermediate scattering "
        f"| {budget['mid_XYZ']*100:8.4f} "
        f"| {budget['mid_LG']*100:8.4f} "
        f"|     -    "
        f"| {budget['mid_total']*100:9.4f} |"
    )

    # T2* dephasing
    lines.append(
        f"| T2* dephasing           "
        f"| {t2_xyz*100:8.4f} "
        f"|     -    "
        f"|     -    "
        f"| {t2_xyz*100:9.4f} |"
    )

    # Adjacent mJ coupling
    lines.append(
        f"| Adjacent mJ coupling    "
        f"|     -    "
        f"| {budget['mj_LG']*100:8.4f} "
        f"|     -    "
        f"| {budget['mj_total']*100:9.4f} |"
    )

    # Total
    total_XYZ = budget["ryd_T1_XYZ"] + budget["mid_XYZ"] + t2_xyz
    total_LG = budget["ryd_T1_LG"] + budget["mid_LG"] + budget["mj_LG"]
    total_AL = budget["ryd_T1_AL"]
    grand_total = total_XYZ + total_LG + total_AL
    lines.append(
        f"| **Total**               "
        f"| **{total_XYZ*100:.4f}** "
        f"| **{total_LG*100:.4f}** "
        f"| **{total_AL*100:.4f}** "
        f"| **{grand_total*100:.4f}** |"
    )

    return "\n".join(lines)


def format_cross_check(infidelity_nodecay, infidelity_decay, scattering_sum, label):
    """Format cross-check results."""
    decay_contribution = infidelity_decay - infidelity_nodecay
    lines = []
    lines.append(f"\n  Cross-check ({label}):")
    lines.append(f"    Infidelity (no decay):   {infidelity_nodecay:.6e}")
    lines.append(f"    Infidelity (with decay): {infidelity_decay:.6e}")
    lines.append(f"    Decay contribution:      {decay_contribution:.6e}")
    lines.append(f"    Scattering integral sum: {scattering_sum:.6e}")
    if decay_contribution > 0:
        ratio = scattering_sum / decay_contribution
        lines.append(f"    Ratio (integral/decay):  {ratio:.4f}")
    return "\n".join(lines)


# ======================================================================
# Main
# ======================================================================

def run_error_budget(param_set, detuning_sign, x, T2_star, mc_shots, label):
    """Run full error budget for one configuration.

    Uses CZGateSimulator methods directly: diagnose_run(), _decay_integrate(),
    avg_fidelity(), and run_monte_carlo_simulation().
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # --- Create simulators ---
    sim_decay = CZGateSimulator(
        decayflag=True, param_set=param_set, strategy="TO",
        blackmanflag=False, detuning_sign=detuning_sign,
    )
    sim_nodecay = CZGateSimulator(
        decayflag=False, param_set=param_set, strategy="TO",
        blackmanflag=False, detuning_sign=detuning_sign,
    )

    # --- Decay constants from simulator attributes ---
    Gamma_ryd = sim_decay.ryd_state_decay_rate
    Gamma_mid = sim_decay.mid_state_decay_rate
    Gamma_RD = 1 / sim_decay.atom.getStateLifetime(
        sim_decay.ryd_level, 0, 0.5, temperature=0
    )
    Gamma_BBR = Gamma_ryd - Gamma_RD

    print(f"  Gamma_ryd_total = {Gamma_ryd:.2f} /s "
          f"(tau = {1/Gamma_ryd*1e6:.2f} us)")
    print(f"  Gamma_RD        = {Gamma_RD:.2f} /s "
          f"(tau_RD = {1/Gamma_RD*1e6:.2f} us)")
    print(f"  Gamma_BBR       = {Gamma_BBR:.2f} /s")
    print(f"  Gamma_mid       = {Gamma_mid:.2e} /s "
          f"(tau = {1/Gamma_mid*1e9:.1f} ns)")

    # --- Scattering budget (averaged over 12 SSS states) ---
    print("\n  Computing scattering integrals (12 SSS states)...")
    t_gate = x[5] * sim_decay.time_scale
    t_list = np.linspace(0, t_gate, 1000)

    P_mid_total = 0.0
    P_ryd_total = 0.0
    P_mj_total = 0.0

    for sss_label in SSS_LABELS:
        mid_pop, ryd_pop, ryd_garb_pop = sim_decay.diagnose_run(x, sss_label)
        P_mid_total += sim_decay._decay_integrate(t_list, mid_pop, Gamma_mid)[0, -1]
        P_ryd_total += sim_decay._decay_integrate(t_list, ryd_pop, Gamma_ryd)[0, -1]
        P_mj_total += ryd_garb_pop[-1]

    P_mid_avg = P_mid_total / N_SSS
    P_ryd_avg = P_ryd_total / N_SSS
    P_mj_avg = P_mj_total / N_SSS

    # Decompose Rydberg T1: BBR → atom loss, radiative → XYZ/LG
    P_ryd_RD = (Gamma_RD / Gamma_ryd) * P_ryd_avg
    budget = {
        "ryd_T1_XYZ": (2 / 8) * P_ryd_RD,
        "ryd_T1_LG":  (6 / 8) * P_ryd_RD,
        "ryd_T1_AL":  (Gamma_BBR / Gamma_ryd) * P_ryd_avg,
        "ryd_T1_total": P_ryd_avg,
        "mid_XYZ": (2 / 8) * P_mid_avg,
        "mid_LG":  (6 / 8) * P_mid_avg,
        "mid_total": P_mid_avg,
        "mj_LG": P_mj_avg,
        "mj_total": P_mj_avg,
    }

    # --- T2* dephasing via Monte Carlo ---
    print(f"  Computing T2* dephasing (MC, {mc_shots} shots, T2*={T2_star*1e6:.1f} us)...")
    result_no_t2 = sim_nodecay.run_monte_carlo_simulation(
        x, n_shots=mc_shots, T2_star=None, temperature=None, seed=42,
    )
    result_with_t2 = sim_nodecay.run_monte_carlo_simulation(
        x, n_shots=mc_shots, T2_star=T2_star, temperature=None, seed=43,
    )
    t2_xyz = max(result_with_t2.mean_infidelity - result_no_t2.mean_infidelity, 0.0)

    # --- Cross-check: decay vs no-decay infidelity ---
    print("  Computing cross-check...")
    infidelity_decay = sim_decay.avg_fidelity(x)
    infidelity_nodecay = sim_nodecay.avg_fidelity(x)
    scattering_sum = P_mid_avg + P_ryd_avg

    # Format and print
    table = format_table(budget, t2_xyz, label)
    check = format_cross_check(infidelity_nodecay, infidelity_decay, scattering_sum, label)

    print(table)
    print(check)

    return {"budget": budget, "t2_xyz": t2_xyz, "table": table}


def main():
    parser = argparse.ArgumentParser(
        description="Simulated error budget table for CZ gate"
    )
    parser.add_argument(
        "--param-set", choices=["our", "lukin", "both"], default="both",
        help="Parameter set (default: both)",
    )
    parser.add_argument(
        "--t2-star", type=float, default=4.0,
        help="T2* in microseconds (default: 4.0)",
    )
    parser.add_argument(
        "--mc-shots", type=int, default=1000,
        help="Monte Carlo shots for T2* (default: 1000)",
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Run optimization for lukin params (required if no cached params)",
    )
    parser.add_argument(
        "--output-dir", default="docs",
        help="Output directory (default: docs)",
    )
    args = parser.parse_args()

    T2_star = args.t2_star * 1e-6  # Convert to seconds

    param_sets = ["our", "lukin"] if args.param_set == "both" else [args.param_set]
    results = {}

    for ps in param_sets:
        for sign, sign_label in [(1, "bright"), (-1, "dark")]:
            label = f"{ps} / {sign_label} detuning"

            # Get optimized parameters
            if ps == "our":
                x = X_TO_OUR_BRIGHT if sign == 1 else X_TO_OUR_DARK
            elif ps == "lukin":
                if args.optimize:
                    print(f"\n  Optimizing TO protocol for lukin/{sign_label}...")
                    sim_opt = CZGateSimulator(
                        decayflag=False, param_set=ps, strategy="TO",
                        blackmanflag=False, detuning_sign=sign,
                    )
                    x0 = X_TO_OUR_BRIGHT if sign == 1 else X_TO_OUR_DARK
                    x = list(sim_opt.optimize(x0).x)
                    print(f"  Optimized params: {x}")
                else:
                    print(f"\n  WARNING: No cached lukin params for {sign_label}.")
                    print("  Run with --optimize to generate them, or use --param-set our.")
                    continue

            result = run_error_budget(ps, sign, x, T2_star, args.mc_shots, label)
            results[(ps, sign_label)] = result

    # Save combined output
    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, "error_budget_table.md")
    with open(outpath, "w") as f:
        f.write("# CZ Gate Simulated Error Budget\n\n")
        f.write(f"T2* = {args.t2_star} μs, MC shots = {args.mc_shots}\n")
        for key, res in results.items():
            f.write(res["table"])
            f.write("\n\n")
    print(f"\n  Saved table to {outpath}")


if __name__ == "__main__":
    main()
