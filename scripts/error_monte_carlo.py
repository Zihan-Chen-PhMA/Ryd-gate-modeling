"""Monte Carlo error sources with XYZ/AL/LG branching decomposition.

Usage:
    python scripts/error_monte_carlo.py [--load FILE] [--save FILE] [--n-mc N]
    uv run python scripts/error_monte_carlo.py --save results.json
"""

import argparse
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

from ryd_gate.ideal_cz import CZGateSimulator, MonteCarloResult

X_TO_OUR_DARK = [
   -0.9509172186259588, 1.105272315809505, 0.383911389220584,
   1.2848721417313045, 1.3035218398648376, 1.246566016566724
]


def print_branching_table(result: MonteCarloResult, label: str) -> None:
    """Print formatted branching breakdown table."""
    n = result.n_shots

    def sem(std):
        return std / np.sqrt(n)

    print(f"\n{'='*70}")
    print(f"  {label} ({n} shots)")
    print(f"{'='*70}")
    print(f"  {'Category':<20} {'Mean':>14} {'Std':>14} {'SEM':>14}")
    print(f"  {'-'*62}")
    print(f"  {'Total infidelity':<20} {result.mean_infidelity:>14.6e} "
          f"{result.std_infidelity:>14.6e} {sem(result.std_infidelity):>14.6e}")

    if result.mean_branch_XYZ is not None:
        print(f"  {'XYZ (Pauli)':<20} {result.mean_branch_XYZ:>14.6e} "
              f"{result.std_branch_XYZ:>14.6e} {sem(result.std_branch_XYZ):>14.6e}")
        print(f"  {'AL (Atom Loss)':<20} {result.mean_branch_AL:>14.6e} "
              f"{result.std_branch_AL:>14.6e} {sem(result.std_branch_AL):>14.6e}")
        print(f"  {'LG (Leakage)':<20} {result.mean_branch_LG:>14.6e} "
              f"{result.std_branch_LG:>14.6e} {sem(result.std_branch_LG):>14.6e}")
        print(f"  {'Phase error':<20} {result.mean_branch_phase:>14.6e} "
              f"{result.std_branch_phase:>14.6e} {sem(result.std_branch_phase):>14.6e}")
        branch_sum = (result.mean_branch_XYZ + result.mean_branch_AL
                     + result.mean_branch_LG + result.mean_branch_phase)
        print(f"  {'-'*62}")
        print(f"  {'Sum of branches':<20} {branch_sum:>14.6e}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="MC error budget with XYZ/AL/LG branching decomposition",
    )
    parser.add_argument("--load", type=str, help="Load results from file instead of running MC")
    parser.add_argument("--save", type=str, help="Save results to file after running")
    parser.add_argument("--n-mc", type=int, default=500, help="Number of MC shots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.load:
        result = MonteCarloResult.load_from_file(args.load)
        print_branching_table(result, f"Loaded from {args.load}")
        return

    # ==================== Dephasing with branching ====================

    sim_dephasing = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_dephasing=True,
        sigma_detuning=130e3,
        n_mc_shots=args.n_mc,
        mc_seed=args.seed,
    )

    print(f"Running dephasing MC with branching ({args.n_mc} shots)...")
    result_deph = sim_dephasing.run_monte_carlo_simulation(
        X_TO_OUR_DARK,
        n_shots=args.n_mc,
        sigma_detuning=130e3,
        seed=args.seed,
        compute_branching=True,
    )
    print_branching_table(result_deph, "Rydberg dephasing (130 kHz)")

    # ==================== Position error with branching ====================

    sigma_pos = (70e-9, 70e-9, 130e-9)  # meters

    sim_position = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_position_error=True,
        sigma_pos_xyz=sigma_pos,
        n_mc_shots=args.n_mc,
        mc_seed=args.seed,
    )

    print(f"\nRunning position MC with branching ({args.n_mc} shots)...")
    result_pos = sim_position.run_monte_carlo_simulation(
        X_TO_OUR_DARK,
        n_shots=args.n_mc,
        sigma_pos_xyz=sigma_pos,
        seed=args.seed + 1,
        compute_branching=True,
    )
    print_branching_table(result_pos, "Position error (70,70,130 nm)")

    # ==================== All errors with branching ====================

    sim_all = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=1,
        enable_rydberg_decay=True, enable_intermediate_decay=True,
        enable_polarization_leakage=True,
        enable_rydberg_dephasing=True,
        enable_position_error=True,
        sigma_detuning=130e3,
        sigma_pos_xyz=sigma_pos,
        n_mc_shots=args.n_mc,
        mc_seed=args.seed,
    )

    print(f"\nRunning ALL errors MC with branching ({args.n_mc} shots)...")
    result_all = sim_all.run_monte_carlo_simulation(
        X_TO_OUR_DARK,
        n_shots=args.n_mc,
        sigma_detuning=130e3,
        sigma_pos_xyz=sigma_pos,
        seed=args.seed + 2,
        compute_branching=True,
    )
    print_branching_table(result_all, "ALL errors combined")

    # ==================== Save ====================

    if args.save:
        result_deph.save_to_file(args.save + "_dephasing.txt")
        result_pos.save_to_file(args.save + "_position.txt")
        result_all.save_to_file(args.save + "_all.txt")
        print(f"\nResults saved to {args.save}_*.txt")


if __name__ == "__main__":
    main()
