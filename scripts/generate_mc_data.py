"""Generate Monte Carlo simulation data for error budget tables.

Runs MC simulations for both bright and dark detuning with branching
decomposition, and saves results to data/ for reproducibility.

Usage:
    uv run python scripts/generate_mc_data.py [--n-shots N] [--seed S]
"""
import argparse
import os
os.environ["JAX_PLATFORMS"] = "cpu"

from pathlib import Path
from ryd_gate.ideal_cz import CZGateSimulator

X_TO_OUR_DARK = [
    -0.6989301339711643, 1.0296229082590798, 0.3759232324550267, 1.5710180991068543, 1.4454279613697887, 1.3406239758422793
]
X_TO_OUR_BRIGHT = [
0.6246672641243727, 1.2369507331752663, -0.470787497434612, 1.6547386752699043, 3.41960305947842, 1.3338111168065905
]

SIGMA_POS = (70e-9, 70e-9, 130e-9)  # meters
SIGMA_DETUNING = 130e3  # Hz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)

    for sign, label, x in [
        (1, "bright", X_TO_OUR_BRIGHT),
        # (1, "dark", X_TO_OUR_DARK),
    ]:
        print(f"\n{'='*70}")
        print(f"  {label.upper()} DETUNING (sign={sign})")
        print(f"{'='*70}")

        # 1. Dephasing only
        print(f"\n[1/3] Dephasing ({SIGMA_DETUNING/1e3:.0f} kHz)...")
        sim_deph = CZGateSimulator(
            param_set="our", strategy="TO",
            blackmanflag=True, detuning_sign=sign,
            enable_rydberg_dephasing=True,
            sigma_detuning=SIGMA_DETUNING,
        )
        result_deph = sim_deph.run_monte_carlo_simulation(
            x, n_shots=args.n_shots,
            sigma_detuning=SIGMA_DETUNING, seed=args.seed,
            compute_branching=True,
        )
        result_deph.save_to_file(f"data/mc_{label}_dephasing.txt")

        # 2. Position error only
        print(f"\n[2/3] Position error (70,70,130 nm)...")
        sim_pos = CZGateSimulator(
            param_set="our", strategy="TO",
            blackmanflag=True, detuning_sign=sign,
            enable_position_error=True,
            sigma_pos_xyz=SIGMA_POS,
        )
        result_pos = sim_pos.run_monte_carlo_simulation(
            x, n_shots=args.n_shots,
            sigma_pos_xyz=SIGMA_POS, seed=args.seed + 1,
            compute_branching=True,
        )
        result_pos.save_to_file(f"data/mc_{label}_position.txt")

        # 3. All errors combined
        print(f"\n[3/3] All errors combined...")
        sim_all = CZGateSimulator(
            param_set="our", strategy="TO",
            blackmanflag=True, detuning_sign=sign,
            enable_rydberg_decay=True,
            enable_intermediate_decay=True,
            enable_polarization_leakage=True,
            enable_rydberg_dephasing=True,
            enable_position_error=True,
            sigma_detuning=SIGMA_DETUNING,
            sigma_pos_xyz=SIGMA_POS,
        )
        result_all = sim_all.run_monte_carlo_simulation(
            x, n_shots=args.n_shots,
            sigma_detuning=SIGMA_DETUNING,
            sigma_pos_xyz=SIGMA_POS, seed=args.seed + 2,
            compute_branching=True,
        )
        result_all.save_to_file(f"data/mc_{label}_all.txt")

        print(f"\nSaved to data/mc_{label}_*.txt")


if __name__ == "__main__":
    main()
