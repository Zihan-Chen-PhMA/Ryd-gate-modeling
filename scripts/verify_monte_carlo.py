#!/usr/bin/env python3
"""Verify Monte Carlo simulation by plotting fidelity vs temperature effects.

This script generates plots showing how gate fidelity changes with:
1. T2* coherence time (Doppler dephasing)
2. Atomic temperature (position spread)
3. Combined effects

Results are saved as PNG images for verification of the Monte Carlo implementation.

Uses multiprocessing to run independent MC simulations in parallel across CPU cores.
"""

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Optimized TO pulse parameters (from test_cz_gate_phase.py)
X_TO = [-0.64168872, 1.14372811, 0.35715965, 1.51843443, 2.96448688, 1.21214853]

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-worker simulator — set in main() before pool creation.
# On Linux (fork), child processes inherit this via copy-on-write.
_worker_sim = None


def _run_mc_worker(kwargs):
    """Worker function: run one MC simulation using the inherited simulator.

    Each forked child has its own copy of _worker_sim, so mutations
    in run_monte_carlo_simulation don't affect other workers.

    Returns a dict with mean_fidelity, std_fidelity, mean_infidelity, std_infidelity.
    """
    result = _worker_sim.run_monte_carlo_simulation(X_TO, **kwargs)
    return {
        'mean_fidelity': result.mean_fidelity,
        'std_fidelity': result.std_fidelity,
        'mean_infidelity': result.mean_infidelity,
        'std_infidelity': result.std_infidelity,
    }


def plot_fidelity_vs_t2_star(results_by_t2, ideal_fidelity, t2_star_values):
    """Plot fidelity vs T2* coherence time (Doppler dephasing effect)."""
    print("Plotting T2* dephasing results...")

    mean_fidelities = [r['mean_fidelity'] for r in results_by_t2]
    std_fidelities = [r['std_fidelity'] for r in results_by_t2]

    for t2, r in zip(t2_star_values, results_by_t2):
        print(f"  T2* = {t2*1e6:.1f} us: F = {r['mean_fidelity']:.4f} +/- {r['std_fidelity']:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    t2_us = t2_star_values * 1e6

    ax.errorbar(t2_us, mean_fidelities, yerr=std_fidelities,
                fmt='o-', capsize=4, linewidth=2, markersize=8,
                label='Monte Carlo (T2* dephasing)')
    ax.axhline(y=ideal_fidelity, color='g', linestyle='--', linewidth=1.5,
               label=f'Ideal (no noise): F = {ideal_fidelity:.4f}')

    ax.set_xlabel('T2* Coherence Time (us)', fontsize=12)
    ax.set_ylabel('Gate Fidelity', fontsize=12)
    ax.set_title('CZ Gate Fidelity vs T2* (Doppler Dephasing)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 11)

    plt.tight_layout()
    filepath = OUTPUT_DIR / "fidelity_vs_t2_star.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()
    return filepath


def plot_fidelity_vs_temperature(results_by_temp, ideal_fidelity, temp_values,
                                  position_spreads, trap_freq):
    """Plot fidelity vs atomic temperature (position spread effect)."""
    print("Plotting temperature dependence results...")

    mean_fidelities = [r['mean_fidelity'] for r in results_by_temp]
    std_fidelities = [r['std_fidelity'] for r in results_by_temp]

    for temp, sigma, r in zip(temp_values, position_spreads, results_by_temp):
        print(f"  T = {temp:.0f} uK (sigma_pos = {sigma*1e3:.1f} nm): "
              f"F = {r['mean_fidelity']:.4f} +/- {r['std_fidelity']:.4f}")

    fig, ax1 = plt.subplots(figsize=(8, 6))

    color1 = 'tab:blue'
    ax1.errorbar(temp_values, mean_fidelities, yerr=std_fidelities,
                 fmt='o-', capsize=4, linewidth=2, markersize=8,
                 color=color1, label='Monte Carlo (position spread)')
    ax1.axhline(y=ideal_fidelity, color='g', linestyle='--', linewidth=1.5,
                label=f'Ideal: F = {ideal_fidelity:.4f}')

    ax1.set_xlabel('Atomic Temperature (uK)', fontsize=12)
    ax1.set_ylabel('Gate Fidelity', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.plot(temp_values, np.array(position_spreads) * 1e3, 's--',
             color=color2, markersize=6, alpha=0.7, label='sigma_pos')
    ax2.set_ylabel('Position Spread sigma_pos (nm)', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title(f'CZ Gate Fidelity vs Temperature (trap_freq = {trap_freq} kHz)', fontsize=14)

    plt.tight_layout()
    filepath = OUTPUT_DIR / "fidelity_vs_temperature.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()
    return filepath


def plot_combined_effects(result_t2_only, results_pos_only, results_combined,
                          ideal_fidelity, temp_values, T2_star):
    """Plot fidelity with combined T2* and temperature effects."""
    print("Plotting combined effects results...")

    # T2* only is a single result, replicated for all temp points
    fid_t2_only = [result_t2_only['mean_fidelity']] * len(temp_values)
    fid_pos_only = [r['mean_fidelity'] for r in results_pos_only]
    fid_combined = [r['mean_fidelity'] for r in results_combined]
    std_combined = [r['std_fidelity'] for r in results_combined]

    for temp, rp, rb in zip(temp_values, results_pos_only, results_combined):
        print(f"  T = {temp:.0f} uK: T2* only = {result_t2_only['mean_fidelity']:.4f}, "
              f"Pos only = {rp['mean_fidelity']:.4f}, "
              f"Combined = {rb['mean_fidelity']:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axhline(y=ideal_fidelity, color='gray', linestyle=':', linewidth=1.5,
               label=f'Ideal (no noise): F = {ideal_fidelity:.4f}')
    ax.plot(temp_values, fid_t2_only, 's--', linewidth=1.5, markersize=7,
            color='tab:green', label=f'T2* dephasing only (T2* = {T2_star*1e6:.0f} us)')
    ax.plot(temp_values, fid_pos_only, '^--', linewidth=1.5, markersize=7,
            color='tab:orange', label='Position spread only')
    ax.errorbar(temp_values, fid_combined, yerr=std_combined,
                fmt='o-', capsize=4, linewidth=2, markersize=8,
                color='tab:blue', label='Combined (T2* + position)')

    ax.set_xlabel('Atomic Temperature (uK)', fontsize=12)
    ax.set_ylabel('Gate Fidelity', fontsize=12)
    ax.set_title('CZ Gate Fidelity: Error Source Contributions', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 55)

    plt.tight_layout()
    filepath = OUTPUT_DIR / "fidelity_combined_effects.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()
    return filepath


def plot_infidelity_contributions(result_t2_only, results_pos_only, results_combined,
                                   ideal_infidelity, temp_values):
    """Plot infidelity contributions showing additive error behavior."""
    print("Plotting infidelity contributions results...")

    infid_t2_contrib_val = max(0, result_t2_only['mean_infidelity'] - ideal_infidelity)
    infid_t2_contrib = [infid_t2_contrib_val] * len(temp_values)
    infid_pos_contrib = []
    infid_total = []

    for rp, rb in zip(results_pos_only, results_combined):
        pos_contrib = rp['mean_infidelity'] - ideal_infidelity
        infid_pos_contrib.append(max(0, pos_contrib))
        infid_total.append(rb['mean_infidelity'])

    fig, ax = plt.subplots(figsize=(10, 6))

    width = 3.5
    x_pos = temp_values

    ax.bar(x_pos, [ideal_infidelity] * len(temp_values), width,
           label='Ideal infidelity', color='lightgray')
    ax.bar(x_pos, infid_t2_contrib, width, bottom=[ideal_infidelity] * len(temp_values),
           label='T2* contribution', color='tab:green', alpha=0.8)
    ax.bar(x_pos, infid_pos_contrib, width,
           bottom=[ideal_infidelity + t for t in infid_t2_contrib],
           label='Position contribution', color='tab:orange', alpha=0.8)

    ax.plot(x_pos, infid_total, 'ko-', markersize=8, linewidth=2,
            label='Total (measured)')

    ax.set_xlabel('Atomic Temperature (uK)', fontsize=12)
    ax.set_ylabel('Gate Infidelity (1 - F)', fontsize=12)
    ax.set_title('CZ Gate Error Budget Breakdown', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0, 55)

    plt.tight_layout()
    filepath = OUTPUT_DIR / "infidelity_contributions.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()
    return filepath


def main():
    """Generate all verification plots using parallel MC simulations."""
    from ryd_gate.ideal_cz import CZGateSimulator

    print("=" * 60)
    print("Monte Carlo Simulation Verification (Parallel)")
    print("=" * 60)

    # Initialize simulator (also inherited by forked worker processes)
    global _worker_sim
    print("\nInitializing CZGateSimulator...")
    sim = CZGateSimulator(decayflag=False, param_set='our', strategy='TO', blackmanflag=False)
    _worker_sim = sim

    ideal_infidelity = sim.avg_fidelity(X_TO)
    ideal_fidelity = 1 - ideal_infidelity
    print(f"Ideal gate infidelity: {ideal_infidelity:.6f}")
    print(f"Ideal gate fidelity: {ideal_fidelity:.6f}")

    # Settings
    n_shots = 50
    trap_freq = 50.0  # kHz
    seed = 42

    print(f"\nSettings: n_shots={n_shots}, trap_freq={trap_freq} kHz, seed={seed}")
    print("-" * 60)

    # Define all parameter sweeps
    t2_star_values = np.array([1.0, 2.0, 3.0, 5.0, 10.0]) * 1e-6
    temp_values_short = np.array([5.0, 15.0, 25.0, 40.0, 50.0])
    temp_values_long = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])
    T2_star_combined = 3e-6

    # Pre-compute position spreads (cheap, no parallelism needed)
    position_spreads = [sim._compute_position_sigma(t, trap_freq, None)
                        for t in temp_values_short]

    # Build all unique MC job specs: (tag, index, kwargs)
    jobs = []

    # Plot 1: T2* sweep (5 jobs)
    for i, t2 in enumerate(t2_star_values):
        jobs.append(('t2_sweep', i, {
            'n_shots': n_shots, 'T2_star': float(t2),
            'temperature': None, 'seed': seed,
        }))

    # Plot 2: Temperature sweep (5 jobs)
    for i, temp in enumerate(temp_values_short):
        jobs.append(('temp_sweep', i, {
            'n_shots': n_shots, 'T2_star': None,
            'temperature': float(temp), 'trap_freq': trap_freq, 'seed': seed,
        }))

    # Plots 3 & 4: T2*-only is a single job (same params for all temp points)
    jobs.append(('combined_t2', 0, {
        'n_shots': n_shots, 'T2_star': T2_star_combined,
        'temperature': None, 'seed': seed,
    }))

    # Position-only and combined: one job per temperature point
    for i, temp in enumerate(temp_values_long):
        jobs.append(('combined_pos', i, {
            'n_shots': n_shots, 'T2_star': None,
            'temperature': float(temp), 'trap_freq': trap_freq, 'seed': seed + 1,
        }))
        jobs.append(('combined_both', i, {
            'n_shots': n_shots, 'T2_star': T2_star_combined,
            'temperature': float(temp), 'trap_freq': trap_freq, 'seed': seed + 2,
        }))

    print(f"\nDispatching {len(jobs)} MC simulations across CPU cores...")

    # Run all MC simulations in parallel (workers inherit sim via fork)
    with ProcessPoolExecutor() as pool:
        futures = [(tag, idx, pool.submit(_run_mc_worker, kwargs))
                   for tag, idx, kwargs in jobs]
        tagged_results = [(tag, idx, f.result()) for tag, idx, f in futures]

    print("All simulations complete.\n")

    # Sort results back into per-plot arrays
    def collect(tag):
        items = sorted([(idx, r) for t, idx, r in tagged_results if t == tag])
        return [r for _, r in items]

    results_t2_sweep = collect('t2_sweep')
    results_temp_sweep = collect('temp_sweep')
    result_combined_t2 = collect('combined_t2')[0]  # Single result
    results_combined_pos = collect('combined_pos')
    results_combined_both = collect('combined_both')

    # Generate all four plots
    plot_files = []

    print("1. T2* Dephasing Analysis")
    plot_files.append(plot_fidelity_vs_t2_star(
        results_t2_sweep, ideal_fidelity, t2_star_values))

    print("\n2. Temperature (Position Spread) Analysis")
    plot_files.append(plot_fidelity_vs_temperature(
        results_temp_sweep, ideal_fidelity, temp_values_short,
        position_spreads, trap_freq))

    print("\n3. Combined Effects Analysis")
    plot_files.append(plot_combined_effects(
        result_combined_t2, results_combined_pos, results_combined_both,
        ideal_fidelity, temp_values_long, T2_star_combined))

    print("\n4. Infidelity Contributions (Error Budget)")
    plot_files.append(plot_infidelity_contributions(
        result_combined_t2, results_combined_pos, results_combined_both,
        ideal_infidelity, temp_values_long))

    print("\n" + "=" * 60)
    print("Verification complete! Generated plots:")
    for f in plot_files:
        print(f"  - {f}")
    print("=" * 60)

    return plot_files


if __name__ == "__main__":
    main()
