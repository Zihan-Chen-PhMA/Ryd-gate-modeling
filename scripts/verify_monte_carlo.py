#!/usr/bin/env python3
"""Verify Monte Carlo simulation by plotting fidelity vs noise effects.

This script generates plots showing how gate fidelity changes with:
1. Detuning noise (sigma_detuning sweep)
2. Position noise (sigma_pos_xyz sweep)
3. Combined effects

Results are saved as PNG images for verification of the Monte Carlo implementation.

Uses multiprocessing to run independent MC simulations in parallel across CPU cores.
"""

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from scipy.constants import k as kB

# Optimized TO pulse parameters (from test_cz_gate_phase.py)
X_TO = [-0.64168872, 1.14372811, 0.35715965, 1.51843443, 2.96448688, 1.21214853]

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-worker simulator — set in main() before pool creation.
# On Linux (fork), child processes inherit this via copy-on-write.
_worker_sim = None


def _temp_to_sigma_pos(temperature_uK, trap_freq_kHz):
    """Convert temperature and trap frequency to position spread in meters.

    Parameters
    ----------
    temperature_uK : float
        Atomic temperature in μK.
    trap_freq_kHz : float
        Trap frequency in kHz.

    Returns
    -------
    float
        Position spread standard deviation in meters.
    """
    T_K = temperature_uK * 1e-6  # μK to K
    omega = 2 * np.pi * trap_freq_kHz * 1e3  # kHz to rad/s
    m_Rb = 87 * 1.66054e-27  # 87Rb mass in kg
    return np.sqrt(kB * T_K / (m_Rb * omega**2))  # meters


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


def plot_fidelity_vs_sigma_detuning(results_by_sigma, ideal_fidelity, sigma_values):
    """Plot fidelity vs detuning noise (dephasing effect)."""
    print("Plotting dephasing results...")

    mean_fidelities = [r['mean_fidelity'] for r in results_by_sigma]
    std_fidelities = [r['std_fidelity'] for r in results_by_sigma]

    for sigma, r in zip(sigma_values, results_by_sigma):
        print(f"  sigma_detuning = {sigma/1e3:.0f} kHz: "
              f"F = {r['mean_fidelity']:.4f} +/- {r['std_fidelity']:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sigma_kHz = np.array(sigma_values) / 1e3

    ax.errorbar(sigma_kHz, mean_fidelities, yerr=std_fidelities,
                fmt='o-', capsize=4, linewidth=2, markersize=8,
                label='Monte Carlo (dephasing)')
    ax.axhline(y=ideal_fidelity, color='g', linestyle='--', linewidth=1.5,
               label=f'Ideal (no noise): F = {ideal_fidelity:.4f}')

    ax.set_xlabel('Detuning Noise sigma_detuning (kHz)', fontsize=12)
    ax.set_ylabel('Gate Fidelity', fontsize=12)
    ax.set_title('CZ Gate Fidelity vs Detuning Noise', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = OUTPUT_DIR / "fidelity_vs_sigma_detuning.png"
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
        print(f"  T = {temp:.0f} uK (sigma_pos = {sigma*1e9:.1f} nm): "
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
    ax2.plot(temp_values, np.array(position_spreads) * 1e9, 's--',
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


def plot_combined_effects(result_deph_only, results_pos_only, results_combined,
                          ideal_fidelity, temp_values, sigma_detuning):
    """Plot fidelity with combined dephasing and position effects."""
    print("Plotting combined effects results...")

    fid_deph_only = [result_deph_only['mean_fidelity']] * len(temp_values)
    fid_pos_only = [r['mean_fidelity'] for r in results_pos_only]
    fid_combined = [r['mean_fidelity'] for r in results_combined]
    std_combined = [r['std_fidelity'] for r in results_combined]

    for temp, rp, rb in zip(temp_values, results_pos_only, results_combined):
        print(f"  T = {temp:.0f} uK: Dephasing only = {result_deph_only['mean_fidelity']:.4f}, "
              f"Pos only = {rp['mean_fidelity']:.4f}, "
              f"Combined = {rb['mean_fidelity']:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axhline(y=ideal_fidelity, color='gray', linestyle=':', linewidth=1.5,
               label=f'Ideal (no noise): F = {ideal_fidelity:.4f}')
    ax.plot(temp_values, fid_deph_only, 's--', linewidth=1.5, markersize=7,
            color='tab:green',
            label=f'Dephasing only (sigma = {sigma_detuning/1e3:.0f} kHz)')
    ax.plot(temp_values, fid_pos_only, '^--', linewidth=1.5, markersize=7,
            color='tab:orange', label='Position spread only')
    ax.errorbar(temp_values, fid_combined, yerr=std_combined,
                fmt='o-', capsize=4, linewidth=2, markersize=8,
                color='tab:blue', label='Combined (dephasing + position)')

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


def plot_infidelity_contributions(result_deph_only, results_pos_only, results_combined,
                                   ideal_infidelity, temp_values):
    """Plot infidelity contributions showing additive error behavior."""
    print("Plotting infidelity contributions results...")

    infid_deph_contrib_val = max(0, result_deph_only['mean_infidelity'] - ideal_infidelity)
    infid_deph_contrib = [infid_deph_contrib_val] * len(temp_values)
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
    ax.bar(x_pos, infid_deph_contrib, width, bottom=[ideal_infidelity] * len(temp_values),
           label='Dephasing contribution', color='tab:green', alpha=0.8)
    ax.bar(x_pos, infid_pos_contrib, width,
           bottom=[ideal_infidelity + d for d in infid_deph_contrib],
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

    # Create simulator with both MC flags enabled for the worker
    global _worker_sim
    print("\nInitializing CZGateSimulator...")
    sigma_detuning_default = 170e3  # 170 kHz in Hz
    sigma_pos_default = (70e-6, 70e-6, 170e-6)  # meters
    sim = CZGateSimulator(
        param_set='our', strategy='TO', blackmanflag=False,
        enable_rydberg_dephasing=True,
        enable_position_error=True,
        sigma_detuning=sigma_detuning_default,
        sigma_pos_xyz=sigma_pos_default,
    )
    _worker_sim = sim

    # Use _gate_infidelity_single for ideal (no noise) computation
    ideal_infidelity = sim._gate_infidelity_single(X_TO)
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
    # Sigma detuning sweep (corresponding to different T2* values)
    sigma_detuning_values = [50e3, 100e3, 170e3, 250e3, 400e3]  # Hz

    temp_values_short = np.array([5.0, 15.0, 25.0, 40.0, 50.0])
    temp_values_long = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])
    sigma_detuning_combined = 170e3  # Hz

    # Pre-compute position spreads
    position_spreads = [_temp_to_sigma_pos(t, trap_freq)
                        for t in temp_values_short]

    # Build all unique MC job specs: (tag, index, kwargs)
    jobs = []

    # Plot 1: sigma_detuning sweep (5 jobs) — dephasing only, no position
    for i, sigma_d in enumerate(sigma_detuning_values):
        jobs.append(('deph_sweep', i, {
            'n_shots': n_shots, 'sigma_detuning': sigma_d,
            'sigma_pos_xyz': None, 'seed': seed,
        }))

    # Plot 2: Temperature sweep (5 jobs) — position only, no dephasing
    for i, temp in enumerate(temp_values_short):
        sigma_pos = _temp_to_sigma_pos(temp, trap_freq)
        sigma_xyz = (sigma_pos, sigma_pos, sigma_pos)
        jobs.append(('temp_sweep', i, {
            'n_shots': n_shots, 'sigma_detuning': None,
            'sigma_pos_xyz': sigma_xyz, 'seed': seed,
        }))

    # Plots 3 & 4: Dephasing-only is a single job
    jobs.append(('combined_deph', 0, {
        'n_shots': n_shots, 'sigma_detuning': sigma_detuning_combined,
        'sigma_pos_xyz': None, 'seed': seed,
    }))

    # Position-only and combined: one job per temperature point
    for i, temp in enumerate(temp_values_long):
        sigma_pos = _temp_to_sigma_pos(temp, trap_freq)
        sigma_xyz = (sigma_pos, sigma_pos, sigma_pos)
        jobs.append(('combined_pos', i, {
            'n_shots': n_shots, 'sigma_detuning': None,
            'sigma_pos_xyz': sigma_xyz, 'seed': seed + 1,
        }))
        jobs.append(('combined_both', i, {
            'n_shots': n_shots, 'sigma_detuning': sigma_detuning_combined,
            'sigma_pos_xyz': sigma_xyz, 'seed': seed + 2,
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

    results_deph_sweep = collect('deph_sweep')
    results_temp_sweep = collect('temp_sweep')
    result_combined_deph = collect('combined_deph')[0]  # Single result
    results_combined_pos = collect('combined_pos')
    results_combined_both = collect('combined_both')

    # Generate all four plots
    plot_files = []

    print("1. Dephasing (sigma_detuning sweep) Analysis")
    plot_files.append(plot_fidelity_vs_sigma_detuning(
        results_deph_sweep, ideal_fidelity, sigma_detuning_values))

    print("\n2. Temperature (Position Spread) Analysis")
    plot_files.append(plot_fidelity_vs_temperature(
        results_temp_sweep, ideal_fidelity, temp_values_short,
        position_spreads, trap_freq))

    print("\n3. Combined Effects Analysis")
    plot_files.append(plot_combined_effects(
        result_combined_deph, results_combined_pos, results_combined_both,
        ideal_fidelity, temp_values_long, sigma_detuning_combined))

    print("\n4. Infidelity Contributions (Error Budget)")
    plot_files.append(plot_infidelity_contributions(
        result_combined_deph, results_combined_pos, results_combined_both,
        ideal_infidelity, temp_values_long))

    print("\n" + "=" * 60)
    print("Verification complete! Generated plots:")
    for f in plot_files:
        print(f"  - {f}")
    print("=" * 60)

    return plot_files


if __name__ == "__main__":
    main()
