#!/usr/bin/env python3
"""Verify Monte Carlo simulation by plotting fidelity vs temperature effects.

This script generates plots showing how gate fidelity changes with:
1. T2* coherence time (Doppler dephasing)
2. Atomic temperature (position spread)
3. Combined effects

Results are saved as PNG images for verification of the Monte Carlo implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ryd_gate.ideal_cz import CZGateSimulator

# Optimized TO pulse parameters (from test_cz_gate_phase.py)
X_TO = [-0.64168872, 1.14372811, 0.35715965, 1.51843443, 2.96448688, 1.21214853]

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_fidelity_vs_t2_star(sim, x, n_shots=100, seed=42):
    """Plot fidelity vs T2* coherence time (Doppler dephasing effect)."""
    print("Generating T2* dephasing plot...")

    # T2* values from 1 μs to 10 μs
    t2_star_values = np.array([1.0, 2.0, 3.0, 5.0, 10.0]) * 1e-6

    mean_fidelities = []
    std_fidelities = []

    for t2_star in t2_star_values:
        result = sim.run_monte_carlo_simulation(
            x, n_shots=n_shots, T2_star=t2_star, temperature=None, seed=seed
        )
        mean_fidelities.append(result.mean_fidelity)
        std_fidelities.append(result.std_fidelity)
        print(f"  T2* = {t2_star*1e6:.1f} μs: F = {result.mean_fidelity:.4f} ± {result.std_fidelity:.4f}")

    # Get ideal fidelity (no noise)
    ideal_infidelity = sim.avg_fidelity(x)
    ideal_fidelity = 1 - ideal_infidelity

    fig, ax = plt.subplots(figsize=(8, 6))
    t2_us = t2_star_values * 1e6

    ax.errorbar(t2_us, mean_fidelities, yerr=std_fidelities,
                fmt='o-', capsize=4, linewidth=2, markersize=8,
                label='Monte Carlo (T2* dephasing)')
    ax.axhline(y=ideal_fidelity, color='g', linestyle='--', linewidth=1.5,
               label=f'Ideal (no noise): F = {ideal_fidelity:.4f}')

    ax.set_xlabel('T2* Coherence Time (μs)', fontsize=12)
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


def plot_fidelity_vs_temperature(sim, x, n_shots=100, trap_freq=50.0, seed=42):
    """Plot fidelity vs atomic temperature (position spread effect)."""
    print("Generating temperature dependence plot...")

    # Temperature values from 5 μK to 50 μK
    temp_values = np.array([5.0, 15.0, 25.0, 40.0, 50.0])

    mean_fidelities = []
    std_fidelities = []
    position_spreads = []

    for temp in temp_values:
        result = sim.run_monte_carlo_simulation(
            x, n_shots=n_shots, T2_star=None,
            temperature=temp, trap_freq=trap_freq, seed=seed
        )
        mean_fidelities.append(result.mean_fidelity)
        std_fidelities.append(result.std_fidelity)

        # Compute position spread for reference
        sigma_pos = sim._compute_position_sigma(temp, trap_freq, None)
        position_spreads.append(sigma_pos)

        print(f"  T = {temp:.0f} μK (σ_pos = {sigma_pos*1e3:.1f} nm): "
              f"F = {result.mean_fidelity:.4f} ± {result.std_fidelity:.4f}")

    # Get ideal fidelity
    ideal_infidelity = sim.avg_fidelity(x)
    ideal_fidelity = 1 - ideal_infidelity

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Primary axis: Fidelity vs Temperature
    color1 = 'tab:blue'
    ax1.errorbar(temp_values, mean_fidelities, yerr=std_fidelities,
                 fmt='o-', capsize=4, linewidth=2, markersize=8,
                 color=color1, label='Monte Carlo (position spread)')
    ax1.axhline(y=ideal_fidelity, color='g', linestyle='--', linewidth=1.5,
                label=f'Ideal: F = {ideal_fidelity:.4f}')

    ax1.set_xlabel('Atomic Temperature (μK)', fontsize=12)
    ax1.set_ylabel('Gate Fidelity', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Secondary axis: Position spread
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.plot(temp_values, np.array(position_spreads) * 1e3, 's--',
             color=color2, markersize=6, alpha=0.7, label='σ_pos')
    ax2.set_ylabel('Position Spread σ_pos (nm)', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title(f'CZ Gate Fidelity vs Temperature (trap_freq = {trap_freq} kHz)', fontsize=14)

    plt.tight_layout()
    filepath = OUTPUT_DIR / "fidelity_vs_temperature.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()

    return filepath


def plot_combined_effects(sim, x, n_shots=100, trap_freq=50.0, seed=42):
    """Plot fidelity with combined T2* and temperature effects."""
    print("Generating combined effects plot...")

    # Fixed T2* = 3 μs (typical experimental value)
    T2_star = 3e-6

    # Temperature values
    temp_values = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])

    # Run simulations for three cases
    fid_t2_only = []
    fid_pos_only = []
    fid_combined = []
    std_combined = []

    for temp in temp_values:
        # T2* only (same for all temps, but run for consistency)
        result_t2 = sim.run_monte_carlo_simulation(
            x, n_shots=n_shots, T2_star=T2_star, temperature=None, seed=seed
        )
        fid_t2_only.append(result_t2.mean_fidelity)

        # Position only
        result_pos = sim.run_monte_carlo_simulation(
            x, n_shots=n_shots, T2_star=None,
            temperature=temp, trap_freq=trap_freq, seed=seed+1
        )
        fid_pos_only.append(result_pos.mean_fidelity)

        # Combined
        result_both = sim.run_monte_carlo_simulation(
            x, n_shots=n_shots, T2_star=T2_star,
            temperature=temp, trap_freq=trap_freq, seed=seed+2
        )
        fid_combined.append(result_both.mean_fidelity)
        std_combined.append(result_both.std_fidelity)

        print(f"  T = {temp:.0f} μK: T2* only = {result_t2.mean_fidelity:.4f}, "
              f"Pos only = {result_pos.mean_fidelity:.4f}, "
              f"Combined = {result_both.mean_fidelity:.4f}")

    # Get ideal fidelity
    ideal_infidelity = sim.avg_fidelity(x)
    ideal_fidelity = 1 - ideal_infidelity

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axhline(y=ideal_fidelity, color='gray', linestyle=':', linewidth=1.5,
               label=f'Ideal (no noise): F = {ideal_fidelity:.4f}')
    ax.plot(temp_values, fid_t2_only, 's--', linewidth=1.5, markersize=7,
            color='tab:green', label=f'T2* dephasing only (T2* = {T2_star*1e6:.0f} μs)')
    ax.plot(temp_values, fid_pos_only, '^--', linewidth=1.5, markersize=7,
            color='tab:orange', label='Position spread only')
    ax.errorbar(temp_values, fid_combined, yerr=std_combined,
                fmt='o-', capsize=4, linewidth=2, markersize=8,
                color='tab:blue', label='Combined (T2* + position)')

    ax.set_xlabel('Atomic Temperature (μK)', fontsize=12)
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


def plot_infidelity_contributions(sim, x, n_shots=100, trap_freq=50.0, seed=42):
    """Plot infidelity contributions showing additive error behavior."""
    print("Generating infidelity contributions plot...")

    T2_star = 3e-6
    temp_values = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])

    ideal_infidelity = sim.avg_fidelity(x)

    infid_t2_contrib = []
    infid_pos_contrib = []
    infid_total = []

    for temp in temp_values:
        # Get error budget
        result_t2 = sim.run_monte_carlo_simulation(
            x, n_shots=n_shots, T2_star=T2_star, temperature=None, seed=seed
        )
        result_pos = sim.run_monte_carlo_simulation(
            x, n_shots=n_shots, T2_star=None,
            temperature=temp, trap_freq=trap_freq, seed=seed+1
        )
        result_both = sim.run_monte_carlo_simulation(
            x, n_shots=n_shots, T2_star=T2_star,
            temperature=temp, trap_freq=trap_freq, seed=seed+2
        )

        t2_contrib = result_t2.mean_infidelity - ideal_infidelity
        pos_contrib = result_pos.mean_infidelity - ideal_infidelity

        infid_t2_contrib.append(max(0, t2_contrib))  # Clamp small negatives
        infid_pos_contrib.append(max(0, pos_contrib))
        infid_total.append(result_both.mean_infidelity)

    fig, ax = plt.subplots(figsize=(10, 6))

    width = 3.5
    x_pos = temp_values

    # Stacked bar chart
    ax.bar(x_pos, [ideal_infidelity] * len(temp_values), width,
           label='Ideal infidelity', color='lightgray')
    ax.bar(x_pos, infid_t2_contrib, width, bottom=[ideal_infidelity] * len(temp_values),
           label='T2* contribution', color='tab:green', alpha=0.8)
    ax.bar(x_pos, infid_pos_contrib, width,
           bottom=[ideal_infidelity + t for t in infid_t2_contrib],
           label='Position contribution', color='tab:orange', alpha=0.8)

    # Overlay total measured infidelity
    ax.plot(x_pos, infid_total, 'ko-', markersize=8, linewidth=2,
            label='Total (measured)')

    ax.set_xlabel('Atomic Temperature (μK)', fontsize=12)
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
    """Generate all verification plots."""
    print("=" * 60)
    print("Monte Carlo Simulation Verification")
    print("=" * 60)

    # Initialize simulator
    print("\nInitializing CZGateSimulator...")
    sim = CZGateSimulator(decayflag=False, param_set='our', strategy='TO')

    # Check ideal fidelity with optimized parameters
    ideal_infidelity = sim.avg_fidelity(X_TO)
    print(f"Ideal gate infidelity: {ideal_infidelity:.6f}")
    print(f"Ideal gate fidelity: {1 - ideal_infidelity:.6f}")

    # Settings
    n_shots = 50  # Reduced for faster execution
    trap_freq = 50.0  # kHz
    seed = 42

    print(f"\nSettings: n_shots={n_shots}, trap_freq={trap_freq} kHz, seed={seed}")
    print("-" * 60)

    # Generate plots
    plot_files = []

    print("\n1. T2* Dephasing Analysis")
    plot_files.append(plot_fidelity_vs_t2_star(sim, X_TO, n_shots=n_shots, seed=seed))

    print("\n2. Temperature (Position Spread) Analysis")
    plot_files.append(plot_fidelity_vs_temperature(sim, X_TO, n_shots=n_shots,
                                                    trap_freq=trap_freq, seed=seed))

    print("\n3. Combined Effects Analysis")
    plot_files.append(plot_combined_effects(sim, X_TO, n_shots=n_shots,
                                            trap_freq=trap_freq, seed=seed))

    print("\n4. Infidelity Contributions (Error Budget)")
    plot_files.append(plot_infidelity_contributions(sim, X_TO, n_shots=n_shots,
                                                     trap_freq=trap_freq, seed=seed))

    print("\n" + "=" * 60)
    print("Verification complete! Generated plots:")
    for f in plot_files:
        print(f"  - {f}")
    print("=" * 60)

    return plot_files


if __name__ == "__main__":
    main()
