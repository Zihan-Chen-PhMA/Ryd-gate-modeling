#!/usr/bin/env python3
"""Generate CZ gate noise analysis report.

This script runs comprehensive diagnostics on the Rydberg CZ gate implementation
to identify and quantify sources of gate infidelity.

Usage:
    python scripts/generate_noise_report.py [--save-plots] [--output-dir DIR]

Output:
    - docs/noise_analysis_results.json: Numerical results
    - docs/figures/population_*.png: Population evolution plots (if --save-plots)
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

# Configure JAX before importing
import jax
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ryd_gate.full_error_model import jax_atom_Evolution


def run_cz_simulation(model, tlist, amp_420, phase_420, amp_1013):
    """Run CZ gate simulation for all 12 SSS initial states.
    
    Parameters
    ----------
    model : jax_atom_Evolution
        Initialized simulator instance.
    tlist : array
        Time points for integration.
    amp_420, phase_420, amp_1013 : callable
        Pulse functions.
        
    Returns
    -------
    sol : array
        Density matrix trajectories, shape (12, n_times, 100, 100).
    sol_mid : array
        Final states after intermediate decay, shape (12, 100, 100).
    """
    import time as time_module
    
    psi0_list = model.SSS_initial_state_list
    rho0_list = [model.psi_to_rho(psi0) for psi0 in psi0_list]
    
    print("Running CZ gate simulation for 12 SSS initial states...")
    print("  (JAX compilation may take several minutes on first run)")
    sys.stdout.flush()
    
    start_time = time_module.time()
    sol = model.integrate_rho_multi_jax(
        tlist, amp_420, phase_420, amp_1013, rho0_list
    )
    elapsed = time_module.time() - start_time
    print(f"  Simulation completed in {elapsed:.1f}s")
    
    # Apply intermediate state decay
    print("Applying intermediate state decay...")
    sys.stdout.flush()
    sol_mid = jnp.array([model.mid_state_decay(sol[n, -1]) for n in range(12)])
    
    return sol, sol_mid


def analyze_infidelity(model, sol_mid, psi0_list):
    """Analyze infidelity breakdown for all SSS states.
    
    Parameters
    ----------
    model : jax_atom_Evolution
        Simulator instance.
    sol_mid : array
        Final density matrices after decay.
    psi0_list : list
        Initial pure states.
        
    Returns
    -------
    results : list
        List of infidelity breakdown dictionaries.
    theta_mean : float
        Mean optimal Z-rotation angle.
    """
    results = []
    theta_values = []
    
    # SSS state labels
    state_labels = ['00', '01', '10', '11', 
                    '0+', '0-', '1+', '1-',
                    '+0', '-0', '+1', '-1']
    
    print("\nAnalyzing infidelity for each initial state...")
    for n in range(12):
        diag = model.diagnose_infidelity(sol_mid[n], psi0_list[n])
        diag['state_label'] = state_labels[n]
        diag['state_index'] = n
        
        # Also calculate fidelity with leakage (P00_withL metric)
        # This treats |L0⟩ as indistinguishable from |0⟩ for experimental comparison
        fid_with_L, theta_L, leakage_contrib = model.CZ_fidelity_with_leakage(
            sol_mid[n], psi0_list[n], diag['theta']
        )
        diag['fidelity_with_leakage'] = float(fid_with_L)
        diag['leakage_contribution'] = float(leakage_contrib)
        
        results.append(diag)
        
        # Collect theta for averaging (exclude superposition states for theta)
        if n in [0, 1, 2, 3, 6, 7, 8, 9]:
            theta_values.append(diag['theta'])
    
    theta_mean = float(np.mean(theta_values))
    
    return results, theta_mean


def compute_summary_statistics(results):
    """Compute summary statistics from infidelity results.
    
    Parameters
    ----------
    results : list
        List of infidelity breakdown dictionaries.
        
    Returns
    -------
    summary : dict
        Summary statistics including means and standard deviations.
    """
    keys = ['total_infidelity', 'leakage_error', 'rydberg_residual',
            'intermediate_residual', 'decay_error', 'coherent_error', 'fidelity',
            'fidelity_with_leakage', 'leakage_contribution']
    
    summary = {}
    for key in keys:
        values = [r[key] for r in results]
        summary[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }
    
    # Add per-state breakdown
    summary['per_state'] = results
    
    return summary


def generate_population_plots(model, tlist, sol, save_dir=None):
    """Generate population evolution plots for key initial states.
    
    Parameters
    ----------
    model : jax_atom_Evolution
        Simulator instance.
    tlist : array
        Time points.
    sol : array
        Density matrix trajectories.
    save_dir : str, optional
        Directory to save plots. If None, displays interactively.
        
    Returns
    -------
    population_data : dict
        Population data for each analyzed state.
    """
    # Key states to plot: |00⟩, |01⟩, |10⟩, |11⟩
    state_indices = [0, 1, 2, 3]
    state_labels = ['00', '01', '10', '11']
    
    population_data = {}
    
    print("\nGenerating population evolution plots...")
    for idx, label in zip(state_indices, state_labels):
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f'population_{label}.png')
        
        pop_data = model.diagnose_plot(
            tlist, sol[idx], 
            initial_label=label, 
            save_path=save_path
        )
        
        # Convert JAX arrays to lists for JSON serialization
        population_data[label] = {
            k: np.array(v).tolist() for k, v in pop_data.items()
        }
        
        if save_path:
            print(f"  Saved: {save_path}")
    
    return population_data


def get_physical_parameters(model):
    """Extract key physical parameters from the model.
    
    Parameters
    ----------
    model : jax_atom_Evolution
        Simulator instance.
        
    Returns
    -------
    params : dict
        Dictionary of physical parameters.
    """
    return {
        'levels': model.levels,
        'level_labels': model.level_label,
        'Delta_MHz': float(model.Delta / (2 * np.pi)),
        'rabi_eff_MHz': float(model.rabi_eff / (2 * np.pi)),
        'rabi_ratio': float(model.rabi_ratio),
        'Gamma_BBR_MHz': float(model.Gamma_BBR),
        'Gamma_RD_MHz': float(model.Gamma_RD),
        'Gamma_mid_MHz': float(model.Gamma_mid),
        'blockade_shift_MHz': float(model.B / (2 * np.pi)) if hasattr(model, 'B') else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate CZ gate noise analysis report'
    )
    parser.add_argument(
        '--save-plots', action='store_true',
        help='Save population evolution plots to docs/figures/'
    )
    parser.add_argument(
        '--output-dir', type=str, default='docs',
        help='Output directory for results (default: docs)'
    )
    parser.add_argument(
        '--n-points', type=int, default=500,
        help='Number of time points for integration (default: 500)'
    )
    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 60)
    print("CZ Gate Noise Analysis Report Generator")
    print("=" * 60)
    
    # Initialize model
    print("\nInitializing jax_atom_Evolution model...")
    print("  - Blockade: enabled")
    print("  - Rydberg decay: enabled")
    print("  - Intermediate decay: enabled")
    print("  - Distance: 3 μm")
    
    model = jax_atom_Evolution(
        blockade=True,
        ryd_decay=True,
        mid_decay=True,
        distance=3
    )
    
    # Get physical parameters
    params = get_physical_parameters(model)
    print(f"\nPhysical parameters:")
    print(f"  Δ = {params['Delta_MHz']:.1f} MHz (intermediate detuning)")
    print(f"  Ω_eff = {params['rabi_eff_MHz']:.1f} MHz (effective Rabi)")
    print(f"  Γ_mid = {params['Gamma_mid_MHz']:.2f} MHz (6P₃/₂ decay)")
    print(f"  Γ_BBR = {params['Gamma_BBR_MHz']:.6f} MHz (BBR decay)")
    print(f"  Γ_RD = {params['Gamma_RD_MHz']:.6f} MHz (radiative decay)")
    
    # Define pulse parameters (from optimized protocol)
    Omega = model.rabi_eff / (2 * jnp.pi)
    tf = 1.219096  # Normalized gate time
    A = 0.1122     # Phase modulation amplitude
    omegaf = 1.0431  # Phase modulation frequency
    phi0 = -0.72565603  # Phase offset
    deltaf = 0     # Frequency detuning
    
    # Pulse functions
    amp_420 = lambda t: 1.0
    phase_420 = lambda t: 2 * jnp.pi * (
        A * jnp.cos(2 * jnp.pi * omegaf * Omega * t - phi0) + deltaf * Omega * t
    )
    amp_1013 = lambda t: 1.0
    
    # Time grid
    gate_time = tf / Omega  # Physical gate time in μs
    tlist = jnp.linspace(0, gate_time, args.n_points)
    
    print(f"\nGate parameters:")
    print(f"  Gate time: {float(gate_time)*1000:.2f} ns")
    print(f"  Phase modulation: A={A}, ω={omegaf}, φ₀={phi0:.4f}")
    print(f"  Time points: {args.n_points}")
    
    # Run simulation
    sol, sol_mid = run_cz_simulation(model, tlist, amp_420, phase_420, amp_1013)
    
    # Analyze infidelity
    psi0_list = model.SSS_initial_state_list
    results, theta_mean = analyze_infidelity(model, sol_mid, psi0_list)
    
    # Compute summary statistics
    summary = compute_summary_statistics(results)
    summary['theta_mean'] = theta_mean
    
    # Print results
    print("\n" + "=" * 60)
    print("INFIDELITY BREAKDOWN (averaged over 12 SSS states)")
    print("=" * 60)
    print(f"{'Error Source':<25} {'Mean':>12} {'Std':>12}")
    print("-" * 60)
    
    error_sources = [
        ('Total Infidelity', 'total_infidelity'),
        ('Decay Error', 'decay_error'),
        ('Leakage Error', 'leakage_error'),
        ('Rydberg Residual', 'rydberg_residual'),
        ('Intermediate Residual', 'intermediate_residual'),
        ('Coherent Error', 'coherent_error'),
    ]
    
    for name, key in error_sources:
        mean = summary[key]['mean']
        std = summary[key]['std']
        print(f"{name:<25} {mean:>12.6f} {std:>12.6f}")
    
    print("-" * 60)
    print(f"{'Mean Fidelity':<25} {summary['fidelity']['mean']:>12.6f}")
    print(f"{'P00_withL (exp. compare)':<25} {summary['fidelity_with_leakage']['mean']:>12.6f}")
    print(f"{'L0 Hidden Contribution':<25} {summary['leakage_contribution']['mean']:>12.6f}")
    print(f"{'Optimal θ (rad)':<25} {theta_mean:>12.6f}")
    
    # Per-state breakdown
    print("\n" + "=" * 70)
    print("PER-STATE FIDELITY BREAKDOWN")
    print("=" * 70)
    print(f"{'State':<8} {'Fidelity':>10} {'P00_withL':>10} {'L0 Contrib':>10} {'Leakage':>10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['state_label']:<8} {r['fidelity']:>10.6f} {r['fidelity_with_leakage']:>10.6f} "
              f"{r['leakage_contribution']:>10.6f} {r['leakage_error']:>10.6f}")
    
    # Generate plots
    plot_save_dir = figures_dir if args.save_plots else None
    population_data = generate_population_plots(model, tlist, sol, plot_save_dir)
    
    # Prepare output data
    output_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'n_time_points': args.n_points,
            'gate_time_us': float(gate_time),
            'gate_time_ns': float(gate_time) * 1000,
        },
        'physical_parameters': params,
        'pulse_parameters': {
            'tf_normalized': tf,
            'A': A,
            'omegaf': omegaf,
            'phi0': phi0,
            'deltaf': deltaf,
        },
        'summary': {k: v for k, v in summary.items() if k != 'per_state'},
        'per_state_results': results,
        'time_list_us': np.array(tlist).tolist(),
    }
    
    # Optionally include population data (can be large)
    if args.save_plots:
        output_data['population_data'] = population_data
    
    # Save results
    output_path = os.path.join(args.output_dir, 'noise_analysis_results.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    if args.save_plots:
        print(f"Plots saved to: {figures_dir}/")
    print("=" * 60)
    
    return output_data


if __name__ == '__main__':
    main()
