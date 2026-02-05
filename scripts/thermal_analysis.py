#!/usr/bin/env python3
"""Compare CZ gate fidelity with and without thermal effects.

Runs the optimized CZ gate protocol under four conditions:
  1. T = 0 (no thermal effects)
  2. Doppler only at several temperatures
  3. Position spread only at several temperatures
  4. Combined Doppler + position spread

Outputs a JSON report to docs/thermal_analysis_results.json.

Usage:
    uv run python scripts/thermal_analysis.py
"""

import json
import os
import sys
import time as time_module
from datetime import datetime

import numpy as np

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ryd_gate.full_error_model import jax_atom_Evolution


def build_pulse_functions(model):
    """Return (amp_420, phase_420, amp_1013, tlist, gate_time) for the
    optimized CZ protocol used in the noise-analysis report."""
    Omega = model.rabi_eff / (2 * jnp.pi)
    tf = 1.219096
    A = 0.1122
    omegaf = 1.0431
    phi0 = -0.72565603
    deltaf = 0

    amp_420 = lambda t: 1.0
    phase_420 = lambda t: 2 * jnp.pi * (
        A * jnp.cos(2 * jnp.pi * omegaf * Omega * t - phi0) + deltaf * Omega * t
    )
    amp_1013 = lambda t: 1.0

    gate_time = tf / Omega
    tlist = jnp.linspace(0, gate_time, 200)
    return amp_420, phase_420, amp_1013, tlist, float(gate_time)


def baseline_fidelity(model, tlist, amp_420, phase_420, amp_1013):
    """Run the standard (T=0) simulation and return per-state fidelities."""
    psi0_list = model.SSS_initial_state_list
    rho0_list = [model.psi_to_rho(p) for p in psi0_list]

    print("  Running baseline (T=0) simulation …")
    sys.stdout.flush()
    t0 = time_module.time()
    sol = model.integrate_rho_multi_jax(tlist, amp_420, phase_420, amp_1013, rho0_list)
    print(f"    integration: {time_module.time() - t0:.1f}s")

    fids = []
    fids_wL = []
    thetas = []
    for n in range(12):
        rho_mid = model.mid_state_decay(sol[n, -1])
        fid, theta = model.CZ_fidelity(rho_mid, state_initial=psi0_list[n])
        fid_wL, _, _ = model.CZ_fidelity_with_leakage(rho_mid, psi0_list[n], theta)
        fids.append(float(fid))
        fids_wL.append(float(fid_wL))
        thetas.append(float(theta))

    return {
        'per_state_fidelity': fids,
        'per_state_fidelity_with_leakage': fids_wL,
        'mean_fidelity': float(np.mean(fids)),
        'mean_fidelity_with_leakage': float(np.mean(fids_wL)),
        'theta_values': thetas,
    }


def thermal_sweep(model, tlist, amp_420, phase_420, amp_1013,
                   temperatures, n_samples, trap_freq=None, label=""):
    """Run simulate_with_thermal_effects across a list of temperatures."""
    results = []
    for T in temperatures:
        tag = f"T={T} μK"
        if trap_freq is not None:
            tag += f", ν_trap={trap_freq} kHz"
        print(f"  {label} {tag}, n_samples={n_samples} …")
        sys.stdout.flush()
        t0 = time_module.time()
        r = model.simulate_with_thermal_effects(
            tlist, amp_420, phase_420, amp_1013,
            T_atom=T, n_samples=n_samples, seed=42,
            trap_freq=trap_freq,
        )
        elapsed = time_module.time() - t0
        entry = {
            'T_uK': T,
            'trap_freq_kHz': trap_freq,
            'n_samples': n_samples,
            'mean_fidelity': r['mean_fidelity'],
            'std_fidelity': r['std_fidelity'],
            'fidelity_list': r['fidelity_list'],
            'doppler_shifts_MHz': r['doppler_shifts'],
            'elapsed_s': round(elapsed, 1),
        }
        if 'position_shifts' in r:
            entry['position_shifts_um'] = r['position_shifts']
        results.append(entry)
        print(f"    F = {r['mean_fidelity']:.6f} ± {r['std_fidelity']:.6f}  ({elapsed:.1f}s)")
    return results


def main():
    os.makedirs('docs', exist_ok=True)

    print("=" * 65)
    print("Thermal Effects Analysis — CZ Gate Fidelity")
    print("=" * 65)

    model = jax_atom_Evolution(blockade=True, ryd_decay=True,
                                mid_decay=True, distance=3)
    amp_420, phase_420, amp_1013, tlist, gate_time = build_pulse_functions(model)

    print(f"\nGate time: {gate_time*1000:.2f} ns")
    print(f"Doppler std at 10 μK: {model.doppler_std(10.0):.4f} MHz")
    print(f"Position spread std at 10 μK / 100 kHz trap: "
          f"{model.position_spread_std(10.0, 100.0):.4f} μm")
    print()

    # 1. Baseline
    print("[1/3] Baseline (T = 0)")
    bl = baseline_fidelity(model, tlist, amp_420, phase_420, amp_1013)
    print(f"  Mean fidelity        = {bl['mean_fidelity']:.6f}")
    print(f"  Mean fidelity (w/ L) = {bl['mean_fidelity_with_leakage']:.6f}")
    print()

    temperatures = [5.0, 20.0, 50.0]
    N_MC = 5  # Monte Carlo samples per temperature

    # 2. Doppler only
    print("[2/3] Doppler-only sweep")
    doppler_results = thermal_sweep(
        model, tlist, amp_420, phase_420, amp_1013,
        temperatures, N_MC, trap_freq=None, label="Doppler")
    print()

    # 3. Combined Doppler + position spread (trap_freq = 100 kHz)
    N_MC_combined = 3
    print("[3/3] Combined (Doppler + position spread, trap = 100 kHz)")
    combined_100 = thermal_sweep(
        model, tlist, amp_420, phase_420, amp_1013,
        temperatures, N_MC_combined, trap_freq=100.0, label="Combined(100kHz)")
    print()

    # Compute deltas
    print("=" * 65)
    print("SUMMARY: Fidelity reduction from thermal effects")
    print("=" * 65)
    print(f"{'T (μK)':<8} {'Baseline':>10} {'Doppler':>10} {'ΔF_D':>10} "
          f"{'Comb100':>10} {'ΔF_C100':>10}")
    print("-" * 68)
    for i, T in enumerate(temperatures):
        fd = doppler_results[i]['mean_fidelity']
        fc100 = combined_100[i]['mean_fidelity']
        dfd = bl['mean_fidelity'] - fd
        dfc100 = bl['mean_fidelity'] - fc100
        print(f"{T:<8.0f} {bl['mean_fidelity']:>10.6f} {fd:>10.6f} {dfd:>10.6f} "
              f"{fc100:>10.6f} {dfc100:>10.6f}")

    # Physical parameter summary
    phys = {
        'distance_um': 3.0,
        'C6_MHz_um6': float(model.C6 / (2 * np.pi)),
        'V_blockade_MHz': float(model.V / (2 * np.pi)),
    }
    doppler_stds = {f"{T}_uK": model.doppler_std(T) for T in temperatures}
    pos_stds_100 = {f"{T}_uK": model.position_spread_std(T, 100.0) for T in temperatures}

    output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'gate_time_ns': gate_time * 1000,
            'n_mc_samples_doppler': N_MC,
            'n_mc_samples_combined': N_MC_combined,
            'temperatures_uK': temperatures,
        },
        'physical_parameters': phys,
        'doppler_std_MHz': doppler_stds,
        'position_spread_std_um_100kHz': pos_stds_100,
        'baseline': bl,
        'doppler_only': doppler_results,
        'combined_trap_100kHz': combined_100,
    }

    out_path = 'docs/thermal_analysis_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == '__main__':
    main()
