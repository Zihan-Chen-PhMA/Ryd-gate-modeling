"""Generate CZ gate fidelity simulation report using JAX.

This script runs the full error model simulation and outputs results
for documentation.
"""

import json
import time
from datetime import datetime

import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from ryd_gate.full_error_model import jax_atom_Evolution


def run_fidelity_simulation():
    """Run CZ gate fidelity simulation and return results."""
    print("Initializing model...")
    start_time = time.time()
    
    model = jax_atom_Evolution()
    init_time = time.time() - start_time
    print(f"Model initialized in {init_time:.2f}s")
    
    # Simulation parameters (from research)
    Omega = model.rabi_eff / (2 * jnp.pi)
    tf = 1.219096
    A = 0.1122
    omegaf = 1.0431
    phi0 = -0.72565603
    deltaf = 0
    
    # Pulse definitions
    amp_420 = lambda t: 1
    phase_420 = lambda t: 2 * jnp.pi * (A * jnp.cos(2 * jnp.pi * omegaf * Omega * t - phi0) + deltaf * Omega * t)
    amp_1013 = lambda t: 1
    
    # Time list for evolution
    tlist = jnp.linspace(0, tf / Omega, 2)
    
    print("Running density matrix evolution...")
    sim_start = time.time()
    
    psi0_list = model.SSS_initial_state_list
    sol = model.integrate_rho_multi_jax(
        tlist,
        amp_420,
        phase_420,
        amp_1013,
        [model.psi_to_rho(psi0) for psi0 in psi0_list],
    )
    
    sim_time = time.time() - sim_start
    print(f"Evolution completed in {sim_time:.2f}s")
    
    # Apply mid-state decay
    print("Applying mid-state decay...")
    sol_mid = jnp.array([model.mid_state_decay(sol[n, -1]) for n in range(12)])
    
    # Calculate fidelities
    print("Calculating fidelities...")
    fidelities = []
    thetas = []
    
    for n in range(12):
        fid_raw, theta = model.CZ_fidelity(sol_mid[n], psi0_list[n])
        fidelities.append(float(fid_raw))
        thetas.append(float(theta))
    
    fid_raw_mean = sum(fidelities) / 12
    theta_mean = sum(thetas[i] for i in [0, 1, 2, 3, 6, 7, 8, 9]) / 8
    
    # Calculate fidelity with mean theta
    fid_with_theta = []
    for n in range(12):
        fid, _ = model.CZ_fidelity(sol_mid[n], psi0_list[n], theta_mean)
        fid_with_theta.append(float(fid))
    
    fid_mean = sum(fid_with_theta) / 12
    
    total_time = time.time() - start_time
    
    # Collect results
    results = {
        "timestamp": datetime.now().isoformat(),
        "simulation_parameters": {
            "gate_time_tf": float(tf),
            "amplitude_A": float(A),
            "frequency_omegaf": float(omegaf),
            "phase_phi0": float(phi0),
            "rabi_eff_MHz": float(model.rabi_eff / (2 * jnp.pi)),
            "distance_um": float(model.default_d),
            "delta_MHz": float(model.Delta / (2 * jnp.pi)),
        },
        "physical_parameters": {
            "C6_coefficient": float(model.C6 / (2 * jnp.pi)),
            "blockade_shift_V_MHz": float(model.V / (2 * jnp.pi)),
            "rabi_420_MHz": float(model.rabi_420 / (2 * jnp.pi)),
            "rabi_1013_MHz": float(model.rabi_1013 / (2 * jnp.pi)),
            "gamma_bbr_MHz": float(model.Gamma_BBR),
            "gamma_rd_MHz": float(model.Gamma_RD),
            "gamma_mid_MHz": float(model.Gamma_mid),
        },
        "fidelity_results": {
            "individual_fidelities": fidelities,
            "individual_thetas_rad": thetas,
            "mean_raw_fidelity": float(fid_raw_mean),
            "mean_theta_rad": float(theta_mean),
            "mean_fidelity_with_correction": float(fid_mean),
            "fidelities_with_mean_theta": fid_with_theta,
        },
        "timing": {
            "init_time_s": init_time,
            "simulation_time_s": sim_time,
            "total_time_s": total_time,
        }
    }
    
    return results


def generate_markdown_report(results):
    """Generate markdown report from simulation results."""
    
    params = results["simulation_parameters"]
    phys = results["physical_parameters"]
    fid = results["fidelity_results"]
    timing = results["timing"]
    
    state_labels = [
        "(|00⟩+|01⟩+|10⟩+|11⟩)/2",
        "(|00⟩-|01⟩-|10⟩+|11⟩)/2",
        "(|00⟩+i|01⟩+i|10⟩-|11⟩)/2",
        "(|00⟩-i|01⟩-i|10⟩-|11⟩)/2",
        "|00⟩",
        "|11⟩",
        "(|00⟩+|01⟩+|10⟩-|11⟩)/2",
        "(|00⟩-|01⟩-|10⟩-|11⟩)/2",
        "(|00⟩+i|01⟩+i|10⟩+|11⟩)/2",
        "(|00⟩-i|01⟩-i|10⟩+|11⟩)/2",
        "(|00⟩+i|11⟩)/√2",
        "(|00⟩-i|11⟩)/√2",
    ]
    
    report = f"""# CZ Gate Fidelity Simulation Results

*Generated: {results["timestamp"]}*

## Summary

| Metric | Value |
|--------|-------|
| **Mean Raw Fidelity** | {fid["mean_raw_fidelity"]:.6f} |
| **Mean Fidelity (θ-corrected)** | {fid["mean_fidelity_with_correction"]:.6f} |
| **Infidelity** | {(1 - fid["mean_fidelity_with_correction"]):.2e} |
| **Mean θ correction** | {fid["mean_theta_rad"]:.4f} rad |

## Simulation Parameters

### Gate Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| t_f | {params["gate_time_tf"]:.6f} | Gate time (normalized) |
| A | {params["amplitude_A"]:.4f} | Phase modulation amplitude |
| ω_f | {params["frequency_omegaf"]:.4f} | Phase modulation frequency |
| φ₀ | {params["phase_phi0"]:.5f} rad | Initial phase offset |
| Ω_eff | {params["rabi_eff_MHz"]:.2f} MHz | Effective Rabi frequency |

### Physical Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Interatomic distance | {params["distance_um"]:.1f} μm | Distance between atoms |
| Δ (detuning) | {params["delta_MHz"]:.1f} MHz | Intermediate state detuning |
| C₆ | {phys["C6_coefficient"]:.0f} MHz·μm⁶ | Van der Waals coefficient |
| V (blockade) | {phys["blockade_shift_V_MHz"]:.1f} MHz | Rydberg blockade shift |
| Ω₄₂₀ | {phys["rabi_420_MHz"]:.2f} MHz | 420 nm Rabi frequency |
| Ω₁₀₁₃ | {phys["rabi_1013_MHz"]:.2f} MHz | 1013 nm Rabi frequency |

### Decay Rates

| Parameter | Value | Description |
|-----------|-------|-------------|
| Γ_BBR | {phys["gamma_bbr_MHz"]:.6f} MHz | Black-body radiation decay |
| Γ_RD | {phys["gamma_rd_MHz"]:.6f} MHz | Radiative decay |
| Γ_mid | {phys["gamma_mid_MHz"]:.2f} MHz | Intermediate state decay |

## Individual State Fidelities

| State | Initial State | Raw Fidelity | θ (rad) | Corrected Fidelity |
|-------|---------------|--------------|---------|-------------------|
"""
    
    for i, label in enumerate(state_labels):
        report += f"| {i+1} | {label} | {fid['individual_fidelities'][i]:.6f} | {fid['individual_thetas_rad'][i]:.4f} | {fid['fidelities_with_mean_theta'][i]:.6f} |\n"
    
    report += f"""
## Error Analysis

The main sources of error in the CZ gate include:

1. **Spontaneous emission from Rydberg states** (Γ_BBR + Γ_RD ≈ {(phys["gamma_bbr_MHz"] + phys["gamma_rd_MHz"])*1000:.3f} kHz)
2. **Intermediate state scattering** (Γ_mid = {phys["gamma_mid_MHz"]:.2f} MHz)
3. **Imperfect blockade** (V/Ω_eff = {phys["blockade_shift_V_MHz"]/params["rabi_eff_MHz"]:.1f})

## Computation Time

| Phase | Time |
|-------|------|
| Model initialization | {timing["init_time_s"]:.2f} s |
| Density matrix evolution | {timing["simulation_time_s"]:.2f} s |
| **Total** | {timing["total_time_s"]:.2f} s |

## References

- Evered *et al.*, "High-fidelity parallel entangling gates on a neutral-atom quantum computer", Nature **622**, 268 (2023)
- See `paper/en_v2.tex` for detailed theoretical derivations

---
*Simulation performed using JAX-accelerated density matrix evolution with full error model including Rydberg blockade, spontaneous decay, and AC Stark shifts.*
"""
    
    return report


if __name__ == "__main__":
    print("=" * 60)
    print("CZ Gate Fidelity Simulation")
    print("=" * 60)
    
    results = run_fidelity_simulation()
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Mean raw fidelity: {results['fidelity_results']['mean_raw_fidelity']:.6f}")
    print(f"Mean fidelity (θ-corrected): {results['fidelity_results']['mean_fidelity_with_correction']:.6f}")
    print(f"Infidelity: {1 - results['fidelity_results']['mean_fidelity_with_correction']:.2e}")
    
    # Generate markdown report
    report = generate_markdown_report(results)
    
    # Save to docs folder
    output_path = "docs/simulation_results.md"
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_path}")
    
    # Also save raw JSON data
    json_path = "docs/simulation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Raw data saved to: {json_path}")
