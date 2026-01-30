"""Example: run a CZ gate simulation using the full error model.

This script reproduces the CZ fidelity calculation from the research notebook.
It requires JAX and ARC to be installed.
"""

import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from ryd_gate.full_error_model import jax_atom_Evolution

model = jax_atom_Evolution()

Omega = model.rabi_eff / (2 * jnp.pi)

tf = 1.219096
A = 0.1122
omegaf = 1.0431
phi0 = -0.72565603
deltaf = 0

amp_420 = lambda t: 1
phase_420 = lambda t: 2 * jnp.pi * (A * jnp.cos(2 * jnp.pi * omegaf * Omega * t - phi0) + deltaf * Omega * t)
amp_1013 = lambda t: 1

tlist = jnp.linspace(0, tf / Omega, 2)

psi0_list = model.SSS_initial_state_list
sol = model.integrate_rho_multi_jax(
    tlist,
    amp_420,
    phase_420,
    amp_1013,
    [model.psi_to_rho(psi0) for psi0 in psi0_list],
)

sol_mid = jnp.array([model.mid_state_decay(sol[n, -1]) for n in range(12)])

fid_raw_mean = 0
theta_mean = 0
for n in range(12):
    fid_raw, theta = model.CZ_fidelity(sol_mid[n], psi0_list[n])
    fid_raw_mean += fid_raw / 12
    if n in [0, 1, 2, 3, 6, 7, 8, 9]:
        theta_mean += theta / 8

fid_mean = 0
for n in range(12):
    fid, _ = model.CZ_fidelity(sol_mid[n], psi0_list[n], theta_mean)
    fid_mean += fid / 12

print(f"Mean raw fidelity: {fid_raw_mean:.6f}")
print(f"Mean fidelity (mean theta): {fid_mean:.6f}")
