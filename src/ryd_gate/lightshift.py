"""AC Stark shift calculations for Rydberg two-photon excitation."""

import numpy as np
from arc import Rubidium87
from arc.wigner import Wigner6j
from scipy.constants import c, e, h, hbar, pi

a0 = 5.29177210903e-11  # Bohr radius in meters


def calculate_polarizabilities(F_ground, omega_laser_freq, atom_obj, max_n=20):
    """Calculate scalar, vector, and tensor polarizabilities for a given
    hyperfine ground state.

    Parameters
    ----------
    F_ground : int
        Total angular momentum quantum number of the ground state.
    omega_laser_freq : float
        Angular frequency of the laser (rad/s).
    atom_obj : arc.Rubidium87
        ARC atom object.
    max_n : int, optional
        Maximum principal quantum number to sum over.

    Returns
    -------
    dict
        Keys ``'scalar'``, ``'vector'``, ``'tensor'``.
    """
    alpha0_total = 0.0
    alpha1_total = 0.0
    alpha2_total = 0.0

    I = atom_obj.I
    n_g, l_g, j_g = 5, 0, 0.5

    for n_e in range(5, max_n):
        for l_e in [1]:
            for j_e in [0.5, 1.5]:
                try:
                    red_me_atomic = atom_obj.getReducedMatrixElementJ(n_g, l_g, j_g, n_e, l_e, j_e, s=0.5)
                except ValueError:
                    continue
                F_prime_min = int(abs(j_e - I))
                F_prime_max = int(j_e + I)

                for F_prime in range(F_prime_min, F_prime_max + 1):
                    E_g_hfs = atom_obj.getEnergy(n_g, l_g, j_g) * e
                    E_e_hfs = atom_obj.getEnergy(n_e, l_e, j_e) * e
                    omega_FpF = (E_e_hfs - E_g_hfs) / hbar

                    if np.isclose(omega_FpF**2, omega_laser_freq**2):
                        continue

                    red_me_si_sq = (2 * F_prime + 1) * (2 * j_g + 1) * (
                        Wigner6j(j_g, j_e, 1, F_prime, F_ground, I) * red_me_atomic * e * a0
                    ) ** 2
                    common_factor = red_me_si_sq / (hbar * (omega_FpF**2 - omega_laser_freq**2))

                    alpha0_total += (2 / 3.0) * omega_FpF * common_factor

                    w6j_vector = Wigner6j(1, 1, 1, F_ground, F_ground, F_prime)
                    angular_vector_prefactor = (-1) ** (F_ground + F_prime + 1) * np.sqrt(
                        6 * F_ground * (2 * F_ground + 1) / (F_ground + 1)
                    )
                    alpha1_total += angular_vector_prefactor * w6j_vector * omega_FpF * common_factor

                    if F_ground > 0.5:
                        w6j_tensor = Wigner6j(1, 1, 2, F_ground, F_ground, F_prime)
                        angular_tensor_prefactor = (-1) ** (F_ground + F_prime) * np.sqrt(
                            40 * F_ground * (2 * F_ground + 1) * (2 * F_ground - 1)
                            / (3 * (F_ground + 1) * (2 * F_ground + 3))
                        )
                        alpha2_total += angular_tensor_prefactor * w6j_tensor * omega_FpF * common_factor

    return {"scalar": alpha0_total, "vector": alpha1_total, "tensor": alpha2_total}


def calculate_stark_shift_per_E_sq(F, m_F, polarizabilities, polarization_vector):
    """Calculate AC Stark shift per unit electric-field intensity.

    Parameters
    ----------
    F : int
        Total angular momentum.
    m_F : int
        Magnetic quantum number.
    polarizabilities : dict
        Output of :func:`calculate_polarizabilities`.
    polarization_vector : list
        Spherical components ``[E_{-1}, E_0, E_{+1}]``, normalised to 1.

    Returns
    -------
    float
        Energy shift in J / (V/m)^2.
    """
    E_m1, E_0, E_1 = polarization_vector
    norm_sq = abs(E_m1) ** 2 + abs(E_0) ** 2 + abs(E_1) ** 2
    if not np.isclose(norm_sq, 1.0):
        raise ValueError("Polarization vector must be normalized to 1.")
    alpha0 = polarizabilities["scalar"]
    alpha1 = polarizabilities["vector"]
    alpha2 = polarizabilities["tensor"]
    shift_scalar = -0.5 * alpha0
    shift_vector = 0
    if F > 0:
        field_vector_term = abs(E_m1) ** 2 - abs(E_1) ** 2
        shift_vector = -0.5 * alpha1 * field_vector_term * (m_F / (2 * F))
    shift_tensor = 0
    if F > 0.5:
        field_tensor_term = (3 * abs(E_0) ** 2 - 1) / 2.0
        atomic_tensor_term = (3 * m_F**2 - F * (F + 1)) / (F * (2 * F - 1))
        shift_tensor = -0.5 * alpha2 * field_tensor_term * atomic_tensor_term
    return shift_scalar + shift_vector + shift_tensor


def get_mat_from_diff_shift(wavelength_nm, polarization, exp_diff_shift_hz, g_state, e_state):
    """Infer the Rabi frequency from an experimentally measured differential
    light shift.

    Parameters
    ----------
    wavelength_nm : float
        Laser wavelength in nm.
    polarization : str
        ``'sigma+'`` or ``'sigma-'``.
    exp_diff_shift_hz : float
        Measured differential shift between F=2 and F=1 (Hz).
    g_state : tuple
        ``(n, l, j, mj)`` of the ground state.
    e_state : tuple
        ``(n, l, j, mj)`` of the excited state.

    Returns
    -------
    float
        Rabi frequency (rad/s).
    """
    atom = Rubidium87()
    omega_laser = 2 * pi * c / (wavelength_nm * 1e-9)

    if polarization == "sigma+":
        pol_vec = [0, 0, 1]
        q = 1
    elif polarization == "sigma-":
        pol_vec = [1, 0, 0]
        q = -1
    else:
        raise ValueError("Polarization must be 'sigma+' or 'sigma-'.")

    m_F = 0
    pols_F1 = calculate_polarizabilities(1, omega_laser, atom)
    pols_F2 = calculate_polarizabilities(2, omega_laser, atom)
    shift_per_E_sq_F1 = calculate_stark_shift_per_E_sq(1, m_F, pols_F1, pol_vec)
    shift_per_E_sq_F2 = calculate_stark_shift_per_E_sq(2, m_F, pols_F2, pol_vec)
    diff_shift_per_E_sq = shift_per_E_sq_F2 - shift_per_E_sq_F1

    exp_diff_shift_joules = exp_diff_shift_hz * h
    E0_sq = exp_diff_shift_joules / diff_shift_per_E_sq
    E0 = np.sqrt(np.abs(E0_sq))

    n_g, l_g, j_g, mj_g = g_state
    n_e, l_e, j_e, mj_e = e_state
    omega_rabi = np.abs(
        E0 * atom.getDipoleMatrixElement(n_g, l_g, j_g, mj_g, n_e, l_e, j_e, mj_e, q) * e * a0 / hbar
    )
    return omega_rabi


def calculate_omega_eff(diff_shift_420_hz, diff_shift_1013_hz, Delta_hz):
    """Calculate effective two-photon Rabi frequency from measured differential
    light shifts.

    Parameters
    ----------
    diff_shift_420_hz : float
        Differential light shift from the 420 nm laser (Hz).
    diff_shift_1013_hz : float
        Differential light shift from the 1013 nm laser (Hz).
    Delta_hz : float
        Intermediate-state detuning (Hz).

    Returns
    -------
    float
        Effective Rabi frequency (rad/s).
    """
    g_state_420 = (5, 0, 0.5, -0.5)
    e_state_420 = (6, 1, 1.5, -1.5)

    g_state_1013 = (6, 1, 1.5, -1.5)
    e_state_1013 = (70, 0, 0.5, -0.5)

    omega_420 = get_mat_from_diff_shift(420, "sigma-", diff_shift_420_hz, g_state_420, e_state_420) / np.sqrt(2)
    omega_1013 = get_mat_from_diff_shift(1013, "sigma+", diff_shift_1013_hz, g_state_1013, e_state_1013)

    omega_eff = -(omega_420 * omega_1013) / (2 * Delta_hz)
    return omega_eff
