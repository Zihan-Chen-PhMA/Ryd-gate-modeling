"""JAX-based two-atom density-matrix simulator with Rydberg blockade and decay."""

# %%
import numpy as np
import matplotlib.pyplot as plt
from qutip import gates
from qutip.random_objects import rand_unitary
from scipy.linalg import block_diag
from scipy.optimize import minimize_scalar
from scipy import integrate
from collections import defaultdict
from arc import Rubidium87
from arc.wigner import Wigner6j, CG

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.experimental import sparse as jsparse
from jax.experimental.ode import odeint
from jaxopt import ScipyRootFinding

import time

from jax import config
config.update("jax_enable_x64", True)

class jax_atom_Evolution():
    """JAX-accelerated two-atom density matrix simulator for Rydberg CZ gates.
    
    Models 87Rb atoms with 10-level structure including ground states (|0⟩, |1⟩),
    intermediate 6P3/2 states (|e1⟩, |e2⟩, |e3⟩), Rydberg states (|r1⟩, |r2⟩, |rP⟩),
    and leakage states (|L0⟩, |L1⟩). Includes Rydberg blockade, spontaneous decay,
    and AC Stark shifts from 420nm and 1013nm lasers.
    """

    def __init__(self, blockade=True, ryd_decay=True, mid_decay=True, 
                 distance=3, psi0=None):
        """Initialize the two-atom Rydberg gate simulator.
        
        Parameters
        ----------
        blockade : bool
            Enable Rydberg blockade interaction (default True).
        ryd_decay : bool
            Enable Rydberg state decay via BBR and radiative channels (default True).
        mid_decay : bool
            Enable intermediate 6P3/2 state decay (default True).
        distance : float
            Interatomic distance in micrometers (default 3 μm).
        psi0 : array, optional
            Custom initial state; defaults to |11⟩.
        """
        self.atom = Rubidium87()

        self.levels = 10
        self.level_label = ['0','1','e1','e2','e3','r1','r2','rP','L0','L1']

        self.init_state()

        self.level_dict = defaultdict(list)

        for i, label in  enumerate(self.level_label):
            self.level_dict[label] = {
                'idx': i,
                'ket': self.state_list[i],
            }
        
        self.level_dict['0']['Qnum'] = [5,0,1/2,1,0]
        self.level_dict['1']['Qnum'] = [5,0,1/2,2,0]

        self.level_dict['e1']['Qnum'] = [6,1,3/2,1,-1]
        self.level_dict['e2']['Qnum'] = [6,1,3/2,2,-1]
        self.level_dict['e3']['Qnum'] = [6,1,3/2,3,-1]

        self.level_dict['r1']['Qnum'] = [70,0,1/2,-1/2]
        self.level_dict['r2']['Qnum'] = [70,0,1/2,1/2]

        self.init_blockade(distance)

        self.Delta = - 2 * jnp.pi * 9.1 * 1000  #MHz
        self.rabi_eff = 2 * jnp.pi * 5 #MHz
        self.rabi_ratio = 1.63

        self._420_amp = 1.0
        self._1013_amp = 1.0

        self.init_420_ham()
        self.init_1013_ham()

        self.init_420_light_shift()
        self.init_1013_light_shift()

        self.r2_detuning = 2 * jnp.pi * 60 #MHz
        self.delta = - self._420_lightshift_1 + self._1013_lightshift_r - (1-self._1013_amp**2)*self._1013_lightshift_1

        self.delta_eff = - (1-self._420_amp**2) * self._420_lightshift_1 + (1-self._1013_amp**2) * (self._1013_lightshift_r - self._1013_lightshift_1)

        self.cops = []
        self.cops_sparse = []

        if ryd_decay:
            self.Gamma_BBR = 1 / 410.41 #MHz
        else:
            self.Gamma_BBR = 0
        self.cops_BBR = jnp.sqrt(self.Gamma_BBR)*self.state_rP*jnp.conj(self.state_r1).T
        if ryd_decay:
            self.cops.append(self.cops_BBR)
            self.cops_sparse.append([self.Gamma_BBR, self.level_dict['r1']['idx'], self.level_dict['rP']['idx']])

        if ryd_decay:
            self.Gamma_RD = 1 / 151.55 - 1 / 410.41 #MHz
        else:
            self.Gamma_RD = 0
        self.rydberg_RD_branch_ratio()
        self.cops_RD_0 = jnp.sqrt(self.branch_ratio_0*self.Gamma_RD)*self.state_0*jnp.conj(self.state_r1).T
        self.cops_RD_1 = jnp.sqrt(self.branch_ratio_1*self.Gamma_RD)*self.state_1*jnp.conj(self.state_r1).T
        self.cops_RD_L0 = jnp.sqrt(self.branch_ratio_L0*self.Gamma_RD)*self.state_L0*jnp.conj(self.state_r1).T
        self.cops_RD_L1 = jnp.sqrt(self.branch_ratio_L1*self.Gamma_RD)*self.state_L1*jnp.conj(self.state_r1).T
        if ryd_decay:
            self.cops += [self.cops_RD_0,self.cops_RD_1,self.cops_RD_L0,self.cops_RD_L1]
            self.cops_sparse += [
                [self.branch_ratio_0*self.Gamma_RD, self.level_dict['r1']['idx'], self.level_dict['0']['idx']],
                [self.branch_ratio_1*self.Gamma_RD, self.level_dict['r1']['idx'], self.level_dict['1']['idx']],
                [self.branch_ratio_L0*self.Gamma_RD, self.level_dict['r1']['idx'], self.level_dict['L0']['idx']],
                [self.branch_ratio_L1*self.Gamma_RD, self.level_dict['r1']['idx'], self.level_dict['L1']['idx']],
            ]

        if mid_decay:
            self.Gamma_mid = 1 / 0.11 #MHz
        else:
            self.Gamma_mid = 0

        e1_branch_ratio_0, e1_branch_ratio_1, e1_branch_ratio_L0, e1_branch_ratio_L1 = self.mid_branch_ratio('e1')
        self.cops_e1_0 = jnp.sqrt(e1_branch_ratio_0*self.Gamma_mid)*self.state_0*jnp.conj(self.state_e1).T
        self.cops_e1_1 = jnp.sqrt(e1_branch_ratio_1*self.Gamma_mid)*self.state_1*jnp.conj(self.state_e1).T
        self.cops_e1_L0 = jnp.sqrt(e1_branch_ratio_L0*self.Gamma_mid)*self.state_L0*jnp.conj(self.state_e1).T
        self.cops_e1_L1 = jnp.sqrt(e1_branch_ratio_L1*self.Gamma_mid)*self.state_L1*jnp.conj(self.state_e1).T
        if mid_decay:
            self.cops += [self.cops_e1_0,self.cops_e1_1,self.cops_e1_L0,self.cops_e1_L1]
            self.cops_sparse += [
                [e1_branch_ratio_0*self.Gamma_mid, self.level_dict['e1']['idx'], self.level_dict['0']['idx']],
                [e1_branch_ratio_1*self.Gamma_mid, self.level_dict['e1']['idx'], self.level_dict['1']['idx']],
                [e1_branch_ratio_L0*self.Gamma_mid, self.level_dict['e1']['idx'], self.level_dict['L0']['idx']],
                [e1_branch_ratio_L1*self.Gamma_mid, self.level_dict['e1']['idx'], self.level_dict['L1']['idx']],
            ]

        e2_branch_ratio_0, e2_branch_ratio_1, e2_branch_ratio_L0, e2_branch_ratio_L1 = self.mid_branch_ratio('e2')
        self.cops_e2_0 = jnp.sqrt(e2_branch_ratio_0*self.Gamma_mid)*self.state_0*jnp.conj(self.state_e2).T
        self.cops_e2_1 = jnp.sqrt(e2_branch_ratio_1*self.Gamma_mid)*self.state_1*jnp.conj(self.state_e2).T
        self.cops_e2_L0 = jnp.sqrt(e2_branch_ratio_L0*self.Gamma_mid)*self.state_L0*jnp.conj(self.state_e2).T
        self.cops_e2_L1 = jnp.sqrt(e2_branch_ratio_L1*self.Gamma_mid)*self.state_L1*jnp.conj(self.state_e2).T
        if mid_decay:
            self.cops += [self.cops_e2_0,self.cops_e2_1,self.cops_e2_L0,self.cops_e2_L1]
            self.cops_sparse += [
                [e2_branch_ratio_0*self.Gamma_mid, self.level_dict['e2']['idx'], self.level_dict['0']['idx']],
                [e2_branch_ratio_1*self.Gamma_mid, self.level_dict['e2']['idx'], self.level_dict['1']['idx']],
                [e2_branch_ratio_L0*self.Gamma_mid, self.level_dict['e2']['idx'], self.level_dict['L0']['idx']],
                [e2_branch_ratio_L1*self.Gamma_mid, self.level_dict['e2']['idx'], self.level_dict['L1']['idx']],
            ]

        e3_branch_ratio_0, e3_branch_ratio_1, e3_branch_ratio_L0, e3_branch_ratio_L1 = self.mid_branch_ratio('e3')
        self.cops_e3_0 = jnp.sqrt(e3_branch_ratio_0*self.Gamma_mid)*self.state_0*jnp.conj(self.state_e3).T
        self.cops_e3_1 = jnp.sqrt(e3_branch_ratio_1*self.Gamma_mid)*self.state_1*jnp.conj(self.state_e3).T
        self.cops_e3_L0 = jnp.sqrt(e3_branch_ratio_L0*self.Gamma_mid)*self.state_L0*jnp.conj(self.state_e3).T
        self.cops_e3_L1 = jnp.sqrt(e3_branch_ratio_L1*self.Gamma_mid)*self.state_L1*jnp.conj(self.state_e3).T
        if mid_decay:
            self.cops += [self.cops_e3_0,self.cops_e3_1,self.cops_e3_L0,self.cops_e3_L1]
            self.cops_sparse += [
                [e3_branch_ratio_0*self.Gamma_mid, self.level_dict['e3']['idx'], self.level_dict['0']['idx']],
                [e3_branch_ratio_1*self.Gamma_mid, self.level_dict['e3']['idx'], self.level_dict['1']['idx']],
                [e3_branch_ratio_L0*self.Gamma_mid, self.level_dict['e3']['idx'], self.level_dict['L0']['idx']],
                [e3_branch_ratio_L1*self.Gamma_mid, self.level_dict['e3']['idx'], self.level_dict['L1']['idx']],
            ]

        self.cops_tq = [jnp.kron(jnp.eye(self.levels),c) for c in self.cops] + [jnp.kron(c,jnp.eye(self.levels)) for c in self.cops]
        self.cdagc_tq = jnp.sum(jnp.array([jnp.conj(c).T @ c for c in self.cops_tq]), axis=0)

        self.cops_tq_sparse = [[item[0], item[1]*self.levels + k, item[2]*self.levels + k] for item in self.cops_sparse for k in range(10)] +\
            [[item[0], item[1] + k*self.levels, item[2] + k*self.levels] for item in self.cops_sparse for k in range(10)]
        self.cops_mid_tq_sparse = [[item[0], item[1]*self.levels + k, item[2]*self.levels + k] for item in self.cops_sparse if item[1] in [2,3,4] for k in range(10)] +\
            [[item[0], item[1] + k*self.levels, item[2] + k*self.levels] for item in self.cops_sparse if item[1] in [2,3,4] for k in range(10)]

        self.init_diag_ham(blockade)

        self.init_hamiltonian_sparse()

        if psi0 is None:
            self.psi0 = jnp.kron(self.state_1,self.state_1)
        else:
            self.psi0 = psi0

        self.rho0 = self.psi0*jnp.conj(self.psi0).T

        self.init_SSS_states()
    
    def init_state0(self, psi0):
        """Set a custom initial state and compute corresponding density matrix.
        
        Parameters
        ----------
        psi0 : array
            Initial pure state vector (100,1) for two-atom system.
        """
        self.psi0 = psi0
        self.rho0 = self.psi0*jnp.conj(self.psi0).T

    def psi_to_rho(self, psi):
        """Convert a pure state vector to a density matrix.
        
        Parameters
        ----------
        psi : array
            Pure state vector.
            
        Returns
        -------
        array
            Density matrix ρ = |ψ⟩⟨ψ|.
        """
        return psi @ jnp.conj(psi).T

    def init_state(self):
        """Initialize single-atom basis states for the 10-level system.
        
        Creates column vectors for states: |0⟩, |1⟩, |e1⟩, |e2⟩, |e3⟩,
        |r1⟩, |r2⟩, |rP⟩, |L0⟩, |L1⟩.
        """
        identity = jnp.eye(self.levels,dtype=jnp.complex128)
        self.state_list = []

        self.state_0 = identity[0].reshape(-1,1)
        self.state_list.append(self.state_0)
        self.state_1 = identity[1].reshape(-1,1)
        self.state_list.append(self.state_1)
        self.state_e1 = identity[2].reshape(-1,1)
        self.state_list.append(self.state_e1)
        self.state_e2 = identity[3].reshape(-1,1)
        self.state_list.append(self.state_e2)
        self.state_e3 = identity[4].reshape(-1,1)
        self.state_list.append(self.state_e3)
        self.state_r1 = identity[5].reshape(-1,1)
        self.state_list.append(self.state_r1)
        self.state_r2 = identity[6].reshape(-1,1)
        self.state_list.append(self.state_r2)
        self.state_rP = identity[7].reshape(-1,1)
        self.state_list.append(self.state_rP)
        self.state_L0 = identity[8].reshape(-1,1)
        self.state_list.append(self.state_L0)
        self.state_L1 = identity[9].reshape(-1,1)
        self.state_list.append(self.state_L1)
    
    def init_blockade(self, distance):
        """Initialize Rydberg-Rydberg van der Waals blockade interaction.
        
        Computes blockade shift V = C6/d^6 and constructs the two-atom
        blockade Hamiltonian H_blockade = V |rr⟩⟨rr|.
        
        Parameters
        ----------
        distance : float
            Interatomic separation in micrometers.
        """
        self.C6 = 2 * jnp.pi * 874 * 1000 #MHz*um^6
        self.default_d = distance #um
        self.d = self.default_d
        self.V = self.C6 / self.d ** 6

        self._0_opt = (self.state_0 * jnp.conj(self.state_0).T +
                      self.state_L0 * jnp.conj(self.state_L0).T)
        self._1_opt = (self.state_0 * jnp.conj(self.state_1).T +
                      self.state_L0 * jnp.conj(self.state_L1).T)

        self.ryd_opt = (self.state_r1 * jnp.conj(self.state_r1).T +
                       self.state_r2 * jnp.conj(self.state_r2).T +
                       self.state_rP * jnp.conj(self.state_rP).T)

        self.Hblockade = self.V*jnp.kron(self.ryd_opt, self.ryd_opt)

    def init_diag_ham(self, blockade=True):
        """Initialize diagonal (time-independent) part of the Hamiltonian.
        
        Sets energy levels including hyperfine splitting, intermediate state
        detuning Δ, and non-Hermitian decay terms -iΓ/2.
        
        Parameters
        ----------
        blockade : bool
            Include Rydberg blockade in two-qubit Hamiltonian.
        """
        self.Hconst = jnp.zeros((self.levels,self.levels),dtype=jnp.complex128)

        # 87Rb ground state hyperfine splitting: 5S1/2 F=1 ↔ F=2
        # The exact value is 6.834682610904 GHz (defines the SI second).
        # |0⟩ (F=1) is set 6.835 GHz below |1⟩ (F=2) which serves as energy reference.
        self.Hconst = self.Hconst.at[0,0].set(- 2 * jnp.pi * 6.835 * 1000)
        self.Hconst = self.Hconst.at[2,2].set(- self.Delta - 2 * jnp.pi * 51 - 1j*self.Gamma_mid/2) #MHz
        self.Hconst = self.Hconst.at[3,3].set(- self.Delta - 1j*self.Gamma_mid/2) #MHz
        self.Hconst = self.Hconst.at[4,4].set(- self.Delta + 2 * jnp.pi * 87 - 1j*self.Gamma_mid/2) #MHz
        self.Hconst = self.Hconst.at[5,5].set(- self.delta - 1j*(self.Gamma_BBR + self.Gamma_RD)/2)
        self.Hconst = self.Hconst.at[6,6].set(- self.delta + self.r2_detuning)

        if blockade:
            self.Hconst_tq = jnp.kron(jnp.eye(self.levels), self.Hconst) + jnp.kron(self.Hconst, jnp.eye(self.levels)) + self.Hblockade
        else:
            self.Hconst_tq = jnp.kron(jnp.eye(self.levels), self.Hconst) + jnp.kron(self.Hconst, jnp.eye(self.levels))

    def init_420_ham(self):
        """Initialize 420nm laser coupling Hamiltonian.
        
        Constructs the coupling from ground states |0⟩,|1⟩ to intermediate
        6P3/2 states |e1⟩,|e2⟩,|e3⟩ using Clebsch-Gordan coefficients
        for σ- polarization.
        """
        self.rabi_420 = jnp.sqrt(2 * jnp.abs(self.Delta) * self.rabi_eff)*self.rabi_ratio#2*jnp.pi*237#
        
        # Dipole matrix element ratio for σ⁻ transitions from |1⟩ (5S1/2, F=2, mF=0) to 6P3/2:
        #
        #                     6P3/2
        #                    ┌─────────┐
        #       (weaker)     │ mJ=-1/2 │  ← secondary transition (numerator)
        #         ↗          └─────────┘
        #   |1⟩ ─┤   σ⁻
        #         ↘          ┌─────────┐
        #       (stronger)   │ mJ=-3/2 │  ← primary "stretched" transition (denominator)
        #                    └─────────┘
        #
        # This ratio (determined by Clebsch-Gordan coefficients) quantifies the relative
        # strength of the secondary mJ=-1/2 coupling vs the primary mJ=-3/2 coupling.
        self.d_mid_ratio = (self.atom.getDipoleMatrixElementHFStoFS(5,0,1/2,2,0,
                                                                    6,1,3/2,-1/2,
                                                                    -1)/
                            self.atom.getDipoleMatrixElementHFStoFS(5,0,1/2,2,0,
                                                                    6,1,3/2,-3/2,
                                                                    -1))
        # Rabi frequency for the secondary (unwanted) transition pathway
        self.rabi_420_garbage = self.rabi_420*self.d_mid_ratio

        self.H_420 = jnp.zeros((self.levels,self.levels),dtype=jnp.complex128)

        # ═══════════════════════════════════════════════════════════════════════════
        # 420nm transitions: |1⟩ (5S1/2, F=2, mF=0) → |e1⟩,|e2⟩,|e3⟩ (6P3/2)
        # ═══════════════════════════════════════════════════════════════════════════
        #
        #   6P3/2 ──┬── |e3⟩ F=3,mF=-1  ←─┬─ CG(3/2,-3/2,3/2,1/2,3,-1) = √(1/5)
        #           │                      │  CG(3/2,-1/2,3/2,-1/2,3,-1) = √(3/5)
        #           ├── |e2⟩ F=2,mF=-1  ←─┼─ CG(3/2,-3/2,3/2,1/2,2,-1) = √(1/2)
        #           │                      │  CG(3/2,-1/2,3/2,-1/2,2,-1) = 0 (!)
        #           └── |e1⟩ F=1,mF=-1  ←─┼─ CG(3/2,-3/2,3/2,1/2,1,-1) = √(3/10)
        #                     ↑            │  CG(3/2,-1/2,3/2,-1/2,1,-1) = √(2/5)
        #                     │ 420nm σ⁻   │
        #   5S1/2 ───── |1⟩ F=2,mF=0 ──────┘
        #
        # Coupling strength = (Ω_420 × CG_primary + Ω_420_garbage × CG_secondary) / 2
        # ───────────────────────────────────────────────────────────────────────────
        self.H_420 = self.H_420.at[2,1].set(
            (self.rabi_420*
                CG(3/2,-3/2,3/2,1/2,1,-1)
            + self.rabi_420_garbage*
                CG(3/2,-1/2,3/2,-1/2,1,-1)
            )/2
        )
        self.H_420 = self.H_420.at[3,1].set(
            (self.rabi_420*
                CG(3/2,-3/2,3/2,1/2,2,-1)
            + self.rabi_420_garbage*
                CG(3/2,-1/2,3/2,-1/2,2,-1)
            )/2
        )
        self.H_420 = self.H_420.at[4,1].set(
            (self.rabi_420*
                CG(3/2,-3/2,3/2,1/2,3,-1)
            + self.rabi_420_garbage*
                CG(3/2,-1/2,3/2,-1/2,3,-1)
            )/2
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # 420nm transitions: |0⟩ (5S1/2, F=1, mF=0) → |e1⟩,|e2⟩,|e3⟩ (6P3/2)
        # ═══════════════════════════════════════════════════════════════════════════
        #
        # The OPPOSITE SIGN on rabi_420 comes from hyperfine state decomposition:
        #
        #   Hyperfine states decomposed into uncoupled basis |mJ, mI⟩:
        #   ┌─────────────────────────────────────────────────────────────────────┐
        #   │ |F=2,mF=0⟩ = +|mJ=+½,mI=-½⟩ + |mJ=-½,mI=+½⟩  (same sign)        │
        #   │ |F=1,mF=0⟩ = +|mJ=+½,mI=-½⟩ - |mJ=-½,mI=+½⟩  (opposite sign!) │
        #   └─────────────────────────────────────────────────────────────────────┘
        #
        #   The F=1 and F=2 states must be orthogonal, so the CG coefficients
        #   for the |mJ=-½,mI=+½⟩ component have OPPOSITE relative signs.
        #
        #   When σ⁻ light couples primarily to mJ=-½ → mJ'=-3/2 transition,
        #   this sign difference propagates to the effective Rabi frequency:
        #
        #   H[e,1] = (+Ω_420×CG_p + Ω_garb×CG_s)/2   ← from |F=2,mF=0⟩
        #   H[e,0] = (-Ω_420×CG_p + Ω_garb×CG_s)/2   ← from |F=1,mF=0⟩ (sign flip!)
        #
        # This is purely from angular momentum algebra, NOT from standing-wave geometry.
        # ───────────────────────────────────────────────────────────────────────────
        self.H_420 = self.H_420.at[2,0].set(
            (-self.rabi_420*
                CG(3/2,-3/2,3/2,1/2,1,-1) + 
            self.rabi_420_garbage*
                CG(3/2,-1/2,3/2,-1/2,1,-1)
            )/2
        )
        self.H_420 = self.H_420.at[3,0].set(
            (-self.rabi_420*
                CG(3/2,-3/2,3/2,1/2,2,-1) + 
            self.rabi_420_garbage*
                CG(3/2,-1/2,3/2,-1/2,2,-1)
            )/2
        )
        self.H_420 = self.H_420.at[4,0].set(
            (-self.rabi_420*
                CG(3/2,-3/2,3/2,1/2,3,-1) + 
            self.rabi_420_garbage*
                CG(3/2,-1/2,3/2,-1/2,3,-1)
            )/2
        )

        self.H_420_tq = jnp.kron(jnp.eye(self.levels), self.H_420) + jnp.kron(self.H_420, jnp.eye(self.levels))
        self.H_420_tq_conj = jnp.conj(self.H_420_tq).T
    
    def init_1013_ham(self):
        """Initialize 1013nm laser coupling Hamiltonian.
        
        Constructs the coupling from intermediate 6P3/2 states to Rydberg
        70S1/2 states |r1⟩,|r2⟩ using Clebsch-Gordan coefficients
        for σ+ polarization.
        """
        self.rabi_1013 = jnp.sqrt(2 * jnp.abs(self.Delta) * self.rabi_eff)/self.rabi_ratio#2*jnp.pi*303#
        self.d_ryd_ratio = (self.atom.getDipoleMatrixElement(6,1,3/2,-1/2,
                                                             *self.level_dict['r2']['Qnum'],
                                                             1)/
                            self.atom.getDipoleMatrixElement(6,1,3/2,-3/2,
                                                             *self.level_dict['r1']['Qnum'],
                                                             1))
        self.rabi_1013_garbage = self.rabi_1013*self.d_ryd_ratio

        self.H_1013 = jnp.zeros((self.levels,self.levels),dtype=jnp.complex128)
   
        # ═══════════════════════════════════════════════════════════════════════════
        # 1013nm PRIMARY transitions: |e1⟩,|e2⟩,|e3⟩ (6P3/2) → |r1⟩ (70S1/2, mJ=-1/2)
        # ═══════════════════════════════════════════════════════════════════════════
        #
        #   70S1/2 ───── |r1⟩ mJ=-1/2 ←────┬── from |e1⟩: CG(3/2,-3/2,3/2,1/2,1,-1)
        #                     ↑             ├── from |e2⟩: CG(3/2,-3/2,3/2,1/2,2,-1)
        #                     │ 1013nm σ⁺   └── from |e3⟩: CG(3/2,-3/2,3/2,1/2,3,-1)
        #                     │
        #   6P3/2 ──┬── |e3⟩ F=3,mF=-1     CG values: √(1/5), √(1/2), √(3/10)
        #           ├── |e2⟩ F=2,mF=-1     (same CG as 420nm, different physical meaning:
        #           └── |e1⟩ F=1,mF=-1      here coupling 6P3/2→70S1/2 via mJ projection)
        #
        # Coupling strength = Ω_1013 × CG / 2
        # ───────────────────────────────────────────────────────────────────────────
        self.H_1013 = self.H_1013.at[5,2].set(
            (self.rabi_1013/2)*CG(3/2,-3/2,3/2,1/2,1,-1)
        )
        self.H_1013 = self.H_1013.at[5,3].set(
            (self.rabi_1013/2)*CG(3/2,-3/2,3/2,1/2,2,-1)
        )
        self.H_1013 = self.H_1013.at[5,4].set(
            (self.rabi_1013/2)*CG(3/2,-3/2,3/2,1/2,3,-1)
        )
    
        # ═══════════════════════════════════════════════════════════════════════════
        # 1013nm SECONDARY transitions: |e1⟩,|e2⟩,|e3⟩ (6P3/2) → |r2⟩ (70S1/2, mJ=+1/2)
        # ═══════════════════════════════════════════════════════════════════════════
        #
        #   70S1/2 ───── |r2⟩ mJ=+1/2 ←────┬── from |e1⟩: CG(3/2,-1/2,3/2,-1/2,1,-1)=√(2/5)
        #                     ↑             ├── from |e2⟩: CG(3/2,-1/2,3/2,-1/2,2,-1)= 0 (!)
        #                     │ 1013nm σ⁺   └── from |e3⟩: CG(3/2,-1/2,3/2,-1/2,3,-1)=√(3/5)
        #                     │
        #   6P3/2 ──┬── |e3⟩ F=3,mF=-1     Note: |e2⟩→|r2⟩ coupling is ZERO!
        #           ├── |e2⟩ F=2,mF=-1     This is an exact symmetry selection rule.
        #           └── |e1⟩ F=1,mF=-1
        #
        # This is the "garbage" Rydberg state that leads to gate errors.
        # Coupling strength = Ω_1013_garbage × CG / 2
        # ───────────────────────────────────────────────────────────────────────────
        self.H_1013 = self.H_1013.at[6,2].set(
            (self.rabi_1013_garbage/2)*CG(3/2,-1/2,3/2,-1/2,1,-1)
        )
        self.H_1013 = self.H_1013.at[6,3].set(
            (self.rabi_1013_garbage/2)*CG(3/2,-1/2,3/2,-1/2,2,-1)
        )
        self.H_1013 = self.H_1013.at[6,4].set(
            (self.rabi_1013_garbage/2)*CG(3/2,-1/2,3/2,-1/2,3,-1)
        )

        self.H_1013_tq = jnp.kron(jnp.eye(self.levels), self.H_1013) + jnp.kron(self.H_1013, jnp.eye(self.levels))
        self.H_1013_tq_conj = jnp.conj(self.H_1013_tq).T

        self.H_1013_tq_hermi = self.H_1013_tq + self.H_1013_tq_conj
    
    def hamiltonian(self, t, args):
        """Compute the full time-dependent Hamiltonian at time t.
        
        Parameters
        ----------
        t : float
            Time in microseconds.
        args : dict
            Dictionary with keys "amp_420", "phase_420", "amp_1013" mapping
            to callable pulse functions.
            
        Returns
        -------
        array
            100x100 complex Hamiltonian matrix for two-atom system.
        """
        amp_420 = args["amp_420"](t)
        phase_420 = args["phase_420"](t)
        amp_1013 = args["amp_1013"](t)

        Hamitonian = (self.Hconst_tq + 
                      amp_420 * (jnp.exp(-1j*phase_420) * self.H_420_tq + jnp.exp(1j*phase_420) * self.H_420_tq_conj) +
                      amp_1013 * self.H_1013_tq_hermi )#- self.cdagc_tq * 1j * 0.5)
        
        return Hamitonian

    def init_hamiltonian_sparse(self):
        """Initialize sparse representation of Hamiltonian for efficient evolution.
        
        Extracts diagonal and off-diagonal elements separately for optimized
        matrix-vector products during ODE integration.
        """
        self.H_diag_tq_sparse = jnp.diag(self.Hconst_tq)
        self.H_diag_tq_sparse_conj = jnp.conj(self.H_diag_tq_sparse)

        row_idx, col_idx = jnp.where(self.H_420_tq)
        self.H_420_tq_sparse_value = jnp.array(self.H_420_tq[row_idx, col_idx])

        self.H_420_tq_sparse_idx = jnp.stack([row_idx, col_idx],axis=1)

        row_idx, col_idx = jnp.where(self.H_1013_tq)
        self.H_1013_tq_sparse_value = jnp.array(self.H_1013_tq[row_idx, col_idx])

        self.H_1013_tq_sparse_idx = jnp.stack([row_idx, col_idx],axis=1)

        self.H_offdiag_tq_sparse_idx = jnp.concatenate([self.H_420_tq_sparse_idx, self.H_1013_tq_sparse_idx], axis=0)

    def hamiltonian_sparse(self, t, args):
        """Compute sparse off-diagonal Hamiltonian elements at time t.
        
        Parameters
        ----------
        t : float
            Time in microseconds.
        args : dict
            Pulse function dictionary.
            
        Returns
        -------
        array
            Sparse representation of off-diagonal coupling terms.
        """
        amp_420 = args["amp_420"](t)
        phase_420 = args["phase_420"](t)
        amp_1013 = args["amp_1013"](t)

        Hamitonian_sparse = jnp.concatenate([
            amp_420 * jnp.exp(-1j*phase_420) * self.H_420_tq_sparse_value,
            amp_1013 * self.H_1013_tq_sparse_value
        ])

        return Hamitonian_sparse

    
    def integrate_rho_jax(self, tlist, amp_420, phase_420, amp_1013, rho0=None):
        """Integrate Lindblad master equation using JAX-accelerated ODE solver.
        
        Parameters
        ----------
        tlist : array
            Time points for evolution (μs).
        amp_420, phase_420, amp_1013 : callable
            Pulse amplitude/phase functions of time.
        rho0 : array, optional
            Initial density matrix; defaults to self.rho0.
            
        Returns
        -------
        array
            Density matrices at each time point, shape (len(tlist), 100, 100).
        """
        if rho0 is None:
            rho0_flat = self.rho0.reshape((-1,))
        else:
            rho0_flat = rho0.reshape((-1,))

        args = {
            "amp_420": amp_420,
            "phase_420": phase_420,
            "amp_1013": amp_1013
        }

        # H_fun = lambda t:self.hamiltonian(t,args)
        H_fun = lambda t:self.hamiltonian_sparse(t,args)

        def rhs(rho_flat, t, *args):
            rho = rho_flat.reshape((self.levels**2, self.levels**2))
            Ht = H_fun(t)

            # Ht_dag = jnp.conj(Ht).T
            # comm = -1j * (Ht @ rho - rho @ Ht_dag)
            # drho = comm
            # dissipator = jnp.zeros_like(rho)
            # for c in self.cops_tq:
            #     c_rho_cdag = c @ rho @ jnp.conj(c).T
            #     dissipator = dissipator + c_rho_cdag
            # drho = comm + dissipator

            drho = -1j * (self.H_diag_tq_sparse[:, None] - self.H_diag_tq_sparse_conj[None, :]) * rho

            p = self.H_offdiag_tq_sparse_idx[:, 0]
            q = self.H_offdiag_tq_sparse_idx[:, 1]

            drho = drho.at[p, :].add(-1j * Ht[:, None] * rho[q, :])
            drho = drho.at[q, :].add(-1j * jnp.conj(Ht)[:, None] * rho[p, :])
            drho = drho.at[:, p].add(1j * jnp.conj(Ht)[None, :] * rho[:, q])
            drho = drho.at[:, q].add(1j * Ht[None, :] * rho[:, p])

            for gamma, j, i in self.cops_sparse:
                drho = drho.at[i::10,i::10].add(gamma * rho[j::10,j::10])
                drho = drho.at[(i*10):(10+i*10),(i*10):(10+i*10)].add(gamma * rho[(j*10):(10+j*10),(j*10):(10+j*10)])

            drho = 0.5 * (drho + jnp.conj(drho.T))

            return drho.reshape((-1,))

        @jax.jit
        def run_ode(rho0_flat, tlist):
            sol = odeint(rhs, rho0_flat, tlist, *(), rtol=1e-10, atol=1e-10)
            return sol
        
        sol_flat = run_ode(rho0_flat, tlist)
        sol = sol_flat.reshape((len(tlist), self.levels**2, self.levels**2))
        return sol
    
    def integrate_rho_multi_jax(self, tlist, amp_420, phase_420, amp_1013, rho0_list=None):
        """Integrate multiple initial states in parallel using JAX vmap.
        
        Parameters
        ----------
        tlist : array
            Time points for evolution (μs).
        amp_420, phase_420, amp_1013 : callable
            Pulse amplitude/phase functions.
        rho0_list : list of arrays, optional
            List of initial density matrices for batch evolution.
            
        Returns
        -------
        array
            Batch of density matrices, shape (batch, len(tlist), 100, 100).
        """
        if rho0_list is None:
            rho0_batch = jnp.stack([self.rho0.reshape((-1,))])
            batch = 1
        else:
            rho0_batch = jnp.stack([
                rho0.reshape((-1,)) for rho0 in rho0_list
            ])
            batch = len(rho0_list)

        args = {
            "amp_420": amp_420,
            "phase_420": phase_420,
            "amp_1013": amp_1013
        }

        # H_fun = lambda t:self.hamiltonian(t,args)
        H_fun = lambda t:self.hamiltonian_sparse(t,args)

        def rhs(rho_flat, t, *args):
            rho = rho_flat.reshape((self.levels**2, self.levels**2))
            Ht = H_fun(t)

            # Ht_dag = jnp.conj(Ht).T
            # comm = -1j * (Ht @ rho - rho @ Ht_dag)
            # drho = comm
            # dissipator = jnp.zeros_like(rho)
            # for c in self.cops_tq:
            #     c_rho_cdag = c @ rho @ jnp.conj(c).T
            #     dissipator = dissipator + c_rho_cdag
            # drho = comm + dissipator

            drho = -1j * (self.H_diag_tq_sparse[:, None] - self.H_diag_tq_sparse_conj[None, :]) * rho

            p = self.H_offdiag_tq_sparse_idx[:, 0]
            q = self.H_offdiag_tq_sparse_idx[:, 1]

            drho = drho.at[p, :].add(-1j * Ht[:, None] * rho[q, :])
            drho = drho.at[q, :].add(-1j * jnp.conj(Ht)[:, None] * rho[p, :])
            drho = drho.at[:, p].add(1j * jnp.conj(Ht)[None, :] * rho[:, q])
            drho = drho.at[:, q].add(1j * Ht[None, :] * rho[:, p])

            for gamma, j, i in self.cops_sparse:
                drho = drho.at[i::10,i::10].add(gamma * rho[j::10,j::10])
                drho = drho.at[(i*10):(10+i*10),(i*10):(10+i*10)].add(gamma * rho[(j*10):(10+j*10),(j*10):(10+j*10)])

            drho = 0.5 * (drho + jnp.conj(drho.T))

            return drho.reshape((-1,))

        @jax.jit
        def run_ode(rho0_flat, tlist):
            sol = odeint(rhs, rho0_flat, tlist, *(), atol=1e-10, rtol=1e-10)
            return sol
        
        run_ode_batch = jax.jit(vmap(run_ode, in_axes=(0,None)))
        
        sol_flat = run_ode_batch(rho0_batch,tlist)
        sol = sol_flat.reshape((batch, len(tlist), self.levels**2, self.levels**2))
        return sol
    
    def integrate_rho_multi2_jax(self, tlist, amp_420, phase_420, amp_1013, rho0_list=None):
        """Integrate multiple initial states as concatenated batch (alternative method).
        
        Parameters
        ----------
        tlist : array
            Time points for evolution (μs).
        amp_420, phase_420, amp_1013 : callable
            Pulse amplitude/phase functions.
        rho0_list : list of arrays, optional
            List of initial density matrices.
            
        Returns
        -------
        array
            Density matrices, shape (len(tlist), batch, 100, 100).
        """
        if rho0_list is None:
            rho0_batch = jnp.concatenate([self.rho0.reshape((-1,))])
            batch = 1
        else:
            rho0_batch = jnp.concatenate([
                rho0.reshape((-1,)) for rho0 in rho0_list
            ])
            batch = len(rho0_list)

        args = {
            "amp_420": amp_420,
            "phase_420": phase_420,
            "amp_1013": amp_1013
        }

        # H_fun = lambda t:self.hamiltonian(t,args)
        H_fun = lambda t:self.hamiltonian_sparse(t,args)

        def rhs(rho_flat, t, *args):
            rho = rho_flat.reshape((batch, self.levels**2, self.levels**2))
            Ht = H_fun(t)

            # Ht_dag = jnp.conj(Ht).T
            # comm = -1j * (Ht @ rho - rho @ Ht_dag)
            # drho = comm
            # dissipator = jnp.zeros_like(rho)
            # for c in self.cops_tq:
            #     c_rho_cdag = c @ rho @ jnp.conj(c).T
            #     dissipator = dissipator + c_rho_cdag
            # drho = comm + dissipator

            drho = -1j * (self.H_diag_tq_sparse[None, :, None] - self.H_diag_tq_sparse_conj[None, None, :]) * rho

            p = self.H_offdiag_tq_sparse_idx[:, 0]
            q = self.H_offdiag_tq_sparse_idx[:, 1]

            drho = drho.at[:, p, :].add(-1j * Ht[None, :, None] * rho[:, q, :])
            drho = drho.at[:, q, :].add(-1j * jnp.conj(Ht)[None, :, None] * rho[:, p, :])
            drho = drho.at[:, :, p].add(1j * jnp.conj(Ht)[None, None, :] * rho[:, :, q])
            drho = drho.at[:, :, q].add(1j * Ht[None, None, :] * rho[:, :, p])

            for gamma, j, i in self.cops_sparse:
                drho = drho.at[:, i::10,i::10].add(gamma * rho[:, j::10,j::10])
                drho = drho.at[:, (i*10):(10+i*10),(i*10):(10+i*10)].add(gamma * rho[:, (j*10):(10+j*10),(j*10):(10+j*10)])

            drho = 0.5 * (drho + jnp.conj(drho.T))

            return drho.reshape((-1,))

        @jax.jit
        def run_ode(rho0_flat, tlist):
            sol = odeint(rhs, rho0_flat, tlist, *())
            return sol
        
        sol_flat = run_ode(rho0_batch, tlist)
        sol = sol_flat.reshape((len(tlist), batch, self.levels**2, self.levels**2))
        return sol

    # def integrate_psi_jax(self, tlist, amp_420, phase_420, amp_1013, psi0 = None):
    #     if psi0 is None:
    #         psi0_flat = self.psi0.reshape((-1,))
    #     else:
    #         psi0_flat = psi0.reshape((-1,))

    #     args = {
    #         "amp_420": amp_420,
    #         "phase_420": phase_420,
    #         "amp_1013": amp_1013
    #     }

    #     H_fun = lambda t:self.hamiltonian(t,args)

    #     def rhs(psi_flat, t, *args):
    #         psi = psi_flat.reshape((-1,1))
    #         Ht = H_fun(t)

    #         dpsi = -1j* (Ht @ psi)
        
    #         return dpsi.reshape((-1,))
    
    #     @jax.jit
    #     def run_ode(psi0_flat, tlist):
    #         sol = odeint(rhs, psi0_flat, tlist, *(), atol=1e-16)
    #         return sol
        
    #     sol_flat = run_ode(psi0_flat, tlist)
    #     sol = sol_flat.reshape((len(tlist), self.levels**2))
    #     return sol

    def mid_state_decay(self, rho0=None):
        """Apply instantaneous decay of intermediate 6P3/2 states to ground states.
        
        Useful for modeling fast intermediate state decay between gate operations.
        Redistributes population according to branching ratios.
        
        Parameters
        ----------
        rho0 : array, optional
            Input density matrix; defaults to self.rho0.
            
        Returns
        -------
        array
            Density matrix with intermediate state population decayed.
        """
        if rho0 is None:
            rho0 = self.rho0
        else:
            rho0 = rho0

        double_e = [22, 23, 24, 32, 33, 34, 42, 43, 44]

        for idx in double_e:
            gamma = jnp.array([c[0] for c in self.cops_mid_tq_sparse if c[1] == idx])
            j_idx = [c[2] for c in self.cops_mid_tq_sparse if c[1] == idx]

            gamma_sum = jnp.sum(gamma)

            rho0 = rho0.at[j_idx, j_idx].add(gamma * rho0[idx, idx] / gamma_sum)
            rho0 = rho0.at[idx, :].set(0)
            rho0 = rho0.at[:, idx].set(0)
        
        for idx in [2,3,4]:
            gamma = jnp.array([c[0] for c in self.cops_sparse if c[1] == idx])
            j_idx = [c[2] for c in self.cops_sparse if c[1] == idx]

            gamma_sum = jnp.sum(gamma)

            for gam, j in zip(gamma, j_idx):
                rho0 = rho0.at[j::10,j::10].add(gam * rho0[idx::10,idx::10] / gamma_sum)
                rho0 = rho0.at[(j*10):(10+j*10),(j*10):(10+j*10)].add(gam * rho0[(idx*10):(10+idx*10),(idx*10):(10+idx*10)] / gamma_sum)

            rho0 = rho0.at[idx::10, :].set(0)
            rho0 = rho0.at[:, idx::10].set(0)
            rho0 = rho0.at[(idx*10):(10+idx*10), :].set(0)
            rho0 = rho0.at[:, (idx*10):(10+idx*10)].set(0)

        return rho0
    
    def init_420_light_shift(self):
        """Calculate AC Stark shifts from the 420nm laser on ground states.
        
        Computes differential light shift between |0⟩ and |1⟩ states due to
        off-resonant coupling to 6P3/2 intermediate states.
        """
        # _420_lightshift_1 = 4/3 * self.rabi_420**2 / (4*self.Delta)
        _420_lightshift_1 = ((self.rabi_420*
                                CG(3/2,-3/2,3/2,1/2,1,-1) + 
                            self.rabi_420_garbage*
                                CG(3/2,-1/2,3/2,-1/2,1,-1)
                            )**2 / (4*(self.Delta + 2 * jnp.pi * 51))+
                            (self.rabi_420*
                                CG(3/2,-3/2,3/2,1/2,2,-1) + 
                            self.rabi_420_garbage*
                                CG(3/2,-1/2,3/2,-1/2,2,-1)
                            )**2 / (4*(self.Delta)) +
                            (self.rabi_420*
                                CG(3/2,-3/2,3/2,1/2,3,-1) + 
                            self.rabi_420_garbage*
                                CG(3/2,-1/2,3/2,-1/2,3,-1)
                            )**2 / (4*(self.Delta - 2 * jnp.pi * 87)))

        _420_lightshift_0 = ((-self.rabi_420*
                                CG(3/2,-3/2,3/2,1/2,1,-1) + 
                            self.rabi_420_garbage*
                                CG(3/2,-1/2,3/2,-1/2,1,-1)
                            )**2 / (4*(self.Delta + 2 * jnp.pi * 51 - 2 * jnp.pi * 6.835 * 1000)) +
                            (-self.rabi_420*
                                CG(3/2,-3/2,3/2,1/2,2,-1) + 
                            self.rabi_420_garbage*
                                CG(3/2,-1/2,3/2,-1/2,2,-1)
                            )**2 / (4*(self.Delta - 2 * jnp.pi * 6.835 * 1000)) +
                            (-self.rabi_420*
                                CG(3/2,-3/2,3/2,1/2,3,-1) + 
                            self.rabi_420_garbage*
                                CG(3/2,-1/2,3/2,-1/2,3,-1)
                            )**2 / (4*(self.Delta - 2 * jnp.pi * 87 - 2 * jnp.pi * 6.835 * 1000)))# / (4*(self.Delta - 2 * jnp.pi * 6.835 * 1000))
        
        self._420_lightshift_1 = _420_lightshift_1
        self._420_lightshift_0 = _420_lightshift_0
        self._420_diff_shift = _420_lightshift_1 - _420_lightshift_0
    
    def init_1013_light_shift(self):
        """Calculate AC Stark shifts from the 1013nm laser on ground and Rydberg states.
        
        Computes light shifts on Rydberg state (via 6P3/2) and differential
        shift on ground states via off-resonant D1/D2 transitions.
        """
        # _1013_lightshift_r = self.rabi_1013**2 / (4*self.Delta)
        _1013_lightshift_r = self.rabi_1013**2 * (
            CG(3/2,-3/2,3/2,1/2,1,-1)**2 / (4*(self.Delta + 2 * jnp.pi * 51 + self._420_lightshift_1)) +
            CG(3/2,-3/2,3/2,1/2,2,-1)**2 / (4*(self.Delta + self._420_lightshift_1))+
            CG(3/2,-3/2,3/2,1/2,3,-1)**2 / (4*(self.Delta - 2 * jnp.pi * 87 + self._420_lightshift_1))
        )

        def g(_1013_lightshift_r):
            return (_1013_lightshift_r - 
                    self.rabi_1013**2 * (
                CG(3/2,-3/2,3/2,1/2,1,-1)**2 / (4*(self.Delta + 2 * jnp.pi * 51 + self._420_lightshift_1 - _1013_lightshift_r)) +
                CG(3/2,-3/2,3/2,1/2,2,-1)**2 / (4*(self.Delta + self._420_lightshift_1 - _1013_lightshift_r))+
                CG(3/2,-3/2,3/2,1/2,3,-1)**2 / (4*(self.Delta - 2 * jnp.pi * 87 + self._420_lightshift_1 - _1013_lightshift_r))
            )
            )
        
        solver = ScipyRootFinding(optimality_fun=g,method="hybr")
        res = solver.run(_1013_lightshift_r)
        _1013_lightshift_r = res.params

        self._1013_lightshift_r = _1013_lightshift_r

        _1013_freq = self.atom.getTransitionFrequency(6,1,3/2, 70,0,1/2) / 10**6 - self.Delta / (2*jnp.pi)
        D1_freq = self.atom.getTransitionFrequency(5,0,1/2, 5,1,1/2) / 10**6
        D2_freq = self.atom.getTransitionFrequency(5,0,1/2, 5,1,3/2) / 10**6

        Detuning_from_D1 = 2 * jnp.pi * (_1013_freq - D1_freq)
        Detuning_from_D2 = 2 * jnp.pi * (_1013_freq - D2_freq)

        _1013_lightshift_1 = (self.rabi_1013 / 
            self.atom.getDipoleMatrixElement(6,1,3/2,-3/2,*self.level_dict['r1']['Qnum'],1)
        )**2 * ((
            self.atom.getDipoleMatrixElementHFS(5,0,1/2,2,0  ,5,1,3/2,3,1,  1)**2 +
            self.atom.getDipoleMatrixElementHFS(5,0,1/2,2,0  ,5,1,3/2,2,1,  1)**2 + 
            self.atom.getDipoleMatrixElementHFS(5,0,1/2,2,0  ,5,1,3/2,1,1,  1)**2
        ) / (4*(Detuning_from_D2 + 2 * jnp.pi * 2.563 * 1000)) + (
            self.atom.getDipoleMatrixElementHFS(5,0,1/2,2,0,  5,1,1/2,2,1,  1)**2 + 
            self.atom.getDipoleMatrixElementHFS(5,0,1/2,2,0,  5,1,1/2,1,1,  1)**2
        ) / (4*(Detuning_from_D1 + 2 * jnp.pi * 2.563 * 1000)))

        _1013_lightshift_0 = (self.rabi_1013 / 
            self.atom.getDipoleMatrixElement(6,1,3/2,-3/2,*self.level_dict['r1']['Qnum'],1)
        )**2 * ((
            self.atom.getDipoleMatrixElementHFS(5,0,1/2,1,0,  5,1,3/2,3,1,  1)**2 +
            self.atom.getDipoleMatrixElementHFS(5,0,1/2,1,0,  5,1,3/2,2,1,  1)**2 + 
            self.atom.getDipoleMatrixElementHFS(5,0,1/2,1,0,  5,1,3/2,1,1,  1)**2
        ) / (4*(Detuning_from_D2 - 2 * jnp.pi * 4.272 * 1000)) + (
            self.atom.getDipoleMatrixElementHFS(5,0,1/2,1,0,  5,1,1/2,2,1,  1)**2 + 
            self.atom.getDipoleMatrixElementHFS(5,0,1/2,1,0,  5,1,1/2,1,1,  1)**2
        ) / (4*(Detuning_from_D1 - 2 * jnp.pi * 4.272 * 1000)))

        self._1013_diff_shift = _1013_lightshift_1 - _1013_lightshift_0
        self._1013_lightshift_1 = _1013_lightshift_1
        self._1013_lightshift_0 = _1013_lightshift_0

    def rydberg_RD_branch_ratio(self):
        """Calculate branching ratios for Rydberg radiative decay to ground states.
        
        Computes decay fractions to |0⟩, |1⟩, |L0⟩, |L1⟩ via intermediate
        5P states using Clebsch-Gordan coefficients for the 70S → 5P → 5S cascade.
        """
        I, mI = 3/2, 1/2
        nr, lr, jr, mjr = self.level_dict['r1']['Qnum']
        fr = [2,1]
        mfr = mI + mjr

        ne,le = 5,1
        je = [3/2,1/2]

        ng, lg, jg = 5, 0, 1/2

        a = []
        b = []

        for _je in je:
            fe = np.arange(np.abs(I-_je),I+_je+1,1)
            mje = np.arange(-_je,_je+1,1)
            for _fe in fe:
                mfe = np.arange(-_fe,_fe+1,1)
                for _mfe in mfe:
                    t = 0

                    for _fr in fr:
                        if np.abs(mfr) <= _fr:
                            if np.abs(mfr - _mfe) < 2:
                                t+=CG(jr,mjr,I,mI,_fr,mfr)*self.atom.getDipoleMatrixElementHFS(ne,le,_je,_fe,_mfe,nr,lr,jr,_fr,mfr,q=mfr-_mfe)
                    a.append(t**2)

                    bb = []
                    for fg in [2,1]:
                        mfg = np.arange(-fg,fg+1,1)
                        for _mfg in mfg:
                            if np.abs(_mfg - _mfe) < 2:
                                bb.append((self.atom.getDipoleMatrixElementHFS(ne,le,_je,_fe,_mfe,ng,lg,jg,fg,_mfg,q=_mfg - _mfe))**2)
                            else:
                                bb.append(0.0)
                    bb = [i/np.sum(np.array(bb)) for i in bb]
                    b.append(bb)

        a = [i/np.sum(np.array(a)) for i in a]

        branch_ratio = np.array([a[i]*np.array(b[i]) for i in range(len(a))]).sum(axis=0)

        self.branch_ratio_0 = branch_ratio[6]
        self.branch_ratio_1 = branch_ratio[2]
        self.branch_ratio_L0 = branch_ratio[5] + branch_ratio[7]
        self.branch_ratio_L1 = branch_ratio[0] + branch_ratio[1] + branch_ratio[3] + branch_ratio[4]

    def mid_branch_ratio(self, level_label):
        """Calculate branching ratios for 6P3/2 intermediate state decay.
        
        Parameters
        ----------
        level_label : str
            Intermediate state label ('e1', 'e2', or 'e3').
            
        Returns
        -------
        tuple
            Branching ratios (ratio_0, ratio_1, ratio_L0, ratio_L1) to ground states.
        """
        ne,le,je,fe,mfe = self.level_dict[level_label]['Qnum']

        ng,lg,jg = 5,0,1/2
        fg = [2,1]

        a = []
        for _fg in fg:
            mfg = np.arange(-_fg,_fg+1,1)
            for _mfg in mfg:
                if np.abs(_mfg - mfe) < 2:
                    a.append((self.atom.getDipoleMatrixElementHFS(ne,le,je,fe,mfe,ng,lg,jg,_fg,_mfg,q=_mfg - mfe))**2)
                else:
                    a.append(0.0)
        branch_ratio = [i/np.sum(np.array(a)) for i in a]

        branch_ratio_0 = branch_ratio[6]
        branch_ratio_1 = branch_ratio[2]
        branch_ratio_L0 = branch_ratio[5] + branch_ratio[7]
        branch_ratio_L1 = branch_ratio[0] + branch_ratio[1] + branch_ratio[3] + branch_ratio[4]

        return branch_ratio_0, branch_ratio_1, branch_ratio_L0, branch_ratio_L1

    def init_SSS_states(self):
        """Initialize Symmetric State Set (SSS) for gate fidelity characterization.
        
        Creates 12 specific two-qubit input states that span the computational
        subspace, enabling efficient average gate fidelity calculation.
        """
        state_00 = jnp.kron(self.state_0, self.state_0)
        state_01 = jnp.kron(self.state_0, self.state_1)
        state_10 = jnp.kron(self.state_1, self.state_0)
        state_11 = jnp.kron(self.state_1, self.state_1)

        self.SSS_initial_state_list = [
            0.5*state_00 + 0.5*state_01 + 0.5*state_10 + 0.5*state_11,
            0.5*state_00 - 0.5*state_01 - 0.5*state_10 + 0.5*state_11,
            0.5*state_00 + 0.5j*state_01 + 0.5j*state_10 - 0.5*state_11,
            0.5*state_00 - 0.5j*state_01 - 0.5j*state_10 - 0.5*state_11,
            state_00,
            state_11,
            0.5*state_00 + 0.5*state_01 + 0.5*state_10 - 0.5*state_11,
            0.5*state_00 - 0.5*state_01 - 0.5*state_10 - 0.5*state_11,
            0.5*state_00 + 0.5j*state_01 + 0.5j*state_10 + 0.5*state_11,
            0.5*state_00 - 0.5j*state_01 - 0.5j*state_10 + 0.5*state_11,
            state_00/jnp.sqrt(2) + 1j*state_11/jnp.sqrt(2),
            state_00/jnp.sqrt(2) - 1j*state_11/jnp.sqrt(2),
        ]
    
    def SSS_rec(self, idx):
        """Construct recovery unitary for SSS state fidelity measurement.
        
        Parameters
        ----------
        idx : int
            SSS state index (0-11).
            
        Returns
        -------
        array
            Recovery unitary that maps ideal CZ output back to |00⟩.
        """
        p1 = [0,0,1,1,2,0,2,0,1,0,3,1]
        p2 = [3,1,0,2,0,0,1,1,1,0,0,0]
        p3 = [0,0,0,0,0,0,1,1,1,1,1,1]

        g1 = gates.qrot(np.pi/2,p2[idx]*np.pi/2)*gates.qrot(np.pi/2,p1[idx]*np.pi/2)
        g2 = gates.qrot(-np.pi/2,0)*gates.qrot(np.pi/2,p3[idx]*np.pi/2)

        G1 = jnp.array(block_diag(g1.full(),np.eye(8)),dtype=jnp.complex128)
        G2 = jnp.array(block_diag(g2.full(),np.eye(8)),dtype=jnp.complex128)

        return jnp.kron(G2,G2) @ self.CZ_ideal() @ jnp.kron(G1,G1)

    def rand_U(self):
        """Generate a random single-qubit unitary extended to the full Hilbert space.
        
        Returns
        -------
        array
            Random two-qubit unitary acting on computational subspace.
        """
        U = jnp.array(block_diag(rand_unitary(2).full(),np.eye(8)),dtype=jnp.complex128)
        return jnp.kron(U,U)

    def CZ_ideal(self):
        """Return the ideal CZ gate matrix for the two-atom system.
        
        Returns
        -------
        array
            100x100 identity with -1 at |11⟩ position (index 11,11).
        """
        CZ = jnp.eye(self.levels**2,dtype=jnp.complex128)
        CZ = CZ.at[11,11].set(-1)
        return CZ
    
    def CZ_fidelity(self, state_final, state_initial=None, theta=None):
        """Calculate CZ gate fidelity with optional single-qubit Z-rotation correction.
        
        Parameters
        ----------
        state_final : array
            Final density matrix after gate operation.
        state_initial : array, optional
            Initial state; defaults to self.psi0.
        theta : float, optional
            Fixed Z-rotation angle; if None, optimizes over theta.
            
        Returns
        -------
        tuple
            (fidelity, optimal_theta) where fidelity is ⟨ψ_ideal|ρ_final|ψ_ideal⟩.
        """
        if state_initial is None:
            state_initial = self.psi0
        CZ_psi0 = self.CZ_ideal() @ state_initial

        if theta is None:
            def fidelity_theta(theta):
                
                U_theta = jnp.array(block_diag(gates.rz(theta).full(),np.eye(8)),dtype=jnp.complex128)
                U_theta_tq = jnp.kron(U_theta,U_theta)

                theta_CZ_psi0 = U_theta_tq @ CZ_psi0
                fid = jnp.abs(jnp.conj(theta_CZ_psi0).T @ state_final @ theta_CZ_psi0)
            
                return fid[0,0]
            
            res = minimize_scalar(lambda theta:-fidelity_theta(theta), bounds=(-jnp.pi,jnp.pi), method='bounded')

            return -res.fun, res.x

        else:
            U_theta = jnp.array(block_diag(gates.rz(theta).full(),np.eye(8)),dtype=jnp.complex128)
            U_theta_tq = jnp.kron(U_theta,U_theta)

            theta_CZ_psi0 = U_theta_tq @ CZ_psi0
            fid = jnp.abs(jnp.conj(theta_CZ_psi0).T @ state_final @ theta_CZ_psi0)

            return fid[0,0], theta
    
    def CZ_back_to_00(self, state_final, state_idx, state_initial=None, theta=None):
        """Apply recovery operations to map final state back to |00⟩ basis.
        
        Parameters
        ----------
        state_final : array
            Final density matrix.
        state_idx : int
            SSS state index for selecting recovery unitary.
        state_initial : array, optional
            Initial state.
        theta : float, optional
            Z-rotation angle; if None, determined from fidelity optimization.
            
        Returns
        -------
        array
            Recovered density matrix in |00⟩ reference frame.
        """
        if state_initial is None:
            state_initial = self.psi0
        CZ_psi0 = self.CZ_ideal() @ state_initial

        if theta is None:
            def fidelity_theta(theta):
                
                U_theta = jnp.array(block_diag(gates.rz(theta).full(),np.eye(8)),dtype=jnp.complex128)
                U_theta_tq = jnp.kron(U_theta,U_theta)

                theta_CZ_psi0 = U_theta_tq @ CZ_psi0
                fid = jnp.abs(jnp.conj(theta_CZ_psi0).T @ state_final @ theta_CZ_psi0)
            
                return fid[0,0]
            
            res = minimize_scalar(lambda theta:-fidelity_theta(theta), bounds=(-jnp.pi,jnp.pi), method='bounded')

            theta = res.x

        U_theta = jnp.array(block_diag(gates.rz(-theta).full(),np.eye(8)),dtype=jnp.complex128)
        U_theta_tq = jnp.kron(U_theta,U_theta)

        U_rec = self.SSS_rec(state_idx)

        state_final_00 = (U_rec @ self.CZ_ideal() @ U_theta_tq) @ state_final @ jnp.conj((U_rec @ self.CZ_ideal() @ U_theta_tq)).T

        return state_final_00
    
    def get_polarizability_fs(self, K, nLJ_target, nLJ_coupled, laser_freq):
        """Calculate rank-K polarizability tensor component in fine structure basis.
        
        Parameters
        ----------
        K : int
            Tensor rank (0=scalar, 1=vector, 2=tensor).
        nLJ_target : tuple
            Target state quantum numbers (n, L, J).
        nLJ_coupled : list of tuples
            Coupled states contributing to polarizability.
        laser_freq : float
            Laser frequency in MHz.
            
        Returns
        -------
        float
            Polarizability component α_K.
        """
        n,L,J = nLJ_target

        alpha_K = 0
        for _nLJ in nLJ_coupled:
            _n,_L,_J = _nLJ
            alpha_K += (jnp.sqrt(2*K+1)*(-1)**(K+J+1+_J)*Wigner6j(1,K,1,J,_J,J)*
                        jnp.abs(self.atom.getReducedMatrixElementJ(_n,_L,_J,n,L,J))**2*1
                        # (1/(2*jnp.pi*(self.atom.getTransitionFrequency(_n,_L,_J,n,L,J)/10**6-laser_freq))+
                        # (-1)**K/(2*jnp.pi*(self.atom.getTransitionFrequency(_n,_L,_J,n,L,J)/10**6+laser_freq)))
            ) 

        if K == 0:
            return (1/jnp.sqrt(3*(2*J+1)))*alpha_K
        elif K == 1:
            return -jnp.sqrt(2*J/((J+1)*(2*J+1)))*alpha_K
        elif K == 2:
            return -jnp.sqrt((2*J*(2*J-1)/(3*(J+1)*(2*J+1)*(2*J+3))))*alpha_K
        
    def get_polarizability_hfs_from_fs(self, K, F, I, nLJ_target, nLJ_coupled, laser_freq):
        """Calculate HFS polarizability from fine structure matrix elements.
        
        Parameters
        ----------
        K : int
            Tensor rank (0, 1, or 2).
        F : float
            Total angular momentum quantum number.
        I : float
            Nuclear spin (3/2 for 87Rb).
        nLJ_target : tuple
            Target state (n, L, J).
        nLJ_coupled : list
            Coupled states.
        laser_freq : float
            Laser frequency in MHz.
            
        Returns
        -------
        float
            HFS polarizability component.
        """
        n,L,J = nLJ_target

        alpha_K = 0
        for _nLJ in nLJ_coupled:
            _n,_L,_J = _nLJ
            alpha_K += (jnp.sqrt(2*K+1)*(-1)**(K+J+1+_J)*Wigner6j(1,K,1,J,_J,J)*
                        jnp.abs(self.atom.getReducedMatrixElementJ(_n,_L,_J,n,L,J))**2*1
                        # (1/(2*jnp.pi*(self.atom.getTransitionFrequency(_n,_L,_J,n,L,J)/10**6-laser_freq))+
                        # (-1)**K/(2*jnp.pi*(self.atom.getTransitionFrequency(_n,_L,_J,n,L,J)/10**6+laser_freq)))
            ) 

        if K == 0:
            return (1/jnp.sqrt(3*(2*J+1)))*alpha_K
        elif K == 1:
            return (-1)**(J+I+F)*jnp.sqrt(2*F*(2*F+1)/(F+1))*Wigner6j(F,1,F,J,I,J)*alpha_K
        elif K == 2:
            return -(-1)**(J+I+F)*jnp.sqrt((2*F*(2*F-1)*(2*F+1)/(3*(F+1)*(2*F+3))))*Wigner6j(F,2,F,J,I,J)*alpha_K
        
    def get_polarizability_hfs(self, K, I, nLJF_target, nLJF_coupled, laser_freq):
        """Calculate polarizability directly in hyperfine structure basis.
        
        Parameters
        ----------
        K : int
            Tensor rank (0, 1, or 2).
        I : float
            Nuclear spin.
        nLJF_target : tuple
            Target state (n, L, J, F).
        nLJF_coupled : list
            Coupled HFS states.
        laser_freq : float
            Laser frequency in MHz.
            
        Returns
        -------
        float
            HFS polarizability α_K.
        """
        n,L,J,F = nLJF_target
        A, B = self.atom.getHFSCoefficients(n,L,J)
        G = F*(F+1) - I*(I+1) - J*(J+1)
        if B != 0:
            GB = (3/2*G*(G+1)-2*I*(I+1)*J*(J+1))/(2*I*(2*I-1)*2*J*(2*J-1))
        else:
            GB = 0

        alpha_K = 0
        for _nLJF in nLJF_coupled:
            _n,_L,_J,_F = _nLJF
            _A, _B = self.atom.getHFSCoefficients(_n,_L,_J)
            _G = _F*(_F+1) - I*(I+1) - _J*(_J+1)
            if _B != 0:
                _GB = (3/2*_G*(_G+1)-2*I*(I+1)*_J*(_J+1))/(2*I*(2*I-1)*2*_J*(2*_J-1))
            else:
                _GB = 0
            alpha_K += (jnp.sqrt(2*K+1)*(-1)**(K+F+1+_F)*(2*F+1)*(2*_F+1)*
                        Wigner6j(1,K,1,F,_F,F)*Wigner6j(F,1,_F,_J,I,J)**2*
                        jnp.abs(self.atom.getReducedMatrixElementJ(_n,_L,_J,n,L,J))**2*1
                        # (1/(2*jnp.pi*((
                        #     self.atom.getTransitionFrequency(_n,_L,_J,n,L,J) - 
                        #     (1/2*_A*_G+_B*_GB) +
                        #     (1/2*A*G+B*GB)
                        # )/10**6-laser_freq))+
                        # (-1)**K/(2*jnp.pi*((
                        #     self.atom.getTransitionFrequency(_n,_L,_J,n,L,J) - 
                        #     (1/2*_A*_G+_B*_GB) +
                        #     (1/2*A*G+B*GB)
                        # )/10**6+laser_freq)))
            )

        if K == 0:
            return (1/jnp.sqrt(3*(2*F+1)))*alpha_K
        elif K == 1:
            return -jnp.sqrt(2*F/((F+1)*(2*F+1)))*alpha_K
        elif K == 2:
            return -jnp.sqrt((2*F*(2*F-1)/(3*(F+1)*(2*F+1)*(2*F+3))))*alpha_K

    # ------------------------------------------------------------------
    # --- INFIDELITY DIAGNOSTICS METHODS ---
    # ------------------------------------------------------------------

    def occ_operator(self, level_label):
        """Create two-atom occupation operator for a given level.
        
        Constructs the operator |i⟩⟨i| ⊗ I + I ⊗ |i⟩⟨i| which measures the
        total population in level i across both atoms.
        
        Parameters
        ----------
        level_label : str
            Level label from level_label list ('0', '1', 'e1', 'e2', 'e3',
            'r1', 'r2', 'rP', 'L0', 'L1').
            
        Returns
        -------
        array
            100x100 occupation operator for two-atom system.
        """
        idx = self.level_dict[level_label]['idx']
        ket = self.state_list[idx]
        proj = ket @ jnp.conj(ket).T
        return jnp.kron(jnp.eye(self.levels), proj) + jnp.kron(proj, jnp.eye(self.levels))

    def diagnose_population(self, sol):
        """Extract population evolution from density matrix trajectory.
        
        Computes time-resolved populations for different state categories
        to diagnose sources of gate infidelity.
        
        Parameters
        ----------
        sol : array
            Density matrix trajectory, shape (n_times, 100, 100).
            
        Returns
        -------
        dict
            Dictionary with population arrays for each category:
            - 'computational': Population in |00⟩, |01⟩, |10⟩, |11⟩
            - 'intermediate': Population in |e1⟩, |e2⟩, |e3⟩ states
            - 'rydberg': Population in target Rydberg |r1⟩
            - 'rydberg_unwanted': Population in |r2⟩ and |rP⟩
            - 'leakage': Population in |L0⟩ and |L1⟩
            - 'total_trace': Trace of density matrix (should be ~1)
        """
        n_times = sol.shape[0]
        
        # Build occupation operators
        occ_ops = {label: self.occ_operator(label) for label in self.level_label}
        
        # Computational basis indices (two-qubit)
        comp_indices = [0, 1, 10, 11]  # |00⟩, |01⟩, |10⟩, |11⟩
        
        populations = {
            'computational': jnp.zeros(n_times),
            'intermediate': jnp.zeros(n_times),
            'rydberg': jnp.zeros(n_times),
            'rydberg_unwanted': jnp.zeros(n_times),
            'leakage': jnp.zeros(n_times),
            'total_trace': jnp.zeros(n_times),
        }
        
        for t in range(n_times):
            rho = sol[t]
            
            # Computational basis population (diagonal elements)
            populations['computational'] = populations['computational'].at[t].set(
                sum(jnp.real(rho[i, i]) for i in comp_indices)
            )
            
            # Intermediate states (e1, e2, e3)
            for label in ['e1', 'e2', 'e3']:
                populations['intermediate'] = populations['intermediate'].at[t].add(
                    jnp.real(jnp.trace(occ_ops[label] @ rho)) / 2
                )
            
            # Target Rydberg state (r1)
            populations['rydberg'] = populations['rydberg'].at[t].set(
                jnp.real(jnp.trace(occ_ops['r1'] @ rho)) / 2
            )
            
            # Unwanted Rydberg states (r2, rP)
            for label in ['r2', 'rP']:
                populations['rydberg_unwanted'] = populations['rydberg_unwanted'].at[t].add(
                    jnp.real(jnp.trace(occ_ops[label] @ rho)) / 2
                )
            
            # Leakage states (L0, L1)
            for label in ['L0', 'L1']:
                populations['leakage'] = populations['leakage'].at[t].add(
                    jnp.real(jnp.trace(occ_ops[label] @ rho)) / 2
                )
            
            # Total trace
            populations['total_trace'] = populations['total_trace'].at[t].set(
                jnp.real(jnp.trace(rho))
            )
        
        return populations

    def diagnose_infidelity(self, rho_final, psi0, theta=None):
        """Decompose gate infidelity into contributing error sources.
        
        Analyzes the final density matrix to identify the physical origins
        of gate error: leakage, residual excitation, decay, and coherent errors.
        
        Parameters
        ----------
        rho_final : array
            Final density matrix after gate operation (100x100).
        psi0 : array
            Initial pure state (100x1).
        theta : float, optional
            Z-rotation correction angle. If None, optimal theta is computed.
            
        Returns
        -------
        dict
            Dictionary with infidelity breakdown:
            - 'total_infidelity': 1 - F (total gate error)
            - 'leakage_error': Population in leakage states |L0⟩, |L1⟩
            - 'rydberg_residual': Residual Rydberg population after gate
            - 'intermediate_residual': Residual intermediate state population
            - 'decay_error': Trace loss due to spontaneous decay
            - 'coherent_error': Remaining coherent error (dephasing, etc.)
            - 'fidelity': Gate fidelity F
            - 'theta': Optimal Z-rotation angle used
        """
        # Get fidelity with optimal theta
        fid, opt_theta = self.CZ_fidelity(rho_final, psi0, theta)
        if theta is None:
            theta = opt_theta
        
        # Build occupation operators
        occ_ops = {label: self.occ_operator(label) for label in self.level_label}
        
        # Calculate population in each error channel
        leakage = 0.0
        for label in ['L0', 'L1']:
            leakage += jnp.real(jnp.trace(occ_ops[label] @ rho_final)) / 2
        
        rydberg_residual = 0.0
        for label in ['r1', 'r2', 'rP']:
            rydberg_residual += jnp.real(jnp.trace(occ_ops[label] @ rho_final)) / 2
        
        intermediate_residual = 0.0
        for label in ['e1', 'e2', 'e3']:
            intermediate_residual += jnp.real(jnp.trace(occ_ops[label] @ rho_final)) / 2
        
        # Trace loss indicates decay
        trace = jnp.real(jnp.trace(rho_final))
        decay_error = 1.0 - trace
        
        # Total infidelity
        total_infidelity = 1.0 - fid
        
        # Coherent error is what remains after accounting for other sources
        # (Note: this is approximate as errors are not strictly additive)
        coherent_error = max(0.0, total_infidelity - leakage - rydberg_residual 
                           - intermediate_residual - decay_error)
        
        return {
            'total_infidelity': float(total_infidelity),
            'leakage_error': float(leakage),
            'rydberg_residual': float(rydberg_residual),
            'intermediate_residual': float(intermediate_residual),
            'decay_error': float(decay_error),
            'coherent_error': float(coherent_error),
            'fidelity': float(fid),
            'theta': float(theta),
        }

    def diagnose_plot(self, tlist, sol, initial_label='11', save_path=None):
        """Plot population evolution during gate for infidelity diagnosis.
        
        Creates a visualization of how population flows between different
        atomic states during the gate operation.
        
        Parameters
        ----------
        tlist : array
            Time points in microseconds.
        sol : array
            Density matrix trajectory, shape (len(tlist), 100, 100).
        initial_label : str
            Label for the initial state (e.g., '11', '01', '00').
        save_path : str, optional
            Path to save the figure. If None, displays interactively.
            
        Returns
        -------
        dict
            Population data dictionary from diagnose_population.
        """
        populations = self.diagnose_population(sol)
        
        # Convert time to nanoseconds for display
        time_ns = np.array(tlist) * 1000
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Top panel: Main populations
        ax1.plot(time_ns, populations['computational'], 
                 label='Computational basis', lw=2, color='#2E86AB')
        ax1.plot(time_ns, populations['rydberg'], 
                 label='Rydberg |r1⟩', lw=2, linestyle='--', color='#A23B72')
        ax1.plot(time_ns, populations['intermediate'], 
                 label='Intermediate (6P₃/₂)', lw=2, linestyle=':', color='#F18F01')
        ax1.set_ylabel('Population', fontsize=12)
        ax1.set_title(f'Population Evolution from |{initial_label}⟩ Initial State', 
                      fontsize=14)
        ax1.legend(loc='right', fontsize=10)
        ax1.set_ylim(-0.05, 1.05)
        
        # Bottom panel: Error channels
        ax2.plot(time_ns, populations['rydberg_unwanted'], 
                 label="Unwanted Rydberg |r2⟩, |rP⟩", lw=2, color='#C73E1D')
        ax2.plot(time_ns, populations['leakage'], 
                 label='Leakage |L0⟩, |L1⟩', lw=2, linestyle='--', color='#3B1F2B')
        ax2.plot(time_ns, 1 - populations['total_trace'], 
                 label='Trace loss (decay)', lw=2, linestyle=':', color='#6B818C')
        ax2.set_xlabel('Time (ns)', fontsize=12)
        ax2.set_ylabel('Population', fontsize=12)
        ax2.set_title('Error Channels', fontsize=14)
        ax2.legend(loc='right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
        
        return populations
