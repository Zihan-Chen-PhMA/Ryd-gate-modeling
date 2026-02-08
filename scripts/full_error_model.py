# %%
!pip install qutip
!pip install ARC-Alkali-Rydberg-Calculator
!pip install jaxopt

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.linalg import block_diag
from scipy.optimize import minimize_scalar, curve_fit
from scipy import integrate
from collections import defaultdict
from arc import * 

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

    def __init__(self, blockade = True, 
                 ryd_decay = True, mid_decay = True, r2_ryd_decay = True,
                 state0_scatter = True, r2_coupling = True,
                 distance = 3,#um
                 amp_420 = 1., amp_1013 = 1.,
                 delta = 0.,
                 psi0 = None):
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

        self._420_amp = amp_420
        self._1013_amp = amp_1013

        self.init_420_ham(state0_scatter)
        self.init_1013_ham(r2_coupling)

        self.init_420_light_shift()
        self.init_1013_light_shift()

        self.r2_detuning = 2 * jnp.pi * 60 #MHz
        self.delta = - self._420_lightshift_1 + self._1013_lightshift_r - (1-self._1013_amp**2)*self._1013_lightshift_1 + delta

        self.delta_eff = - (1-self._420_amp**2) * self._420_lightshift_1 + (1-self._1013_amp**2) * (self._1013_lightshift_r - self._1013_lightshift_1) + delta

        self.init_branch_ratio()

        self.cops = []
        self.cops_sparse = []

        if ryd_decay:
            self.Gamma_BBR = 1 / 151.55 - 1 / 410.41 #MHz
        else:
            self.Gamma_BBR = 0
        self.cops_BBR = jnp.sqrt(self.Gamma_BBR)*self.state_rP*jnp.conj(self.state_r1).T
        self.cops_BBR_r2 = jnp.sqrt(self.Gamma_BBR)*self.state_rP*jnp.conj(self.state_r2).T
        if ryd_decay:
            self.cops.append(self.cops_BBR)
            self.cops_sparse.append([self.Gamma_BBR, self.level_dict['r1']['idx'], self.level_dict['rP']['idx']])
            if r2_ryd_decay:
                self.cops.append(self.cops_BBR_r2)
                self.cops_sparse.append([self.Gamma_BBR, self.level_dict['r2']['idx'], self.level_dict['rP']['idx']])

        if ryd_decay:
            self.Gamma_RD = 1 / 410.41 #MHz
        else:
            self.Gamma_RD = 0
        # self.rydberg_RD_branch_ratio()
        self.cops_RD_0 = jnp.sqrt(self.branch_ratio_0*self.Gamma_RD)*self.state_0*jnp.conj(self.state_r1).T
        self.cops_RD_1 = jnp.sqrt(self.branch_ratio_1*self.Gamma_RD)*self.state_1*jnp.conj(self.state_r1).T
        self.cops_RD_L0 = jnp.sqrt(self.branch_ratio_L0*self.Gamma_RD)*self.state_L0*jnp.conj(self.state_r1).T
        self.cops_RD_L1 = jnp.sqrt(self.branch_ratio_L1*self.Gamma_RD)*self.state_L1*jnp.conj(self.state_r1).T

        self.cops_RD_r2_0 = jnp.sqrt(self.r2_branch_ratio_0*self.Gamma_RD)*self.state_0*jnp.conj(self.state_r2).T
        self.cops_RD_r2_1 = jnp.sqrt(self.r2_branch_ratio_1*self.Gamma_RD)*self.state_1*jnp.conj(self.state_r2).T
        self.cops_RD_r2_L0 = jnp.sqrt(self.r2_branch_ratio_L0*self.Gamma_RD)*self.state_L0*jnp.conj(self.state_r2).T
        self.cops_RD_r2_L1 = jnp.sqrt(self.r2_branch_ratio_L1*self.Gamma_RD)*self.state_L1*jnp.conj(self.state_r2).T
        if ryd_decay:
            self.cops += [self.cops_RD_0,self.cops_RD_1,self.cops_RD_L0,self.cops_RD_L1]
            self.cops_sparse += [
                [self.branch_ratio_0*self.Gamma_RD, self.level_dict['r1']['idx'], self.level_dict['0']['idx']],
                [self.branch_ratio_1*self.Gamma_RD, self.level_dict['r1']['idx'], self.level_dict['1']['idx']],
                [self.branch_ratio_L0*self.Gamma_RD, self.level_dict['r1']['idx'], self.level_dict['L0']['idx']],
                [self.branch_ratio_L1*self.Gamma_RD, self.level_dict['r1']['idx'], self.level_dict['L1']['idx']],
            ]
            if r2_ryd_decay:
                self.cops += [self.cops_RD_r2_0,self.cops_RD_r2_1,self.cops_RD_r2_L0,self.cops_RD_r2_L1]
                self.cops_sparse += [
                    [self.r2_branch_ratio_0*self.Gamma_RD, self.level_dict['r2']['idx'], self.level_dict['0']['idx']],
                    [self.r2_branch_ratio_1*self.Gamma_RD, self.level_dict['r2']['idx'], self.level_dict['1']['idx']],
                    [self.r2_branch_ratio_L0*self.Gamma_RD, self.level_dict['r2']['idx'], self.level_dict['L0']['idx']],
                    [self.r2_branch_ratio_L1*self.Gamma_RD, self.level_dict['r2']['idx'], self.level_dict['L1']['idx']],
                ]

        if mid_decay:
            self.Gamma_mid = 1 / 0.11 #MHz
        else:
            self.Gamma_mid = 0

        # e1_branch_ratio_0, e1_branch_ratio_1, e1_branch_ratio_L0, e1_branch_ratio_L1 = self.mid_branch_ratio('e1')
        self.cops_e1_0 = jnp.sqrt(self.e1_branch_ratio_0*self.Gamma_mid)*self.state_0*jnp.conj(self.state_e1).T
        self.cops_e1_1 = jnp.sqrt(self.e1_branch_ratio_1*self.Gamma_mid)*self.state_1*jnp.conj(self.state_e1).T
        self.cops_e1_L0 = jnp.sqrt(self.e1_branch_ratio_L0*self.Gamma_mid)*self.state_L0*jnp.conj(self.state_e1).T
        self.cops_e1_L1 = jnp.sqrt(self.e1_branch_ratio_L1*self.Gamma_mid)*self.state_L1*jnp.conj(self.state_e1).T
        if mid_decay:
            self.cops += [self.cops_e1_0,self.cops_e1_1,self.cops_e1_L0,self.cops_e1_L1]
            self.cops_sparse += [
                [self.e1_branch_ratio_0*self.Gamma_mid, self.level_dict['e1']['idx'], self.level_dict['0']['idx']],
                [self.e1_branch_ratio_1*self.Gamma_mid, self.level_dict['e1']['idx'], self.level_dict['1']['idx']],
                [self.e1_branch_ratio_L0*self.Gamma_mid, self.level_dict['e1']['idx'], self.level_dict['L0']['idx']],
                [self.e1_branch_ratio_L1*self.Gamma_mid, self.level_dict['e1']['idx'], self.level_dict['L1']['idx']],
            ]

        # e2_branch_ratio_0, e2_branch_ratio_1, e2_branch_ratio_L0, e2_branch_ratio_L1 = self.mid_branch_ratio('e2')
        self.cops_e2_0 = jnp.sqrt(self.e2_branch_ratio_0*self.Gamma_mid)*self.state_0*jnp.conj(self.state_e2).T
        self.cops_e2_1 = jnp.sqrt(self.e2_branch_ratio_1*self.Gamma_mid)*self.state_1*jnp.conj(self.state_e2).T
        self.cops_e2_L0 = jnp.sqrt(self.e2_branch_ratio_L0*self.Gamma_mid)*self.state_L0*jnp.conj(self.state_e2).T
        self.cops_e2_L1 = jnp.sqrt(self.e2_branch_ratio_L1*self.Gamma_mid)*self.state_L1*jnp.conj(self.state_e2).T
        if mid_decay:
            self.cops += [self.cops_e2_0,self.cops_e2_1,self.cops_e2_L0,self.cops_e2_L1]
            self.cops_sparse += [
                [self.e2_branch_ratio_0*self.Gamma_mid, self.level_dict['e2']['idx'], self.level_dict['0']['idx']],
                [self.e2_branch_ratio_1*self.Gamma_mid, self.level_dict['e2']['idx'], self.level_dict['1']['idx']],
                [self.e2_branch_ratio_L0*self.Gamma_mid, self.level_dict['e2']['idx'], self.level_dict['L0']['idx']],
                [self.e2_branch_ratio_L1*self.Gamma_mid, self.level_dict['e2']['idx'], self.level_dict['L1']['idx']],
            ]

        # e3_branch_ratio_0, e3_branch_ratio_1, e3_branch_ratio_L0, e3_branch_ratio_L1 = self.mid_branch_ratio('e3')
        self.cops_e3_0 = jnp.sqrt(self.e3_branch_ratio_0*self.Gamma_mid)*self.state_0*jnp.conj(self.state_e3).T
        self.cops_e3_1 = jnp.sqrt(self.e3_branch_ratio_1*self.Gamma_mid)*self.state_1*jnp.conj(self.state_e3).T
        self.cops_e3_L0 = jnp.sqrt(self.e3_branch_ratio_L0*self.Gamma_mid)*self.state_L0*jnp.conj(self.state_e3).T
        self.cops_e3_L1 = jnp.sqrt(self.e3_branch_ratio_L1*self.Gamma_mid)*self.state_L1*jnp.conj(self.state_e3).T
        if mid_decay:
            self.cops += [self.cops_e3_0,self.cops_e3_1,self.cops_e3_L0,self.cops_e3_L1]
            self.cops_sparse += [
                [self.e3_branch_ratio_0*self.Gamma_mid, self.level_dict['e3']['idx'], self.level_dict['0']['idx']],
                [self.e3_branch_ratio_1*self.Gamma_mid, self.level_dict['e3']['idx'], self.level_dict['1']['idx']],
                [self.e3_branch_ratio_L0*self.Gamma_mid, self.level_dict['e3']['idx'], self.level_dict['L0']['idx']],
                [self.e3_branch_ratio_L1*self.Gamma_mid, self.level_dict['e3']['idx'], self.level_dict['L1']['idx']],
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

        self.CZ_ideal_mat = self.CZ_ideal()
    
    def init_state0(self,psi0):
        self.psi0 = psi0
        self.rho0 = self.psi0*jnp.conj(self.psi0).T

    def psi_to_rho(self, psi):
        return psi @ jnp.conj(psi).T

    def init_state(self):
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
    
    def init_blockade(self,distance):

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

    def init_diag_ham(self,blockade = True):
        self.Hconst = jnp.zeros((self.levels,self.levels),dtype=jnp.complex128)

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

    def init_420_ham(self,state0_scatter):
        self.rabi_420 = jnp.sqrt(2 * jnp.abs(self.Delta) * self.rabi_eff)*self.rabi_ratio#2*jnp.pi*237#
        self.d_mid_ratio = (self.atom.getDipoleMatrixElementHFStoFS(5,0,1/2,2,0,
                                                                    6,1,3/2,-1/2,
                                                                    -1)/
                            self.atom.getDipoleMatrixElementHFStoFS(5,0,1/2,2,0,
                                                                    6,1,3/2,-3/2,
                                                                    -1))
        self.rabi_420_garbage = self.rabi_420*self.d_mid_ratio

        self.H_420 = jnp.zeros((self.levels,self.levels),dtype=jnp.complex128)

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

        if state0_scatter:
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
    
    def init_1013_ham(self,r2_coupling):
        self.rabi_1013 = jnp.sqrt(2 * jnp.abs(self.Delta) * self.rabi_eff)/self.rabi_ratio#2*jnp.pi*303#
        self.d_ryd_ratio = (self.atom.getDipoleMatrixElement(6,1,3/2,-1/2,
                                                             *self.level_dict['r2']['Qnum'],
                                                             1)/
                            self.atom.getDipoleMatrixElement(6,1,3/2,-3/2,
                                                             *self.level_dict['r1']['Qnum'],
                                                             1))
        self.rabi_1013_garbage = self.rabi_1013*self.d_ryd_ratio

        self.H_1013 = jnp.zeros((self.levels,self.levels),dtype=jnp.complex128)
   
        self.H_1013 = self.H_1013.at[5,2].set(
            (self.rabi_1013/2)*CG(3/2,-3/2,3/2,1/2,1,-1)
        )
        self.H_1013 = self.H_1013.at[5,3].set(
            (self.rabi_1013/2)*CG(3/2,-3/2,3/2,1/2,2,-1)
        )
        self.H_1013 = self.H_1013.at[5,4].set(
            (self.rabi_1013/2)*CG(3/2,-3/2,3/2,1/2,3,-1)
        )

        if r2_coupling:
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

        amp_420 = args["amp_420"](t)
        phase_420 = args["phase_420"](t)

        amp_1013 = args["amp_1013"](t)

        Hamitonian = (self.Hconst_tq + 
                      self._420_amp * (jnp.exp(-1j*phase_420) * self.H_420_tq + jnp.exp(1j*phase_420) * self.H_420_tq_conj) +
                      self._1013_amp * self.H_1013_tq_hermi )#- self.cdagc_tq * 1j * 0.5)
        
        return Hamitonian

    def init_hamiltonian_sparse(self):

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

        amp_420 = args["amp_420"](t)
        phase_420 = args["phase_420"](t)

        amp_1013 = args["amp_1013"](t)

        Hamitonian_sparse = jnp.concatenate([
            amp_420 * self._420_amp * jnp.exp(-1j*phase_420) * self.H_420_tq_sparse_value,
            self._1013_amp * self.H_1013_tq_sparse_value
        ])

        return Hamitonian_sparse

    
    def integrate_rho_jax(self, tlist, amp_420, phase_420, amp_1013, rho0 = None):
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
    
    def integrate_rho_multi_jax(self, tlist, amp_420, phase_420, amp_1013, rho0_list = None):

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
    
    def integrate_rho_multi2_jax(self, tlist, amp_420, phase_420, amp_1013, rho0_list = None):

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

    def get_CZ_fidelity_with_params_grads(self,params):

        # def amp_420(t, params):
        #     return 1.0
        
        def amp_420(t, params):
            t_rise = 0.02
            tf, A, omegaf, phi0, deltaf = params
            Omega = self.rabi_eff/(2*jnp.pi)

            t1 = jnp.clip(t,max=t_rise)
            t2 = jnp.clip(t,min=tf/Omega-t_rise)
            return (0.42 - 0.5*jnp.cos(2*jnp.pi*t1/(2*t_rise)) + 0.08*jnp.cos(4*jnp.pi*t1/(2*t_rise))) * (0.42 - 0.5*jnp.cos(2*jnp.pi*(tf/Omega-t2)/(2*t_rise)) + 0.08*jnp.cos(4*jnp.pi*(tf/Omega-t2)/(2*t_rise)))

        
        def phase_420(t, params):
            tf, A, omegaf, phi0, deltaf = params
            return 2 * jnp.pi * A * jnp.cos(omegaf * self.rabi_eff * t - phi0) + deltaf * self.rabi_eff * t
        
        def amp_1013(t, params):
            return 1.0
        
        def objective_fn(params):
            Omega = self.rabi_eff / (2*jnp.pi)
            tf, A, omegaf, phi0, deltaf = params
            tlist = jnp.linspace(0, tf/Omega, 2)

            sol = self.integrate_rho_multi_jax(
                tlist,
                amp_420=lambda t: amp_420(t,params),
                phase_420=lambda t: phase_420(t,params),
                amp_1013=lambda t: amp_1013(t,params),
                rho0_list=[self.psi_to_rho(psi) for psi in self.SSS_initial_state_list]
            )

            sol_mid = jax.vmap(self.mid_state_decay)(sol[:,-1])

            CZ_ideal = self.CZ_ideal_mat

            def get_single_bit_phase(sol_mid_n, psi0_n):
                def CZ_fidelity(theta):
                    rz = jnp.array([
                        [jnp.exp(-0.5j*theta),0.0],
                        [0.0,jnp.exp(0.5j*theta)]
                    ],dtype = jnp.complex128)
                    U_theta = jnp.block([
                        [rz, jnp.zeros((2,8),dtype = jnp.complex128)],
                        [jnp.zeros((8,2),dtype = jnp.complex128),jnp.eye(8,dtype = jnp.complex128)]
                    ])
                    U_theta_tq = jnp.kron(U_theta,U_theta)

                    theta_CZ_psi0 = U_theta_tq @ CZ_psi0
                    fid = jnp.abs(jnp.conj(theta_CZ_psi0).T @ sol_mid_n @ theta_CZ_psi0)
                
                    return fid[0,0]
                    
                CZ_psi0 = CZ_ideal @ psi0_n

                def refine(theta_l, theta_h, n=100):
                    thetas = jnp.linspace(theta_l,theta_h,n)
                    fids = jax.vmap(CZ_fidelity)(thetas)
                    idx = jnp.argmax(fids)
                    theta = thetas[idx]

                    idx = jnp.clip(idx,1,n-2)

                    return thetas[idx-1], theta, thetas[idx+1]

                def multi_refine(theta_l, theta_h, step=4):
                    def body_fn(i, val):
                        theta_l, theta_h = val
                        theta_l, _, theta_h = refine(theta_l,theta_h)
                        return theta_l, theta_h
                    
                    theta_l,theta_h = jax.lax.fori_loop(0, step, body_fn, (theta_l, theta_h))
                    _, theta, _ = refine(theta_l, theta_h)

                    return theta
                
                theta = multi_refine(0.0,2*jnp.pi)

                return jax.lax.stop_gradient(theta)
            
            get_single_bit_phase_jit = jit(get_single_bit_phase) 

            theta_list = jax.vmap(get_single_bit_phase_jit)(sol_mid, jnp.array(self.SSS_initial_state_list))
            theta_mask = jnp.array([0,1,2,3,6,7,8,9])
            theta_mean = jnp.mean(theta_list[theta_mask])

            def get_Ps(sol_mid_n, psi0_n, n, theta_mean):

                def CZ_back_to_00(state_final, state_idx, state_initial = None, theta = None):
                    if state_initial is None:
                        state_initial = self.psi0

                    rz = jnp.array([
                        [jnp.exp(0.5j*theta),0.0],
                        [0.0,jnp.exp(-0.5j*theta)]
                    ],dtype = jnp.complex128)
                    U_theta = jnp.block([
                        [rz, jnp.zeros((2,8),dtype = jnp.complex128)],
                        [jnp.zeros((8,2),dtype = jnp.complex128),jnp.eye(8,dtype = jnp.complex128)]
                    ])
                    U_theta_tq = jnp.kron(U_theta,U_theta)

                    U_rec = self.SSS_rec(state_idx)

                    state_final_00 = (U_rec @ self.CZ_ideal_mat @ U_theta_tq) @ state_final @ jnp.conj((U_rec @ self.CZ_ideal_mat @ U_theta_tq)).T

                    return state_final_00

                sol_state_00 = CZ_back_to_00(sol_mid_n,n,psi0_n,theta_mean)

                diag = jnp.abs(jnp.diag(sol_state_00))

                P_00 = diag[0]
                P_00_withL = jnp.sum(diag[jnp.array([0,8,80,88])])
                P_01_withL = jnp.sum(diag[jnp.array([1,9,81,89])])
                P_10_withL = jnp.sum(diag[jnp.array([10,18,90,98])])
                P_11_withL = jnp.sum(diag[jnp.array([11,19,91,99])])

                P_Loss = jnp.sum(
                    diag[jnp.array([
                        5,6,7,15,16,17,
                        50,51,55,56,57,58,59,
                        60,61,65,66,67,68,69,
                        70,71,75,76,77,78,79,
                        85,86,87,95,96,97
                    ])]
                )

                P_RL = jnp.sum(
                    diag[jnp.array([
                        50,51,58,59,60,61,68,69,70,71,78,79
                    ])]
                )

                P_LR = jnp.sum(
                    diag[jnp.array([
                        5,6,7,15,16,17,85,86,87,95,96,97
                    ])]
                )

                P_LL = jnp.sum(
                    diag[jnp.array([
                        55,56,57,65,66,67,75,76,77
                    ])]
                )
            
                return jnp.array([
                    P_00,
                    P_00_withL,
                    P_01_withL,
                    P_10_withL,
                    P_11_withL,
                    P_Loss,
                    P_RL,
                    P_LR,
                    P_LL
                ])
            
            get_Ps_jit = jit(get_Ps)
            
            obs = jax.vmap(
                get_Ps_jit,
                in_axes=(0,0,0,None)
            )(
                sol_mid,
                jnp.array(self.SSS_initial_state_list),
                jnp.arange(len(self.SSS_initial_state_list)),
                theta_mean
            )

            obs_mean = jnp.mean(obs,axis=0)
        
            P_00_withL_mean = obs_mean[1]

            return P_00_withL_mean, obs_mean

        (P_00_withL_mean, obs_mean), grad = jax.value_and_grad(
            objective_fn, has_aux = True
        )(params)

        return obs_mean, grad

    def mid_state_decay(self, rho0 = None):
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

    # def rydberg_RD_branch_ratio(self):
    #     I, mI = 3/2, 1/2
    #     nr, lr, jr, mjr = self.level_dict['r1']['Qnum']
    #     fr = [2,1]
    #     mfr = mI + mjr

    #     ne,le = 5,1
    #     je = [3/2,1/2]

    #     ng, lg, jg = 5, 0, 1/2

    #     a = []
    #     b = []

    #     for _je in je:
    #         fe = np.arange(np.abs(I-_je),I+_je+1,1)
    #         mje = np.arange(-_je,_je+1,1)
    #         for _fe in fe:
    #             mfe = np.arange(-_fe,_fe+1,1)
    #             for _mfe in mfe:
    #                 t = 0

    #                 for _fr in fr:
    #                     if np.abs(mfr) <= _fr:
    #                         if np.abs(mfr - _mfe) < 2:
    #                             t+=CG(jr,mjr,I,mI,_fr,mfr)*self.atom.getDipoleMatrixElementHFS(ne,le,_je,_fe,_mfe,nr,lr,jr,_fr,mfr,q=mfr-_mfe)
    #                 a.append(t**2)

    #                 bb = []
    #                 for fg in [2,1]:
    #                     mfg = np.arange(-fg,fg+1,1)
    #                     for _mfg in mfg:
    #                         if np.abs(_mfg - _mfe) < 2:
    #                             bb.append((self.atom.getDipoleMatrixElementHFS(ne,le,_je,_fe,_mfe,ng,lg,jg,fg,_mfg,q=_mfg - _mfe))**2)
    #                         else:
    #                             bb.append(0.0)
    #                 bb = [i/np.sum(np.array(bb)) for i in bb]
    #                 b.append(bb)

    #     a = [i/np.sum(np.array(a)) for i in a]

    #     branch_ratio = np.array([a[i]*np.array(b[i]) for i in range(len(a))]).sum(axis=0)

    #     self.branch_ratio_0 = branch_ratio[6]
    #     self.branch_ratio_1 = branch_ratio[2]
    #     self.branch_ratio_L0 = branch_ratio[5] + branch_ratio[7]
    #     self.branch_ratio_L1 = branch_ratio[0] + branch_ratio[1] + branch_ratio[3] + branch_ratio[4]

    # def mid_branch_ratio(self,level_label):
    #     ne,le,je,fe,mfe = self.level_dict[level_label]['Qnum']

    #     ng,lg,jg = 5,0,1/2
    #     # fg = [2,1]

    #     # a = []
    #     # for _fg in fg:
    #     #     mfg = np.arange(-_fg,_fg+1,1)
    #     #     for _mfg in mfg:
    #     #         if np.abs(_mfg - mfe) < 2:
    #     #             a.append((self.atom.getDipoleMatrixElementHFS(ne,le,je,fe,mfe,ng,lg,jg,_fg,_mfg,q=_mfg - mfe))**2)
    #     #         else:
    #     #             a.append(0.0)
    #     # branch_ratio = [i/np.sum(np.array(a)) for i in a]

    #     branch_ratio_6P_3_2 = [
    #         (self.atom.getReducedMatrixElementJ(6,1,3/2,5,0,1/2)**2)*self.atom.getTransitionFrequency(6,1,3/2,5,0,1/2)**3,
    #         (self.atom.getReducedMatrixElementJ(6,1,3/2,6,0,1/2)**2)*self.atom.getTransitionFrequency(6,1,3/2,6,0,1/2)**3,
    #         (self.atom.getReducedMatrixElementJ(6,1,3/2,4,2,5/2)**2)*self.atom.getTransitionFrequency(6,1,3/2,4,2,5/2)**3,
    #         (self.atom.getReducedMatrixElementJ(6,1,3/2,4,2,3/2)**2)*self.atom.getTransitionFrequency(6,1,3/2,4,2,3/2)**3
    #     ]
    #     branch_ratio_6P_3_2_to_5S_1_2 = branch_ratio_6P_3_2[0]/np.sum(branch_ratio_6P_3_2)
    #     branch_ratio_6P_3_2_to_6S_1_2 = branch_ratio_6P_3_2[1]/np.sum(branch_ratio_6P_3_2)
    #     branch_ratio_6P_3_2_to_4D_5_2 = branch_ratio_6P_3_2[2]/np.sum(branch_ratio_6P_3_2)
    #     branch_ratio_6P_3_2_to_4D_3_2 = branch_ratio_6P_3_2[3]/np.sum(branch_ratio_6P_3_2)

    #     branch_ratio_4D_3_2 = [
    #         (self.atom.getReducedMatrixElementJ(4,2,3/2,5,1,3/2)**2)*self.atom.getTransitionFrequency(4,2,3/2,5,1,3/2)**3,
    #         (self.atom.getReducedMatrixElementJ(4,2,3/2,5,1,1/2)**2)*self.atom.getTransitionFrequency(4,2,3/2,5,1,1/2)**3,
    #     ]
    #     branch_ratio_4D_3_2_to_5P_3_2 = branch_ratio_4D_3_2[0]/np.sum(branch_ratio_4D_3_2)
    #     branch_ratio_4D_3_2_to_5P_1_2 = branch_ratio_4D_3_2[1]/np.sum(branch_ratio_4D_3_2)

    #     branch_ratio_6S_1_2 = [
    #         (self.atom.getReducedMatrixElementJ(6,0,1/2,5,1,3/2)**2)*self.atom.getTransitionFrequency(6,0,1/2,5,1,3/2)**3,
    #         (self.atom.getReducedMatrixElementJ(6,0,1/2,5,1,1/2)**2)*self.atom.getTransitionFrequency(6,0,1/2,5,1,1/2)**3,
    #     ]
    #     branch_ratio_6S_1_2_to_5P_3_2 = branch_ratio_6S_1_2[0]/np.sum(branch_ratio_6S_1_2)
    #     branch_ratio_6S_1_2_to_5P_1_2 = branch_ratio_6S_1_2[1]/np.sum(branch_ratio_6S_1_2)

    #     fg_list = [jg+3/2-i for i in range(int(jg+3/2-np.abs(jg-3/2)+1))]
    #     fmf_g_list = [[fg,mfg] for fg in fg_list for mfg in range(int(-fg),int(fg+1))]

    #     fe_list = [je+3/2-i for i in range(int(je+3/2-np.abs(je-3/2)+1))]
    #     fmf_e_list = [[fe,mfe] for fe in fe_list for mfe in range(int(-fe),int(fe+1))]

    #     P_3_2_to_S_1_2 = np.zeros((len(fmf_e_list),len(fmf_g_list)))
    #     S_1_2_to_P_3_2 = np.zeros((len(fmf_g_list),len(fmf_e_list)))
    #     for i in range(len(fmf_e_list)):
    #         for j in range(len(fmf_g_list)):
    #             fg,mfg = fmf_g_list[j]
    #             fe,mfe = fmf_e_list[i]
    #             P_3_2_to_S_1_2[i,j] = self.atom.getBranchingRatio(jg,fg,mfg,je,fe,mfe)
    #             S_1_2_to_P_3_2[j,i] = self.atom.getBranchingRatio(je,fe,mfe,jg,fg,mfg)

    #     f_P_1_2_list = [1/2+3/2-i for i in range(int(1/2+3/2-np.abs(1/2-3/2)+1))]
    #     fmf_P_1_2_list = [[f_P_1_2,mfe] for f_P_1_2 in f_P_1_2_list for mfe in range(int(-f_P_1_2),int(f_P_1_2+1))]
    #     P_1_2_to_S_1_2 = np.zeros((len(fmf_P_1_2_list),len(fmf_g_list)))
    #     S_1_2_to_P_1_2 = np.zeros((len(fmf_g_list),len(fmf_P_1_2_list)))
    #     for i in range(len(fmf_P_1_2_list)):
    #         for j in range(len(fmf_g_list)):
    #             fg,mfg = fmf_g_list[j]
    #             fe,mfe = fmf_P_1_2_list[i]
    #             P_1_2_to_S_1_2[i,j] = self.atom.getBranchingRatio(jg,fg,mfg,1/2,fe,mfe)
    #             S_1_2_to_P_1_2[j,i] = self.atom.getBranchingRatio(1/2,fe,mfe,jg,fg,mfg)
        
    #     f_D_5_2_list = [5/2+3/2-i for i in range(int(5/2+3/2-np.abs(5/2-3/2)+1))]
    #     fmf_D_5_2_list = [[f_D_5_2,mfe] for f_D_5_2 in f_D_5_2_list for mfe in range(int(-f_D_5_2),int(f_D_5_2+1))]
    #     P_3_2_to_D_5_2 = np.zeros((len(fmf_e_list),len(fmf_D_5_2_list)))
    #     D_5_2_to_P_3_2 = np.zeros((len(fmf_D_5_2_list),len(fmf_e_list)))
    #     for i in range(len(fmf_D_5_2_list)):
    #         for j in range(len(fmf_e_list)):
    #             fg,mfg = fmf_D_5_2_list[i]
    #             fe,mfe = fmf_e_list[j]
    #             D_5_2_to_P_3_2[i,j] = self.atom.getBranchingRatio(je,fe,mfe,5/2,fg,mfg)
    #             P_3_2_to_D_5_2[j,i] = self.atom.getBranchingRatio(5/2,fg,mfg,je,fe,mfe)
        
    #     f_D_3_2_list = [3/2+3/2-i for i in range(int(3/2+3/2-np.abs(3/2-3/2)+1))]
    #     fmf_D_3_2_list = [[f_D_3_2,mfe] for f_D_3_2 in f_D_3_2_list for mfe in range(int(-f_D_3_2),int(f_D_3_2+1))]
    #     P_3_2_to_D_3_2 = np.zeros((len(fmf_e_list),len(fmf_D_3_2_list)))
    #     D_3_2_to_P_3_2 = np.zeros((len(fmf_D_3_2_list),len(fmf_e_list)))
    #     for i in range(len(fmf_D_3_2_list)):
    #         for j in range(len(fmf_e_list)):
    #             fg,mfg = fmf_D_3_2_list[i]
    #             fe,mfe = fmf_e_list[j]
    #             D_3_2_to_P_3_2[i,j] = self.atom.getBranchingRatio(je,fe,mfe,3/2,fg,mfg)
    #             P_3_2_to_D_3_2[j,i] = self.atom.getBranchingRatio(3/2,fg,mfg,je,fe,mfe)

    #     D_3_2_to_P_1_2 = np.zeros((len(fmf_D_3_2_list),len(fmf_P_1_2_list)))
    #     for i in range(len(fmf_D_3_2_list)):
    #         for j in range(len(fmf_P_1_2_list)):
    #             fg,mfg = fmf_D_3_2_list[i]
    #             fe,mfe = fmf_P_1_2_list[j]
    #             D_3_2_to_P_1_2[i,j] = self.atom.getBranchingRatio(1/2,fe,mfe,3/2,fg,mfg)


    #     branch_ratio_mat = (branch_ratio_6P_3_2_to_5S_1_2*P_3_2_to_S_1_2 + 
    #                         branch_ratio_6P_3_2_to_6S_1_2*(
    #                             P_3_2_to_S_1_2@(
    #                                 branch_ratio_6S_1_2_to_5P_3_2*(S_1_2_to_P_3_2@P_3_2_to_S_1_2)+
    #                                 branch_ratio_6S_1_2_to_5P_1_2*(S_1_2_to_P_1_2@P_1_2_to_S_1_2)
    #                         )) + 
    #                         branch_ratio_6P_3_2_to_4D_5_2*(
    #                             P_3_2_to_D_5_2@D_5_2_to_P_3_2@P_3_2_to_S_1_2
    #                         ) + 
    #                         branch_ratio_6P_3_2_to_4D_3_2*(
    #                             P_3_2_to_D_3_2@(
    #                                 branch_ratio_4D_3_2_to_5P_3_2*(D_3_2_to_P_3_2@P_3_2_to_S_1_2)+
    #                                 branch_ratio_4D_3_2_to_5P_1_2*(D_3_2_to_P_1_2@P_1_2_to_S_1_2)
    #                         )))
    #     if level_label == 'e1':
    #         branch_ratio = branch_ratio_mat[12]
    #     elif level_label == 'e2':
    #         branch_ratio = branch_ratio_mat[8]
    #     elif level_label == 'e3':
    #         branch_ratio = branch_ratio_mat[2]

    #     branch_ratio_0 = branch_ratio[6]
    #     branch_ratio_1 = branch_ratio[2]
    #     branch_ratio_L0 = branch_ratio[5] + branch_ratio[7]
    #     branch_ratio_L1 = branch_ratio[0] + branch_ratio[1] + branch_ratio[3] + branch_ratio[4]

    #     return branch_ratio_0, branch_ratio_1, branch_ratio_L0, branch_ratio_L1
    
    def init_branch_ratio(self):
        def f_list(j):
            return [j+3/2-i for i in range(int(j+3/2-np.abs(j-3/2)+1))]
        
        def fmf_list(flist):
            return [[f,mf] for f in flist for mf in range(int(-f),int(f+1))]
        
        def branch_ratio_hfs_to_HL(branch_ratio_hfs):
            return branch_ratio_hfs[6], branch_ratio_hfs[2], branch_ratio_hfs[5] + branch_ratio_hfs[7], branch_ratio_hfs[0] + branch_ratio_hfs[1] + branch_ratio_hfs[3] + branch_ratio_hfs[4]
        
        f_2J1 = f_list(1/2)
        f_2J3 = f_list(3/2)
        f_2J5 = f_list(5/2)

        fmf_2J1 = fmf_list(f_2J1)
        fmf_2J3 = fmf_list(f_2J3)
        fmf_2J5 = fmf_list(f_2J5)

        S_2J1_to_P_2J1 = np.zeros((len(fmf_2J1),len(fmf_2J1)))
        P_2J1_to_S_2J1 = np.zeros((len(fmf_2J1),len(fmf_2J1)))
        js, jp = 1/2, 1/2
        for i in range(len(fmf_2J1)):
            for j in range(len(fmf_2J1)):
                fs, mfs = fmf_2J1[i]
                fp, mfp = fmf_2J1[j]
                S_2J1_to_P_2J1[i,j] = self.atom.getBranchingRatio(jp,fp,mfp,js,fs,mfs)
                P_2J1_to_S_2J1[j,i] = self.atom.getBranchingRatio(js,fs,mfs,jp,fp,mfp)
        
        S_2J1_to_P_2J3 = np.zeros((len(fmf_2J1),len(fmf_2J3)))
        P_2J3_to_S_2J1 = np.zeros((len(fmf_2J3),len(fmf_2J1)))
        js, jp = 1/2, 3/2
        for i in range(len(fmf_2J1)):
            for j in range(len(fmf_2J3)):
                fs, mfs = fmf_2J1[i]
                fp, mfp = fmf_2J3[j]
                S_2J1_to_P_2J3[i,j] = self.atom.getBranchingRatio(jp,fp,mfp,js,fs,mfs)
                P_2J3_to_S_2J1[j,i] = self.atom.getBranchingRatio(js,fs,mfs,jp,fp,mfp)
        
        P_2J1_to_D_2J3 = np.zeros((len(fmf_2J1),len(fmf_2J3)))
        D_2J3_to_P_2J1 = np.zeros((len(fmf_2J3),len(fmf_2J1)))
        jp, jd = 1/2, 3/2
        for i in range(len(fmf_2J1)):
            for j in range(len(fmf_2J3)):
                fp, mfp = fmf_2J1[i]
                fd, mfd = fmf_2J3[j]
                P_2J1_to_D_2J3[i,j] = self.atom.getBranchingRatio(jd,fd,mfd,jp,fp,mfp)
                D_2J3_to_P_2J1[j,i] = self.atom.getBranchingRatio(jp,fp,mfp,jd,fd,mfd)
        
        P_2J1_to_D_2J3 = np.zeros((len(fmf_2J1),len(fmf_2J3)))
        D_2J3_to_P_2J1 = np.zeros((len(fmf_2J3),len(fmf_2J1)))
        jp, jd = 1/2, 3/2
        for i in range(len(fmf_2J1)):
            for j in range(len(fmf_2J3)):
                fp, mfp = fmf_2J1[i]
                fd, mfd = fmf_2J3[j]
                P_2J1_to_D_2J3[i,j] = self.atom.getBranchingRatio(jd,fd,mfd,jp,fp,mfp)
                D_2J3_to_P_2J1[j,i] = self.atom.getBranchingRatio(jp,fp,mfp,jd,fd,mfd)
        
        P_2J3_to_D_2J3 = np.zeros((len(fmf_2J3),len(fmf_2J3)))
        D_2J3_to_P_2J3 = np.zeros((len(fmf_2J3),len(fmf_2J3)))
        jp, jd = 3/2, 3/2
        for i in range(len(fmf_2J3)):
            for j in range(len(fmf_2J3)):
                fp, mfp = fmf_2J3[i]
                fd, mfd = fmf_2J3[j]
                P_2J3_to_D_2J3[i,j] = self.atom.getBranchingRatio(jd,fd,mfd,jp,fp,mfp)
                D_2J3_to_P_2J3[j,i] = self.atom.getBranchingRatio(jp,fp,mfp,jd,fd,mfd)

        P_2J3_to_D_2J5 = np.zeros((len(fmf_2J3),len(fmf_2J5)))
        D_2J5_to_P_2J3 = np.zeros((len(fmf_2J5),len(fmf_2J3)))
        jp, jd = 3/2, 5/2
        for i in range(len(fmf_2J3)):
            for j in range(len(fmf_2J5)):
                fp, mfp = fmf_2J3[i]
                fd, mfd = fmf_2J5[j]
                P_2J3_to_D_2J5[i,j] = self.atom.getBranchingRatio(jd,fd,mfd,jp,fp,mfp)
                D_2J5_to_P_2J3[j,i] = self.atom.getBranchingRatio(jp,fp,mfp,jd,fd,mfd)

        '''mid state'''
        branch_ratio_6P_2J3 = [
            (self.atom.getReducedMatrixElementJ(6,1,3/2,5,0,1/2)**2)*self.atom.getTransitionFrequency(6,1,3/2,5,0,1/2)**3,
            (self.atom.getReducedMatrixElementJ(6,1,3/2,6,0,1/2)**2)*self.atom.getTransitionFrequency(6,1,3/2,6,0,1/2)**3,
            (self.atom.getReducedMatrixElementJ(6,1,3/2,4,2,5/2)**2)*self.atom.getTransitionFrequency(6,1,3/2,4,2,5/2)**3,
            (self.atom.getReducedMatrixElementJ(6,1,3/2,4,2,3/2)**2)*self.atom.getTransitionFrequency(6,1,3/2,4,2,3/2)**3
        ]
        branch_ratio_6P_2J3_to_5S_2J1 = branch_ratio_6P_2J3[0]/np.sum(branch_ratio_6P_2J3)
        branch_ratio_6P_2J3_to_6S_2J1 = branch_ratio_6P_2J3[1]/np.sum(branch_ratio_6P_2J3)
        branch_ratio_6P_2J3_to_4D_2J5 = branch_ratio_6P_2J3[2]/np.sum(branch_ratio_6P_2J3)
        branch_ratio_6P_2J3_to_4D_2J3 = branch_ratio_6P_2J3[3]/np.sum(branch_ratio_6P_2J3)

        branch_ratio_6P_2J1 = [
            (self.atom.getReducedMatrixElementJ(6,1,1/2,5,0,1/2)**2)*self.atom.getTransitionFrequency(6,1,1/2,5,0,1/2)**3,
            (self.atom.getReducedMatrixElementJ(6,1,1/2,6,0,1/2)**2)*self.atom.getTransitionFrequency(6,1,1/2,6,0,1/2)**3,
            (self.atom.getReducedMatrixElementJ(6,1,1/2,4,2,3/2)**2)*self.atom.getTransitionFrequency(6,1,1/2,4,2,3/2)**3
        ]
        branch_ratio_6P_2J1_to_5S_2J1 = branch_ratio_6P_2J1[0]/np.sum(branch_ratio_6P_2J1)
        branch_ratio_6P_2J1_to_6S_2J1 = branch_ratio_6P_2J1[1]/np.sum(branch_ratio_6P_2J1)
        branch_ratio_6P_2J1_to_4D_2J3 = branch_ratio_6P_2J1[2]/np.sum(branch_ratio_6P_2J1)

        branch_ratio_4D_2J3 = [
            (self.atom.getReducedMatrixElementJ(4,2,3/2,5,1,3/2)**2)*self.atom.getTransitionFrequency(4,2,3/2,5,1,3/2)**3,
            (self.atom.getReducedMatrixElementJ(4,2,3/2,5,1,1/2)**2)*self.atom.getTransitionFrequency(4,2,3/2,5,1,1/2)**3,
        ]
        branch_ratio_4D_2J3_to_5P_2J3 = branch_ratio_4D_2J3[0]/np.sum(branch_ratio_4D_2J3)
        branch_ratio_4D_2J3_to_5P_2J1 = branch_ratio_4D_2J3[1]/np.sum(branch_ratio_4D_2J3)

        branch_ratio_6S_2J1 = [
            (self.atom.getReducedMatrixElementJ(6,0,1/2,5,1,3/2)**2)*self.atom.getTransitionFrequency(6,0,1/2,5,1,3/2)**3,
            (self.atom.getReducedMatrixElementJ(6,0,1/2,5,1,1/2)**2)*self.atom.getTransitionFrequency(6,0,1/2,5,1,1/2)**3,
        ]
        branch_ratio_6S_2J1_to_5P_2J3 = branch_ratio_6S_2J1[0]/np.sum(branch_ratio_6S_2J1)
        branch_ratio_6S_2J1_to_5P_2J1 = branch_ratio_6S_2J1[1]/np.sum(branch_ratio_6S_2J1)

        branch_ratio_6P_2J3_mat = (branch_ratio_6P_2J3_to_5S_2J1*P_2J3_to_S_2J1 + 
                            branch_ratio_6P_2J3_to_6S_2J1*(
                                P_2J3_to_S_2J1@(
                                    branch_ratio_6S_2J1_to_5P_2J3*(S_2J1_to_P_2J3@P_2J3_to_S_2J1)+
                                    branch_ratio_6S_2J1_to_5P_2J1*(S_2J1_to_P_2J1@P_2J1_to_S_2J1)
                            )) + 
                            branch_ratio_6P_2J3_to_4D_2J5*(
                                P_2J3_to_D_2J5@D_2J5_to_P_2J3@P_2J3_to_S_2J1
                            ) + 
                            branch_ratio_6P_2J3_to_4D_2J3*(
                                P_2J3_to_D_2J3@(
                                    branch_ratio_4D_2J3_to_5P_2J3*(D_2J3_to_P_2J3@P_2J3_to_S_2J1)+
                                    branch_ratio_4D_2J3_to_5P_2J1*(D_2J3_to_P_2J1@P_2J1_to_S_2J1)
                            )))
        
        branch_ratio = branch_ratio_6P_2J3_mat[12]
        self.e1_branch_ratio_0, self.e1_branch_ratio_1, self.e1_branch_ratio_L0, self.e1_branch_ratio_L1 = branch_ratio_hfs_to_HL(branch_ratio/np.sum(branch_ratio))

        branch_ratio = branch_ratio_6P_2J3_mat[8]
        self.e2_branch_ratio_0, self.e2_branch_ratio_1, self.e2_branch_ratio_L0, self.e2_branch_ratio_L1 = branch_ratio_hfs_to_HL(branch_ratio/np.sum(branch_ratio))

        branch_ratio = branch_ratio_6P_2J3_mat[2]
        self.e3_branch_ratio_0, self.e3_branch_ratio_1, self.e3_branch_ratio_L0, self.e3_branch_ratio_L1 = branch_ratio_hfs_to_HL(branch_ratio/np.sum(branch_ratio))

        '''rydberg state'''
        nr, lr, jr, mjr, mIr = 70, 0, 1/2, -1/2, 1/2#self.level_dict['r1']['Qnum']
        ne, le = 5, 1

        jmj_2J3 = [[3/2,-3/2+i] for i in range(4)]
        jmj_2J1 = [[1/2,-1/2+i] for i in range(2)]
        fs_r1_to_fs_P_2J3 = np.zeros((1,len(jmj_2J3)))
        for i in range(len(jmj_2J3)):
            je, mje = jmj_2J3[i]
            fs_r1_to_fs_P_2J3[0,i] += self.atom.getDipoleMatrixElement(ne,le,je,mje,nr,lr,jr,mjr,1)
            fs_r1_to_fs_P_2J3[0,i] += self.atom.getDipoleMatrixElement(ne,le,je,mje,nr,lr,jr,mjr,0)
            fs_r1_to_fs_P_2J3[0,i] += self.atom.getDipoleMatrixElement(ne,le,je,mje,nr,lr,jr,mjr,-1)
        fs_r1_to_fs_P_2J1 = np.zeros((1,len(jmj_2J1)))
        for i in range(len(jmj_2J1)):
            je, mje = jmj_2J1[i]
            fs_r1_to_fs_P_2J1[0,i] += self.atom.getDipoleMatrixElement(ne,le,je,mje,nr,lr,jr,mjr,1)
            fs_r1_to_fs_P_2J1[0,i] += self.atom.getDipoleMatrixElement(ne,le,je,mje,nr,lr,jr,mjr,0)
            fs_r1_to_fs_P_2J1[0,i] += self.atom.getDipoleMatrixElement(ne,le,je,mje,nr,lr,jr,mjr,-1)

        cg_mat_2J3 = np.zeros((len(jmj_2J3),len(fmf_2J3)))
        for i in range(len(jmj_2J3)):
            for j in range(len(fmf_2J3)):
                jj, mjj = jmj_2J3[i]
                f, mf = fmf_2J3[j]
                cg_mat_2J3[i,j] = CG(jj,mjj,3/2,mIr,f,mf)
        
        cg_mat_2J1 = np.zeros((len(jmj_2J1),len(fmf_2J1)))
        for i in range(len(jmj_2J1)):
            for j in range(len(fmf_2J1)):
                jj, mjj = jmj_2J1[i]
                f, mf = fmf_2J1[j]
                cg_mat_2J1[i,j] = CG(jj,mjj,3/2,mIr,f,mf)

        fs_r1_to_P_2J3 = (fs_r1_to_fs_P_2J3@cg_mat_2J3)**2
        fs_r1_to_P_2J1 = (fs_r1_to_fs_P_2J1@cg_mat_2J1)**2

        fs_r1_to_P_2J3 = fs_r1_to_P_2J3/np.sum(fs_r1_to_P_2J3)
        fs_r1_to_P_2J1 = fs_r1_to_P_2J1/np.sum(fs_r1_to_P_2J1)

        branch_ratio_r1_to_nP = np.array([
            (self.atom.getReducedMatrixElementJ(ne,1,je,70,0,1/2)**2)*self.atom.getTransitionFrequency(ne,1,je,70,0,1/2)**3 
            for ne in range(5,nr)
            for je in [3/2,1/2]
        ])
        branch_ratio_r1_to_5P_2J3 = branch_ratio_r1_to_nP[0]/np.sum(branch_ratio_r1_to_nP)
        branch_ratio_r1_to_5P_2J1 = branch_ratio_r1_to_nP[1]/np.sum(branch_ratio_r1_to_nP)
        branch_ratio_r1_to_6P_2J3 = branch_ratio_r1_to_nP[2]/np.sum(branch_ratio_r1_to_nP)
        branch_ratio_r1_to_6P_2J1 = branch_ratio_r1_to_nP[3]/np.sum(branch_ratio_r1_to_nP)
        branch_ratio_r1_else_2J3 = np.sum(branch_ratio_r1_to_nP[4::2])/np.sum(branch_ratio_r1_to_nP)
        branch_ratio_r1_else_2J1 = np.sum(branch_ratio_r1_to_nP[5::2])/np.sum(branch_ratio_r1_to_nP)

        branch_ratio_r1_mat = (
            (branch_ratio_r1_to_5P_2J3+branch_ratio_r1_else_2J3)*(fs_r1_to_P_2J3@P_2J3_to_S_2J1) +
            (branch_ratio_r1_to_5P_2J1+branch_ratio_r1_else_2J1)*(fs_r1_to_P_2J1@P_2J1_to_S_2J1) +
            branch_ratio_r1_to_6P_2J3*(
                fs_r1_to_P_2J3@(
                    branch_ratio_6P_2J3_to_6S_2J1*P_2J3_to_S_2J1@(
                        branch_ratio_6S_2J1_to_5P_2J3*(S_2J1_to_P_2J3@P_2J3_to_S_2J1) +
                        branch_ratio_6S_2J1_to_5P_2J1*(S_2J1_to_P_2J1@P_2J1_to_S_2J1)
                    ) +
                    branch_ratio_6P_2J3_to_4D_2J5*P_2J3_to_D_2J5@D_2J5_to_P_2J3@P_2J3_to_S_2J1 +
                    branch_ratio_6P_2J3_to_4D_2J3*P_2J3_to_D_2J3@(
                        branch_ratio_4D_2J3_to_5P_2J3*(D_2J3_to_P_2J3@P_2J3_to_S_2J1) +
                        branch_ratio_4D_2J3_to_5P_2J1*(D_2J3_to_P_2J1@P_2J1_to_S_2J1)
                    ) + 
                    branch_ratio_6P_2J3_to_5S_2J1*P_2J3_to_S_2J1
                )
            ) + 
            branch_ratio_r1_to_6P_2J1*(
                fs_r1_to_P_2J1@(
                    branch_ratio_6P_2J1_to_6S_2J1*P_2J1_to_S_2J1@(
                        branch_ratio_6S_2J1_to_5P_2J3*(S_2J1_to_P_2J3@P_2J3_to_S_2J1) +
                        branch_ratio_6S_2J1_to_5P_2J1*(S_2J1_to_P_2J1@P_2J1_to_S_2J1)
                    ) +
                    branch_ratio_6P_2J1_to_4D_2J3*P_2J1_to_D_2J3@(
                        branch_ratio_4D_2J3_to_5P_2J3*(D_2J3_to_P_2J3@P_2J3_to_S_2J1) +
                        branch_ratio_4D_2J3_to_5P_2J1*(D_2J3_to_P_2J1@P_2J1_to_S_2J1)
                    ) + 
                    branch_ratio_6P_2J1_to_5S_2J1*P_2J1_to_S_2J1
                )
            )
        )

        self.branch_ratio_0, self.branch_ratio_1, self.branch_ratio_L0, self.branch_ratio_L1 = branch_ratio_hfs_to_HL(branch_ratio_r1_mat[0]/np.sum(branch_ratio_r1_mat[0]))

        '''r2'''
        nr, lr, jr, mjr, mIr = 70, 0, 1/2, 1/2, -1/2#self.level_dict['r1']['Qnum']
        ne, le = 5, 1

        jmj_2J3 = [[3/2,-3/2+i] for i in range(4)]
        jmj_2J1 = [[1/2,-1/2+i] for i in range(2)]
        fs_r2_to_fs_P_2J3 = np.zeros((1,len(jmj_2J3)))
        for i in range(len(jmj_2J3)):
            je, mje = jmj_2J3[i]
            fs_r2_to_fs_P_2J3[0,i] += self.atom.getDipoleMatrixElement(ne,le,je,mje,nr,lr,jr,mjr,1)
            fs_r2_to_fs_P_2J3[0,i] += self.atom.getDipoleMatrixElement(ne,le,je,mje,nr,lr,jr,mjr,0)
            fs_r2_to_fs_P_2J3[0,i] += self.atom.getDipoleMatrixElement(ne,le,je,mje,nr,lr,jr,mjr,-1)
        fs_r2_to_fs_P_2J1 = np.zeros((1,len(jmj_2J1)))
        for i in range(len(jmj_2J1)):
            je, mje = jmj_2J1[i]
            fs_r2_to_fs_P_2J1[0,i] += self.atom.getDipoleMatrixElement(ne,le,je,mje,nr,lr,jr,mjr,1)
            fs_r2_to_fs_P_2J1[0,i] += self.atom.getDipoleMatrixElement(ne,le,je,mje,nr,lr,jr,mjr,0)
            fs_r2_to_fs_P_2J1[0,i] += self.atom.getDipoleMatrixElement(ne,le,je,mje,nr,lr,jr,mjr,-1)

        cg_mat_2J3 = np.zeros((len(jmj_2J3),len(fmf_2J3)))
        for i in range(len(jmj_2J3)):
            for j in range(len(fmf_2J3)):
                jj, mjj = jmj_2J3[i]
                f, mf = fmf_2J3[j]
                cg_mat_2J3[i,j] = CG(jj,mjj,3/2,mIr,f,mf)
        
        cg_mat_2J1 = np.zeros((len(jmj_2J1),len(fmf_2J1)))
        for i in range(len(jmj_2J1)):
            for j in range(len(fmf_2J1)):
                jj, mjj = jmj_2J1[i]
                f, mf = fmf_2J1[j]
                cg_mat_2J1[i,j] = CG(jj,mjj,3/2,mIr,f,mf)

        fs_r2_to_P_2J3 = (fs_r2_to_fs_P_2J3@cg_mat_2J3)**2
        fs_r2_to_P_2J1 = (fs_r2_to_fs_P_2J1@cg_mat_2J1)**2

        fs_r2_to_P_2J3 = fs_r2_to_P_2J3/np.sum(fs_r2_to_P_2J3)
        fs_r2_to_P_2J1 = fs_r2_to_P_2J1/np.sum(fs_r2_to_P_2J1)

        branch_ratio_r2_to_nP = np.array([
            (self.atom.getReducedMatrixElementJ(ne,1,je,70,0,1/2)**2)*self.atom.getTransitionFrequency(ne,1,je,70,0,1/2)**3 
            for ne in range(5,nr)
            for je in [3/2,1/2]
        ])
        branch_ratio_r2_to_5P_2J3 = branch_ratio_r2_to_nP[0]/np.sum(branch_ratio_r2_to_nP)
        branch_ratio_r2_to_5P_2J1 = branch_ratio_r2_to_nP[1]/np.sum(branch_ratio_r2_to_nP)
        branch_ratio_r2_to_6P_2J3 = branch_ratio_r2_to_nP[2]/np.sum(branch_ratio_r2_to_nP)
        branch_ratio_r2_to_6P_2J1 = branch_ratio_r2_to_nP[3]/np.sum(branch_ratio_r2_to_nP)
        branch_ratio_r2_else_2J3 = np.sum(branch_ratio_r2_to_nP[4::2])/np.sum(branch_ratio_r2_to_nP)
        branch_ratio_r2_else_2J1 = np.sum(branch_ratio_r2_to_nP[5::2])/np.sum(branch_ratio_r2_to_nP)

        branch_ratio_r2_mat = (
            (branch_ratio_r2_to_5P_2J3+branch_ratio_r2_else_2J3)*(fs_r2_to_P_2J3@P_2J3_to_S_2J1) +
            (branch_ratio_r2_to_5P_2J1+branch_ratio_r2_else_2J1)*(fs_r2_to_P_2J1@P_2J1_to_S_2J1) +
            branch_ratio_r2_to_6P_2J3*(
                fs_r2_to_P_2J3@(
                    branch_ratio_6P_2J3_to_6S_2J1*P_2J3_to_S_2J1@(
                        branch_ratio_6S_2J1_to_5P_2J3*(S_2J1_to_P_2J3@P_2J3_to_S_2J1) +
                        branch_ratio_6S_2J1_to_5P_2J1*(S_2J1_to_P_2J1@P_2J1_to_S_2J1)
                    ) +
                    branch_ratio_6P_2J3_to_4D_2J5*P_2J3_to_D_2J5@D_2J5_to_P_2J3@P_2J3_to_S_2J1 +
                    branch_ratio_6P_2J3_to_4D_2J3*P_2J3_to_D_2J3@(
                        branch_ratio_4D_2J3_to_5P_2J3*(D_2J3_to_P_2J3@P_2J3_to_S_2J1) +
                        branch_ratio_4D_2J3_to_5P_2J1*(D_2J3_to_P_2J1@P_2J1_to_S_2J1)
                    ) + 
                    branch_ratio_6P_2J3_to_5S_2J1*P_2J3_to_S_2J1
                )
            ) + 
            branch_ratio_r2_to_6P_2J1*(
                fs_r2_to_P_2J1@(
                    branch_ratio_6P_2J1_to_6S_2J1*P_2J1_to_S_2J1@(
                        branch_ratio_6S_2J1_to_5P_2J3*(S_2J1_to_P_2J3@P_2J3_to_S_2J1) +
                        branch_ratio_6S_2J1_to_5P_2J1*(S_2J1_to_P_2J1@P_2J1_to_S_2J1)
                    ) +
                    branch_ratio_6P_2J1_to_4D_2J3*P_2J1_to_D_2J3@(
                        branch_ratio_4D_2J3_to_5P_2J3*(D_2J3_to_P_2J3@P_2J3_to_S_2J1) +
                        branch_ratio_4D_2J3_to_5P_2J1*(D_2J3_to_P_2J1@P_2J1_to_S_2J1)
                    ) + 
                    branch_ratio_6P_2J1_to_5S_2J1*P_2J1_to_S_2J1
                )
            )
        )

        self.r2_branch_ratio_0, self.r2_branch_ratio_1, self.r2_branch_ratio_L0, self.r2_branch_ratio_L1 = branch_ratio_hfs_to_HL(branch_ratio_r2_mat[0]/np.sum(branch_ratio_r2_mat[0]))

    def init_SSS_states(self):

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
    
    # def SSS_rec(self,idx):

    #     p1 = [0,0,1,1,2,0,2,0,1,0,3,1]
    #     p2 = [3,1,0,2,0,0,1,1,1,0,0,0]
    #     p3 = [0,0,0,0,0,0,1,1,1,1,1,1]

    #     g1 = gates.qrot(np.pi/2,p2[idx]*np.pi/2)*gates.qrot(np.pi/2,p1[idx]*np.pi/2)
    #     g2 = gates.qrot(-np.pi/2,0)*gates.qrot(np.pi/2,p3[idx]*np.pi/2)

    #     G1 = jnp.array(block_diag(g1.full(),np.eye(8)),dtype=jnp.complex128)
    #     G2 = jnp.array(block_diag(g2.full(),np.eye(8)),dtype=jnp.complex128)

    #     return jnp.kron(G2,G2) @ self.CZ_ideal_mat @ jnp.kron(G1,G1)
    
    def SSS_rec(self, idx):

        p1 = jnp.array([0,0,1,1,2,0,2,0,1,0,3,1])
        p2 = jnp.array([3,1,0,2,0,0,1,1,1,0,0,0])
        p3 = jnp.array([0,0,0,0,0,0,1,1,1,1,1,1])

        def qrot(theta, phi):
            rz = jnp.array([
                [jnp.exp(-0.5j*phi), 0.0],
                [0.0, jnp.exp(0.5j*phi)]
            ], dtype=jnp.complex128)
            ry = jnp.array([
                [jnp.cos(theta/2), -1j*jnp.sin(theta/2)],
                [-1j*jnp.sin(theta/2), jnp.cos(theta/2)]
            ], dtype=jnp.complex128)
            rz_inv = jnp.array([
                [jnp.exp(0.5j*phi), 0.0],
                [0.0, jnp.exp(-0.5j*phi)]
            ], dtype=jnp.complex128)
            return rz @ ry @ rz_inv

        g1 = qrot(jnp.pi/2, p2[idx]*jnp.pi/2) @ qrot(jnp.pi/2, p1[idx]*jnp.pi/2)
        g2 = qrot(-jnp.pi/2, 0.0) @ qrot(jnp.pi/2, p3[idx]*jnp.pi/2)

        def block_diag_jax(A, B):
            top = jnp.concatenate([A, jnp.zeros((A.shape[0], B.shape[1]), dtype=jnp.complex128)], axis=1)
            bottom = jnp.concatenate([jnp.zeros((B.shape[0], A.shape[1]), dtype=jnp.complex128), B], axis=1)
            return jnp.concatenate([top, bottom], axis=0)

        G1 = block_diag_jax(g1, jnp.eye(8, dtype=jnp.complex128))
        G2 = block_diag_jax(g2, jnp.eye(8, dtype=jnp.complex128))

        return jnp.kron(G2, G2) @ self.CZ_ideal_mat @ jnp.kron(G1, G1)

    def rand_U(self):
        U = jnp.array(block_diag(rand_unitary(2).full(),np.eye(8)),dtype=jnp.complex128)
        return jnp.kron(U,U)

    def CZ_ideal(self):
        CZ = jnp.eye(self.levels**2,dtype=jnp.complex128)
        CZ = CZ.at[11,11].set(-1)
        return CZ
    
    def CZ_fidelity(self, state_final, state_initial = None, theta = None):
        if state_initial is None:
            state_initial = self.psi0
        CZ_psi0 = self.CZ_ideal_mat @ state_initial

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
    
    def CZ_back_to_00(self, state_final, state_idx, state_initial = None, theta = None):
        if state_initial is None:
            state_initial = self.psi0
        CZ_psi0 = self.CZ_ideal_mat @ state_initial

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

        state_final_00 = (U_rec @ self.CZ_ideal_mat @ U_theta_tq) @ state_final @ jnp.conj((U_rec @ self.CZ_ideal_mat @ U_theta_tq)).T

        eigvals, eigvecs = jnp.linalg.eigh(state_final_00)
        eigvals = jnp.clip(eigvals,0.0,None)
        state_final_00 = eigvecs @ jnp.diag(eigvals) @ eigvecs.conj().T

        state_final_00 = (state_final_00 + jnp.conj(state_final_00).T)/2
        state_final_00 = state_final_00 / jnp.real(jnp.trace(state_final_00))
        
        return state_final_00
    
    def get_polarizability_fs(self,K,nLJ_target,nLJ_coupled,laser_freq):
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
        
    def get_polarizability_hfs_from_fs(self,K,F,I,nLJ_target,nLJ_coupled,laser_freq):
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
        
    def get_polarizability_hfs(self,K,I,nLJF_target,nLJF_coupled,laser_freq):
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

model = jax_atom_Evolution()

#%%

'''Rydberg Rabi'''

P = []
for i in range(10):
    delta_f_T2 = np.random.normal(loc=0.,scale=160.0) #kHz

    delta_T2 = 2*jnp.pi * delta_f_T2 / 1000 #MHz

    model_noblockade = jax_atom_Evolution(blockade=False,mid_decay=False,ryd_decay=False,state0_scatter=False,r2_coupling=False)

    amp_420 = lambda t:1
    phase_420 = lambda t : 0
    amp_1013 = lambda t:1

    tlist = jnp.linspace(0, 1, 100)

    sol = model_noblockade.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013, model.psi_to_rho(jnp.kron((model.state_1+model.state_r1)/jnp.sqrt(2), (model.state_1+model.state_r1)/jnp.sqrt(2))))#model.psi_to_rho(model.SSS_initial_state_list[5]))

    sol_A = jnp.trace(sol.reshape(100,10,10,10,10),axis1=2,axis2=4)

    P.append(jnp.abs(jnp.conj((model.state_1+model.state_r1)/jnp.sqrt(2)).T@sol_A@((model.state_1+model.state_r1)/jnp.sqrt(2)))[:,0,0])

plt.plot(tlist,jnp.array(P).mean(axis=0))
# plt.plot(tlist,0.5+0.5*np.cos(2*np.pi*5*tlist))
plt.show()

#%%

'''Rydberg blockade'''

model = jax_atom_Evolution()

amp_420 = lambda t:1
phase_420 = lambda t : 0
amp_1013 = lambda t:1

tlist = jnp.linspace(0, 1, 200)

sol = model.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013, model.psi_to_rho(model.SSS_initial_state_list[5]))

plt.plot(tlist,jnp.abs(sol[:,11,11]))
plt.plot(tlist,0.5+0.5*np.cos(2*np.pi*5*np.sqrt(2)*tlist))
plt.show()

#%%

'''CZ evolution'''

Omega = model.rabi_eff / (2 * jnp.pi)

# tf = 1.21915#1.215
# A = -0.1119#0.1122
# omegaf = 1.0424#1.0431
# phi0 = -0.7256#-0.7318
# deltaf = 0.002#0

tf = 1.3443#1.215
A = -0.1242#0.1122
omegaf = 0.9511#1.0431
phi0 = -0.6988#-0.7318
deltaf = 0.0498#0

t_rise = 0.02

def amp_420(t):
    t1 = jnp.clip(t,max=t_rise)
    t2 = jnp.clip(t,min=tf/Omega-t_rise)
    return (0.42 - 0.5*jnp.cos(2*jnp.pi*t1/(2*t_rise)) + 0.08*jnp.cos(4*jnp.pi*t1/(2*t_rise))) * (0.42 - 0.5*jnp.cos(2*jnp.pi*(tf/Omega-t2)/(2*t_rise)) + 0.08*jnp.cos(4*jnp.pi*(tf/Omega-t2)/(2*t_rise)))

# amp_420 = lambda t:1
phase_420 = lambda t : 2 * jnp.pi * (A * jnp.cos(2 * jnp.pi * omegaf * Omega * t - phi0) + deltaf * Omega * t)
amp_1013 = lambda t:1

tlist = jnp.linspace(0, tf/Omega, 2000)

model = jax_atom_Evolution(mid_decay=False,ryd_decay=False,state0_scatter=False,r2_coupling=False)

sol = model.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013, model.psi_to_rho(model.SSS_initial_state_list[5]))

for i in range(100):
    plt.plot(jnp.abs(sol[:,i,i]))
plt.show()

# model_noblockade = jax_atom_Evolution(blockade=False)
# sol = model_noblockade.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013, model.psi_to_rho(model.SSS_initial_state_list[5]))
# sol_A = jnp.trace(sol.reshape(200,10,10,10,10),axis1=2,axis2=4)

# for i in range(10):
#     plt.plot(jnp.abs(sol_A[:,i,i]))
# plt.show()

#%%

'''params grad'''

model = jax_atom_Evolution(mid_decay=False,ryd_decay=False,state0_scatter=False,r2_coupling=False)

# tf = 1.21915#1.215
# A = -0.1119#0.1122
# omegaf = 1.0424#1.0431
# phi0 = -0.7256#-0.7318
# deltaf = 0.002#0

tf = 1.3443#1.215
A = -0.1242#0.1122
omegaf = 0.9511#1.0431
phi0 = -0.6988#-0.7318
deltaf = 0.0498#0

params = jnp.array([tf,A,omegaf,phi0,deltaf])

obs, grads = model.get_CZ_fidelity_with_params_grads(params)
P_00_mean = obs[0]
P_00_withL_mean = obs[1]
P_01_withL_mean = obs[2]
P_10_withL_mean = obs[3]
P_11_withL_mean = obs[4]
P_Loss_mean = obs[5]
P_RL_mean = obs[6]
P_LR_mean = obs[7]
P_LL_mean = obs[8]

print(f'Mean P_00: {P_00_mean:.6f},\nMean P_00_withL: {P_00_withL_mean:.6f}, Mean P_01_withL: {P_01_withL_mean:.6f}, Mean P_10_withL: {P_10_withL_mean:.6f}, Mean P_11_withL: {P_11_withL_mean:.6f},\nMean P_Loss: {P_Loss_mean:.6f}, Mean P_RL: {P_RL_mean:.6f}, Mean P_LR: {P_LR_mean:.6f}, Mean P_LL: {P_LL_mean:.6f}')

print(f'params grads:', grads)

#%%
P_error_list=[[0,0,0,0]]
#%%
P_list = []

for i in range(150):
    delta_f_T2 = np.random.normal(loc=0.,scale=130.0) #kHz

    x1 = np.random.normal(loc=0.,scale=0.07)
    y1 = np.random.normal(loc=0.,scale=0.07)
    z1 = np.random.normal(loc=0.,scale=0.17)

    x2 = np.random.normal(loc=0.,scale=0.07)
    y2 = np.random.normal(loc=0.,scale=0.07)
    z2 = np.random.normal(loc=0.,scale=0.17)

    d = np.sqrt((x1-x2+3)**2+(y1-y2)**2+(z1-z2)**2)

    delta_T2 = 2*jnp.pi * delta_f_T2 / 1000 #MHz

    model = jax_atom_Evolution(delta=delta_T2)

    Omega = model.rabi_eff / (2 * jnp.pi)

    # tf = 1.21915#1.215
    # A = -0.1119#0.1122
    # omegaf = 1.0424#1.0431
    # phi0 = -0.7256#-0.7318
    # deltaf = 0.002#0

    tf = 1.3443#1.215
    A = -0.1242#0.1122
    omegaf = 0.9511#1.0431
    phi0 = -0.6988#-0.7318
    deltaf = 0.0498#0

    t_rise = 0.02

    def amp_420(t):
        t1 = jnp.clip(t,max=t_rise)
        t2 = jnp.clip(t,min=tf/Omega-t_rise)
        return (0.42 - 0.5*jnp.cos(2*jnp.pi*t1/(2*t_rise)) + 0.08*jnp.cos(4*jnp.pi*t1/(2*t_rise))) * (0.42 - 0.5*jnp.cos(2*jnp.pi*(tf/Omega-t2)/(2*t_rise)) + 0.08*jnp.cos(4*jnp.pi*(tf/Omega-t2)/(2*t_rise)))

    # amp_420 = lambda t:1
    phase_420 = lambda t : 2 * jnp.pi * (A * jnp.cos(2 * jnp.pi * omegaf * Omega * t - phi0) + deltaf * Omega * t)
    amp_1013 = lambda t:1

    tlist = jnp.linspace(0, tf/Omega, 2)

    psi0_list = model.SSS_initial_state_list

    # sol = model.integrate_rho_jax(tlist, amp_420, phase_420, amp_1013, model.psi_to_rho(psi0_list[0]))
    # print("CZ fidelity:", model.CZ_fidelity(sol[-1], psi0_list[0]))

    sol = model.integrate_rho_multi_jax(tlist, amp_420, phase_420, amp_1013, [model.psi_to_rho(psi0) for psi0 in psi0_list])

    sol_mid = jnp.array([model.mid_state_decay(sol[n,-1]) for n in range(12)])

    P_00_withL_list = []
    P_survive_list = []

    fid_raw_mean = 0
    theta_mean = 0
    for n in range(12):
        fid_raw, theta = model.CZ_fidelity(sol_mid[n], psi0_list[n])
        fid_raw_mean += fid_raw / 12
        if n in [0,1,2,3,6,7,8,9]:
            theta_mean += theta / 8
    #     print("CZ fidelity:", fid_raw, " theta:", theta)                     
    # print('mean raw fidelity:', fid_raw_mean)
    # print('mean theta:', theta_mean)

    fid_mean = 0
    for n in range(12):
        fid, theta = model.CZ_fidelity(sol_mid[n], psi0_list[n], theta_mean)
        fid_mean += fid / 12
    #     print("CZ fidelity with mean theta:", fid)
    # print('mean fidelity with mean theta:', fid_mean)

    P_00_mean = 0
    P_01_mean = 0
    P_10_mean = 0
    P_11_mean = 0
    P_00_withL_mean = 0
    P_01_withL_mean = 0
    P_10_withL_mean = 0
    P_11_withL_mean = 0
    P_Loss_mean = 0
    P_RL_mean = 0
    P_LR_mean = 0
    P_LL_mean = 0
    P_Leakage_mean = 0
    P_Leakage_without00_mean = 0
    for n in range(12):
        sol_state_00 = model.CZ_back_to_00(sol_mid[n], n, psi0_list[n], theta_mean)
        P_00 = jnp.real(sol_state_00[0,0])
        P_00_mean += P_00 / 12
        P_01 = jnp.real(sol_state_00[1,1])
        P_01_mean += P_01 / 12
        P_10 = jnp.real(sol_state_00[10,10])
        P_10_mean += P_10 / 12
        P_11 = jnp.real(sol_state_00[11,11])
        P_11_mean += P_11 / 12
        P_00_withL = jnp.sum(jnp.real(jnp.diag(sol_state_00)[jnp.array([0,8,80,88])]))
        P_00_withL_mean += P_00_withL / 12
        P_01_withL = jnp.sum(jnp.real(jnp.diag(sol_state_00)[jnp.array([1,9,81,89])]))
        P_01_withL_mean += P_01_withL / 12
        P_10_withL = jnp.sum(jnp.real(jnp.diag(sol_state_00)[jnp.array([10,18,90,98])]))
        P_10_withL_mean += P_10_withL / 12
        P_11_withL = jnp.sum(jnp.real(jnp.diag(sol_state_00)[jnp.array([11,19,91,99])]))
        P_11_withL_mean += P_11_withL / 12
        P_Loss = jnp.sum(jnp.real(jnp.diag(sol_state_00)[jnp.array([
            5,6,7,15,16,17,
            50,51,55,56,57,58,59,
            60,61,65,66,67,68,69,
            70,71,75,76,77,78,79,
            85,86,87,95,96,97,
        ])]))
        P_Loss_mean += P_Loss / 12
        P_RL = jnp.sum(jnp.real(jnp.diag(sol_state_00)[jnp.array([
            50,51,58,59,60,61,68,69,70,71,78,79,
        ])]))
        P_RL_mean += P_RL / 12
        P_LR = jnp.sum(jnp.real(jnp.diag(sol_state_00)[jnp.array([
            5,6,7,15,16,17,85,86,87,95,96,97,
        ])]))
        P_LR_mean += P_LR / 12
        P_LL = jnp.sum(jnp.real(jnp.diag(sol_state_00)[jnp.array([
            55,56,57,65,66,67,75,76,77,
        ])]))
        P_LL_mean += P_LL / 12
        P_Leakage = jnp.sum(jnp.real(jnp.diag(sol_state_00)[jnp.array([
            8,80,88,
            9,81,89,
            18,90,98,
            19,91,99,
        ])]))
        P_Leakage_mean += P_Leakage / 12
        P_Leakage_without00 = jnp.sum(jnp.real(jnp.diag(sol_state_00)[jnp.array([
            9,81,89,
            18,90,98,
            19,91,99,
        ])]))
        P_Leakage_without00_mean += P_Leakage_without00 / 12
        # print(f"State {n}: P_00: {P_00:.6f},\nP_00_withL: {P_00_withL:.6f}, P_01_withL: {P_01_withL:.6f}, P_10_withL: {P_10_withL:.6f}, P_11_withL: {P_11_withL:.6f},\nP_Loss: {P_Loss:.6f}, P_RL: {P_RL:.6f}, P_LR: {P_LR:.6f}, P_LL: {P_LL:.6f}")
    print('sample:',i,'::',delta_f_T2, 'kHz,', d, 'um')
    print('*'*50)
    print('This time:')
    print(f'Mean P_00: {P_00_mean:.6f}, Mean P_01: {P_01_mean:.6f}, Mean P_10: {P_10_mean:.6f}, Mean P_11: {P_11_mean:.6f},\nMean P_00_withL: {P_00_withL_mean:.6f}, Mean P_01_withL: {P_01_withL_mean:.6f}, Mean P_10_withL: {P_10_withL_mean:.6f}, Mean P_11_withL: {P_11_withL_mean:.6f},\nMean P_Leakage: {P_Leakage_mean:.6f},Mean P_Leakage_without00: {P_Leakage_without00_mean:.6f}\nMean P_Loss: {P_Loss_mean:.6f}, Mean P_RL: {P_RL_mean:.6f}, Mean P_LR: {P_LR_mean:.6f}, Mean P_LL: {P_LL_mean:.6f}')
    print(f"Pauli error: {((P_01_mean+P_10_mean+P_11_mean)/(1-P_00_mean)):.6f},Leakage error: {((P_Leakage_mean)/(1-P_00_mean)):.6f},Loss error: {((P_Loss_mean)/(1-P_00_mean)):.6f}")
    print('-'*50)
    print('All mean:')
    P_list.append([P_00_mean, P_01_mean, P_10_mean, P_11_mean, 
                    P_00_withL_mean, P_01_withL_mean, P_10_withL_mean, P_11_withL_mean,
                    P_Leakage_mean, P_Leakage_without00_mean, 
                    P_Loss_mean, P_RL_mean, P_LR_mean, P_LL_mean
                    ])
    P_list_mean = np.array(P_list).mean(axis=0)
    print(f'Mean P_00: {P_list_mean[0]:.6f}, Mean P_01: {P_list_mean[1]:.6f}, Mean P_10: {P_list_mean[2]:.6f}, Mean P_11: {P_list_mean[3]:.6f},\nMean P_00_withL: {P_list_mean[4]:.6f}, Mean P_01_withL: {P_list_mean[5]:.6f}, Mean P_10_withL: {P_list_mean[6]:.6f}, Mean P_11_withL: {P_list_mean[7]:.6f},\nMean P_Leakage: {P_list_mean[8]:.6f},Mean P_Leakage_without00: {P_list_mean[9]:.6f}\nMean P_Loss: {P_list_mean[10]:.6f}, Mean P_RL: {P_list_mean[11]:.6f}, Mean P_LR: {P_list_mean[12]:.6f}, Mean P_LL: {P_list_mean[13]:.6f}')
    print(f"Pauli error: {((P_list_mean[1]+P_list_mean[2]+P_list_mean[3])/(1-P_list_mean[0])):.6f},Leakage error: {((P_list_mean[8])/(1-P_list_mean[0])):.6f},Loss error: {((P_list_mean[10])/(1-P_list_mean[0])):.6f}")
    print('/'*50)
    print('Error budget:')
    P_error_list[7] = [1-P_list_mean[0],P_list_mean[1]+P_list_mean[2]+P_list_mean[3],P_list_mean[8],P_list_mean[10]]
    P_error = jnp.array(P_error_list)
    P_error_delta = P_error[1:] - P_error[:-1]
    P_error_delta = P_error_delta.at[:,1:].set(P_error_delta[:,1:]/P_error_delta[:,0].reshape(7,-1))
    print(P_error_delta)
    print('%'*50)

# P_error_list.append([1-P_00_mean,P_01_mean+P_10_mean+P_11_mean,P_Leakage_mean,P_Loss_mean])
# P_00_withL_list.append(P_00_withL_mean)
# P_survive_list.append(1-P_Loss_mean)

#%%
for w in range(1):
    sol = model.integrate_rho_multi_jax(tlist, amp_420, phase_420, amp_1013, [sol_mid[n] for n in range(12)])

    sol_mid = jnp.array([model.mid_state_decay(sol[n,-1]) for n in range(12)])

    sol = model.integrate_rho_multi_jax(tlist, amp_420, phase_420, amp_1013, [sol_mid[n] for n in range(12)])

    sol_mid = jnp.array([model.mid_state_decay(sol[n,-1]) for n in range(12)])

    fid_raw_mean = 0
    theta_mean = 0
    for n in range(12):
        fid_raw, theta = model.CZ_fidelity(sol_mid[n], psi0_list[n])
        fid_raw_mean += fid_raw / 12
        if n in [0,1,2,3,6,7,8,9]:
            theta_mean += theta / 8
    #     print("CZ fidelity:", fid_raw, " theta:", theta)                     
    # print('mean raw fidelity:', fid_raw_mean)
    # print('mean theta:', theta_mean)

    fid_mean = 0
    for n in range(12):
        fid, theta = model.CZ_fidelity(sol_mid[n], psi0_list[n], theta_mean)
        fid_mean += fid / 12
    #     print("CZ fidelity with mean theta:", fid)
    # print('mean fidelity with mean theta:', fid_mean)

    P_00_mean = 0
    P_01_mean = 0
    P_10_mean = 0
    P_11_mean = 0
    P_00_withL_mean = 0
    P_01_withL_mean = 0
    P_10_withL_mean = 0
    P_11_withL_mean = 0
    P_Loss_mean = 0
    P_RL_mean = 0
    P_LR_mean = 0
    P_LL_mean = 0
    P_Leakage_mean = 0
    P_Leakage_without00_mean = 0
    for n in range(12):
        sol_state_00 = model.CZ_back_to_00(sol_mid[n], n, psi0_list[n], theta_mean)
        P_00 = jnp.abs(sol_state_00[0,0])
        P_00_mean += P_00 / 12
        P_01 = jnp.abs(sol_state_00[1,1])
        P_01_mean += P_01 / 12
        P_10 = jnp.abs(sol_state_00[10,10])
        P_10_mean += P_10 / 12
        P_11 = jnp.abs(sol_state_00[11,11])
        P_11_mean += P_11 / 12
        P_00_withL = jnp.sum(jnp.abs(jnp.diag(sol_state_00)[jnp.array([0,8,80,88])]))
        P_00_withL_mean += P_00_withL / 12
        P_01_withL = jnp.sum(jnp.abs(jnp.diag(sol_state_00)[jnp.array([1,9,81,89])]))
        P_01_withL_mean += P_01_withL / 12
        P_10_withL = jnp.sum(jnp.abs(jnp.diag(sol_state_00)[jnp.array([10,18,90,98])]))
        P_10_withL_mean += P_10_withL / 12
        P_11_withL = jnp.sum(jnp.abs(jnp.diag(sol_state_00)[jnp.array([11,19,91,99])]))
        P_11_withL_mean += P_11_withL / 12
        P_Loss = jnp.sum(jnp.abs(jnp.diag(sol_state_00)[jnp.array([
            5,6,7,15,16,17,
            50,51,55,56,57,58,59,
            60,61,65,66,67,68,69,
            70,71,75,76,77,78,79,
            85,86,87,95,96,97,
        ])]))
        P_Loss_mean += P_Loss / 12
        P_RL = jnp.sum(jnp.abs(jnp.diag(sol_state_00)[jnp.array([
            50,51,58,59,60,61,68,69,70,71,78,79,
        ])]))
        P_RL_mean += P_RL / 12
        P_LR = jnp.sum(jnp.abs(jnp.diag(sol_state_00)[jnp.array([
            5,6,7,15,16,17,85,86,87,95,96,97,
        ])]))
        P_LR_mean += P_LR / 12
        P_LL = jnp.sum(jnp.abs(jnp.diag(sol_state_00)[jnp.array([
            55,56,57,65,66,67,75,76,77,
        ])]))
        P_LL_mean += P_LL / 12
        P_Leakage = jnp.sum(jnp.abs(jnp.diag(sol_state_00)[jnp.array([
            8,80,88,
            9,81,89,
            18,90,98,
            19,91,99,
        ])]))
        P_Leakage_mean += P_Leakage / 12
        P_Leakage_without00 = jnp.sum(jnp.abs(jnp.diag(sol_state_00)[jnp.array([
            9,81,89,
            18,90,98,
            19,91,99,
        ])]))
        P_Leakage_without00_mean += P_Leakage_without00 / 12
        # print(f"State {n}: P_00: {P_00:.6f},\nP_00_withL: {P_00_withL:.6f}, P_01_withL: {P_01_withL:.6f}, P_10_withL: {P_10_withL:.6f}, P_11_withL: {P_11_withL:.6f},\nP_Loss: {P_Loss:.6f}, P_RL: {P_RL:.6f}, P_LR: {P_LR:.6f}, P_LL: {P_LL:.6f}")
    print(f'Mean P_00: {P_00_mean:.6f}, Mean P_01: {P_01_mean:.6f}, Mean P_10: {P_10_mean:.6f}, Mean P_11: {P_11_mean:.6f},\nMean P_00_withL: {P_00_withL_mean:.6f}, Mean P_01_withL: {P_01_withL_mean:.6f}, Mean P_10_withL: {P_10_withL_mean:.6f}, Mean P_11_withL: {P_11_withL_mean:.6f},\nMean P_Leakage: {P_Leakage_mean:.6f},Mean P_Leakage_without00: {P_Leakage_without00_mean:.6f}\nMean P_Loss: {P_Loss_mean:.6f}, Mean P_RL: {P_RL_mean:.6f}, Mean P_LR: {P_LR_mean:.6f}, Mean P_LL: {P_LL_mean:.6f}')
    print(f"Pauli error: {((P_01_mean+P_10_mean+P_11_mean)/(1-P_00_mean)):.6f},Leakage error: {((P_Leakage_mean)/(1-P_00_mean)):.6f},Loss error: {((P_Loss_mean)/(1-P_00_mean)):.6f}")

    P_00_withL_list.append(P_00_withL_mean)
    P_survive_list.append(1-P_Loss_mean)

    roundd = [2*i+1 for i in range(w+2)]
    def exp(x,F,A):
        return A*F**x

    Popt, _ = curve_fit(exp,roundd,P_00_withL_list,p0=[1,1])
    print('Fideliy P_00_withL: ',Popt)
    plt.plot(roundd,exp(roundd,*Popt))
    plt.scatter(roundd,P_00_withL_list)
    Popt, _ = curve_fit(exp,roundd,P_survive_list,p0=[1,1])
    print('Fideliy P_survive: ',Popt)
    plt.plot(roundd,exp(roundd,*Popt))
    plt.scatter(roundd,P_survive_list)
    plt.show()

#%%
theta_list = jnp.linspace(-jnp.pi, jnp.pi, 100)
for n in range(12):
    f_list = []
    for theta in theta_list:
        U_theta = jnp.array(block_diag(gates.rz(theta).full(),np.eye(8)),dtype=jnp.complex128)
        U_theta_tq = jnp.kron(U_theta,U_theta)

        f = jnp.abs(jnp.conj(U_theta_tq @ model.CZ_ideal() @ psi0_list[n]).T @ sol_mid[n] @ (U_theta_tq @ model.CZ_ideal() @ psi0_list[n]))
        f_list.append(f[0,0])
    plt.plot(theta_list, jnp.array(f_list))

#%%
param_f_result = []
params_step = [0.008192, 0.008192, 0.008192, 0.008192, 0.008192]
params = [1.23,-0.1135,1.043,-0.7256,0.004]
f = []
i = -1
while True:
    i += 1 

    if i % 5 == 0:
        f = []
        j = (i//5) % 5
        params_list = jnp.linspace(params[j] - params_step[j], params[j] + params_step[j], 5)

    params[j] = params_list[i%5]
    tf, A, omegaf, phi0, deltaf = params

    amp_420 = lambda t:1
    phase_420 = lambda t : 2 * jnp.pi * (A * jnp.cos(2 * jnp.pi * omegaf * Omega * t - phi0) + deltaf * Omega * t)
    amp_1013 = lambda t:1

    tlist = jnp.linspace(0, tf/Omega, 2)

    psi0_list = model.SSS_initial_state_list
    sol = model.integrate_rho_multi_jax(tlist, amp_420, phase_420, amp_1013, [model.psi_to_rho(psi0) for psi0 in psi0_list])

    f.append(jnp.mean(jnp.array([model.CZ_fidelity(model.mid_state_decay(sol[n,-1]),psi0_list[n]) for n in range(12)])))
    print("params:", params)

    if i % 5 == 4:
        plt.plot(params_list, jnp.array(f))
        plt.show()

        params[j] = params_list[jnp.argmax(jnp.array(f))]
        print("best params so far:", params)

        param_f_result.append(jnp.concatenate([jnp.array(params), params_list, jnp.array(f)]))
        jnp.save("param_f_result.npy", jnp.array(param_f_result))

        params_step[j] = params_step[j] / 2

        if jnp.max(jnp.array(params_step)) < 0.0001 or jnp.max(jnp.array(f)) > 0.997:
            break

# %%