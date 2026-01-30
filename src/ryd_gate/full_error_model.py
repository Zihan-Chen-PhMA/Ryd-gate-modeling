"""JAX-based two-atom density-matrix simulator with Rydberg blockade and decay."""

# %%
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.linalg import block_diag
from scipy.optimize import minimize_scalar
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
                 ryd_decay = True, mid_decay = True, 
                 distance = 3,#um
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

    def init_420_ham(self):
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
            self._420_amp * jnp.exp(-1j*phase_420) * self.H_420_tq_sparse_value,
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

    def rydberg_RD_branch_ratio(self):
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

    def mid_branch_ratio(self,level_label):
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
    
    def SSS_rec(self,idx):

        p1 = [0,0,1,1,2,0,2,0,1,0,3,1]
        p2 = [3,1,0,2,0,0,1,1,1,0,0,0]
        p3 = [0,0,0,0,0,0,1,1,1,1,1,1]

        g1 = gates.qrot(np.pi/2,p2[idx]*np.pi/2)*gates.qrot(np.pi/2,p1[idx]*np.pi/2)
        g2 = gates.qrot(-np.pi/2,0)*gates.qrot(np.pi/2,p3[idx]*np.pi/2)

        G1 = jnp.array(block_diag(g1.full(),np.eye(8)),dtype=jnp.complex128)
        G2 = jnp.array(block_diag(g2.full(),np.eye(8)),dtype=jnp.complex128)

        return jnp.kron(G2,G2) @ self.CZ_ideal() @ jnp.kron(G1,G1)

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
    
    def CZ_back_to_00(self, state_final, state_idx, state_initial = None, theta = None):
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
