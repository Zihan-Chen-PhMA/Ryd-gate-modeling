"""Simplified CZ gate simulator using SciPy ODE solvers with hyperfine structure."""

import numpy as np
from arc import *
import matplotlib.pyplot as plt
from scipy import signal
from scipy import optimize
from cmath import *
import copy as copy
from scipy import integrate
from scipy.optimize import minimize, minimize_scalar
from scipy import constants
from scipy import interpolate
import warnings
from qutip import Bloch
import qutip as qt
from qutip import tensor, basis, Qobj, mesolve, ket2dm, fidelity
import pickle

from ryd_gate.blackman import blackman_pulse
      

class CZGateSimulator():
    '''
    这个类用于模拟简化的CPHASE门
    decayflag = True 选择是否考虑Rydberg的自发辐射效应。

    Methods:
    --optimization_TO：
    接受参数[x[0],...,x[5]], 420时变相位为：
    x[0]* cos (x[1]*self.rabi_eff * t + x[2]) + x[3]*self.rabi_eff*t
     x[4] 用于标记CPHASE参数diag[1,exp(-ix[4]),exp(-ix[4]),exp(-2x[4]i) * exp(-np.pi * i)]
     x[5] 用于标记门的时间 T = x[5]*self.time_scale = x[5]*2*np.pi/self.rabi_eff

    --optimization_AR：
    Input:
    x =[x[0],...,x[7]], 420时变相位为：
        x[1] * sin(x[0] * self.rabi_eff * t + x[2]) + x[3] * sin(2x[0] * self.rabi_eff * t + x[4]) + x[5] * self.rabi_eff * t 
        x[6] 用于标记门的时间 T = x[6]*self.time_scale = x[5]*2*np.pi/self.rabi_eff
        x[7] 用于标记CPHASE参数diag[1,exp(-ix[7]),exp(-ix[7]),exp(-2x[7]i) * exp(-np.pi * i)]
    '''

    def __init__(self, decayflag, param_set='our' ,strategy = 'AR', blackmanflag = True) -> None:
        """
        初始化，这里设置了实验参数，需要依据具体实验设置修改
        """
        self.param_set = param_set
        self.strategy = strategy
        if self.param_set == 'our':
            self.atom = Rubidium87()
            #Our parameter
            self.ryd_level = 70
            # 认为 rabi420 = rabi1013, 有效rabi频率为5MHz
            self.Delta = 2*np.pi*(6.1)*10**(9)  
            self.rabi_eff = 2 * np.pi * (7) * 10**6
            self.rabi_420 = np.sqrt(2 * self.Delta * self.rabi_eff)
            self.rabi_1013 = np.sqrt(2 * self.Delta * self.rabi_eff)
            self.time_scale = 2*np.pi/self.rabi_eff
            ####Our ratio
            self.d_mid_ratio = (self.atom.getDipoleMatrixElement(5,0,0.5,0.5,6,1,1.5,-0.5,-1)/
                                self.atom.getDipoleMatrixElement(5,0,0.5,-0.5,6,1,1.5,-1.5,-1))
            self.d_ryd_ratio = (self.atom.getDipoleMatrixElement(6,1,1.5,-0.5,self.ryd_level,0,0.5,0.5,1)/
                                self.atom.getDipoleMatrixElement(6,1,1.5,-1.5,self.ryd_level,0,0.5,-0.5,1))
            self.rabi_420_garbage = self.rabi_420*self.d_mid_ratio
            self.rabi_1013_garbage = self.rabi_1013*self.d_ryd_ratio

            ##Our Rydberg interaction&shift
            self.v_ryd = 2*np.pi*(874)*10**(9)/ 4**6 
            self.v_ryd_garb = 2*np.pi*(874)*10**(9)/ 4**6 
            self.ryd_zeeman_shift = -2*np.pi*(56)*10**(6)

            ##Our Decay rate parameters:
            self.mid_state_decay_rate = (1/(110*10**(-9)))
            self.mid_garb_decay_rate = (1/(110*10**(-9)))
            self.ryd_state_decay_rate = (1/(151.55*10**(-6)))
            self.ryd_garb_decay_rate = (1/(151.55*10**(-6)))

            self.blackmanflag = blackmanflag
            self.tq_ham_const = self._tq_ham_const(decayflag)
            self.tq_ham_420 = self._tq_ham_420_our()
            self.tq_ham_1013 = self._tq_ham_1013_our()
            self.tq_ham_420_conj = self._tq_ham_420_our().conj().T
            self.tq_ham_1013_conj = self._tq_ham_1013_our().conj().T
            self.t_rise = 20*10**(-9)   # Blackman pulse rise time 
        elif self.param_set == 'lukin':
            ##Lukin's parameter
            self.atom = Rubidium87()
            self.ryd_level = 53
            self.Delta = 2*np.pi*(7.8)*10**(9)  
            self.rabi_420 = 2*np.pi * 237 *10**6
            self.rabi_1013 = 2*np.pi * 303 *10**6
            # self.rabi_1013 = self.rabi_420 * np.sqrt(4/3)
            self.rabi_eff = self.rabi_420 * self.rabi_1013 /(2 * self.Delta)
            self.time_scale = 2*np.pi/self.rabi_eff
            ####Lukin's ratio
            self.d_mid_ratio = (self.atom.getDipoleMatrixElement(5,0,0.5,-0.5,6,1,1.5,0.5,1)/
                                self.atom.getDipoleMatrixElement(5,0,0.5,0.5,6,1,1.5,1.5,1))
            self.d_ryd_ratio = (self.atom.getDipoleMatrixElement(6,1,1.5,0.5,self.ryd_level,0,0.5,-0.5,-1)/
                                self.atom.getDipoleMatrixElement(6,1,1.5,1.5,self.ryd_level,0,0.5,0.5,-1))
            self.rabi_420_garbage = self.rabi_420*self.d_mid_ratio
            self.rabi_1013_garbage = self.rabi_1013*self.d_ryd_ratio

            ##Lukin's Rydberg interaction&shift
            self.v_ryd = 2*np.pi*(450)*10**(6)
            self.v_ryd_garb = 2*np.pi*(450)*10**(6)
            self.ryd_zeeman_shift = -2*np.pi*(24)*10**(8)

            ##Lukin's Decay rate parameters:
            self.mid_state_decay_rate = (1/(110*10**(-9)))
            self.mid_garb_decay_rate = (1/(110*10**(-9)))
            self.ryd_state_decay_rate = (1/(88*10**(-6)))
            self.ryd_garb_decay_rate = (1/(88*10**(-6)))

            self.blackmanflag = blackmanflag
            self.tq_ham_const = self._tq_ham_const(decayflag)
            self.tq_ham_420 = self._tq_ham_420_lukin()
            self.tq_ham_1013 = self._tq_ham_1013_lukin()
            self.tq_ham_420_conj = self._tq_ham_420_lukin().conj().T
            self.tq_ham_1013_conj = self._tq_ham_1013_lukin().conj().T
            self.t_rise = 20*10**(-9)   # Blackman pulse rise time 
        else:
            raise ValueError(f"Unknown parameter set: '{self.param_set}'. Choose 'our' or 'lukin'.")
        pass

    # ------------------------------------------------------------------
    # --- UNIFIED PUBLIC METHODS (Dispatchers) ---
    # ------------------------------------------------------------------

    def optimize(self, x_initial):
        """
        Runs the optimization procedure for the configured strategy.
        """
        if self.strategy == 'TO':
            return self._optimization_TO(x_initial)
        elif self.strategy == 'AR':
            return self._optimization_AR(x_initial)
        else:
            raise ValueError(f"Unknown strategy: '{self.strategy}'. Choose 'TO' or 'AR'.")

    def avg_fidelity(self, x):
        """
        Calculates the average gate fidelity for the configured strategy.
        """
        if self.strategy == 'TO':
            return self._avg_fidelity_TO(x)
        elif self.strategy == 'AR':
            return self._avg_fidelity_AR(x)
        else:
            raise ValueError(f"Unknown strategy: '{self.strategy}'. Choose 'TO' or 'AR'.")
            
    def diagonise_plot(self, x, initial_state):
        """
        Generates and saves a plot of population evolution for a given initial state.
        """
        if self.strategy == 'TO':
            return self._diagonise_plot_TO(x, initial_state)
        elif self.strategy == 'AR':
            return self._diagonise_plot_AR(x, initial_state)
        else:
            raise ValueError(f"Unknown strategy: '{self.strategy}'. Choose 'TO' or 'AR'.")

    def plot_bloch(self, x, save=True):
        """
        Generates and optionally saves Bloch sphere plots for the '01' -> '0r'
        and '11' -> 'W' transitions. Only implemented for TO strategy.
        """
        if self.strategy == 'TO':
            return self._plotBloch_TO(x, save)
        else:
            print("Bloch sphere plot is only implemented for the 'TO' strategy.")
            return

    # ------------------------------------------------------------------
    # --- HAMILTONIAN AND OPERATOR DEFINITIONS ---
    # ------------------------------------------------------------------

    def _tq_ham_const(self,decayflag):
        ham_tq_mat = np.zeros((49,49),dtype=np.complex128)
        ham_sq_mat = np.zeros((7,7),dtype=np.complex128)
        delta = 0
        middecay = self.mid_state_decay_rate if decayflag else 0
        ryddecay = self.ryd_state_decay_rate if decayflag else 0
        ham_sq_mat[2][2] = (self.Delta - 2*np.pi*51*10**(6) - 1j*middecay/2)
        ham_sq_mat[3][3] = (self.Delta - 1j*middecay/2) 
        ham_sq_mat[4][4] = (self.Delta + 2*np.pi*87*10**(6) -1j*middecay/2)
        ham_sq_mat[5][5] = (delta - 1j*ryddecay/2) 
        ham_sq_mat[6][6] = (delta + self.ryd_zeeman_shift - 
                            1j*ryddecay/2)

        ham_tq_mat = ham_tq_mat + np.kron(np.eye(7),ham_sq_mat)
        ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat,np.eye(7))
        ham_vdw_mat = np.zeros((7,7))
        ham_vdw_mat_garb = np.zeros((7,7))
        ham_vdw_mat[5][5] = 1
        ham_vdw_mat_garb[6][6] = 1
        ham_tq_mat = ham_tq_mat + self.v_ryd*np.kron(ham_vdw_mat + ham_vdw_mat_garb,
                                                ham_vdw_mat + ham_vdw_mat_garb)
        return ham_tq_mat
   
    def _occ_operator(self, index):
        '''
        Operator |i><i| \otimes I +  I \otimes |i><i| 
        '''
        oper_tq = np.zeros((49,49),dtype=np.complex128)
        oper_sq = np.zeros((7,7),dtype=np.complex128)
        oper_sq[index][index] = 1
        oper_tq = oper_tq + np.kron(np.eye(7),oper_sq)
        oper_tq = oper_tq + np.kron(oper_sq, np.eye(7))
        return oper_tq

    ## Our Hamiltonian
    def _tq_ham_1013_our(self):
        ham_tq_mat = np.zeros((49,49),dtype=np.complex128)
        ham_sq_mat = np.zeros((7,7),dtype=np.complex128)
   
        ham_sq_mat[5][2] = (self.rabi_1013/2)*CG(3/2,-3/2,3/2,1/2,
                                                 1,-1)
        ham_sq_mat[5][3] = (self.rabi_1013/2)*CG(3/2,-3/2,3/2,1/2,
                                                 2,-1)
        ham_sq_mat[5][4] = (self.rabi_1013/2)*CG(3/2,-3/2,3/2,1/2,
                                                 3,-1)
    
        ham_sq_mat[6][2] = (self.rabi_1013_garbage/2)*CG(
                                3/2,-1/2,3/2,-1/2,1,-1)
        ham_sq_mat[6][3] = (self.rabi_1013_garbage/2)*CG(
                                3/2,-1/2,3/2,-1/2,2,-1)
        ham_sq_mat[6][4] = (self.rabi_1013_garbage/2)*CG(
                                3/2,-1/2,3/2,-1/2,3,-1)

        ham_tq_mat = ham_tq_mat + np.kron(np.eye(7),ham_sq_mat)
        ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat,np.eye(7))

        return ham_tq_mat

    def _tq_ham_420_our(self):
        ham_tq_mat = np.zeros((49,49),dtype=np.complex128)
        ham_sq_mat = np.zeros((7,7),dtype=np.complex128)
        ham_sq_mat[2][1] = (self.rabi_420*
                                CG(3/2,-3/2,3/2,1/2,1,-1) + 
                            self.rabi_420_garbage*
                                CG(3/2,-1/2,3/2,-1/2,1,-1)
                            )/2
        ham_sq_mat[3][1] = (self.rabi_420*
                                CG(3/2,-3/2,3/2,1/2,2,-1) + 
                            self.rabi_420_garbage*
                                CG(3/2,-1/2,3/2,-1/2,2,-1)
                            )/2
        ham_sq_mat[4][1] = (self.rabi_420*
                                CG(3/2,-3/2,3/2,1/2,3,-1) + 
                            self.rabi_420_garbage*
                                CG(3/2,-1/2,3/2,-1/2,3,-1)
                            )/2
        ham_tq_mat = ham_tq_mat + np.kron(np.eye(7),ham_sq_mat)
        ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat,np.eye(7))
        return ham_tq_mat

    ##Lukin's Hamiltonian
    def _tq_ham_420_lukin(self):
        ham_tq_mat = np.zeros((49,49),dtype=np.complex128)
        ham_sq_mat = np.zeros((7,7),dtype=np.complex128)
        ham_sq_mat[2][1] = (self.rabi_420*
                                CG(3/2,3/2,3/2,-1/2,1,1) + 
                            self.rabi_420_garbage*
                                CG(3/2,1/2,3/2,1/2,1,1)
                            )/2
        ham_sq_mat[3][1] = (self.rabi_420*
                                CG(3/2,3/2,3/2,-1/2,2,1) + 
                            self.rabi_420_garbage*
                                CG(3/2,1/2,3/2,1/2,2,1)
                            )/2
        ham_sq_mat[4][1] = (self.rabi_420*
                                CG(3/2,3/2,3/2,-1/2,3,1) + 
                            self.rabi_420_garbage*
                                CG(3/2,1/2,3/2,1/2,3,1)
                            )/2
        ham_tq_mat = ham_tq_mat + np.kron(np.eye(7),ham_sq_mat)
        ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat,np.eye(7))
        return ham_tq_mat

    def _tq_ham_1013_lukin(self):
        ham_tq_mat = np.zeros((49,49),dtype=np.complex128)
        ham_sq_mat = np.zeros((7,7),dtype=np.complex128)
   
        ham_sq_mat[5][2] = (self.rabi_1013/2)*CG(3/2,3/2,3/2,-1/2, 1,1)
        ham_sq_mat[5][3] = (self.rabi_1013/2)*CG(3/2,3/2,3/2,-1/2,2,1)
        ham_sq_mat[5][4] = (self.rabi_1013/2)*CG(3/2,3/2,3/2,-1/2,3,1)
    
        ham_sq_mat[6][2] = (self.rabi_1013_garbage/2)*CG(3/2,1/2,3/2,1/2,1,1)
        ham_sq_mat[6][3] = (self.rabi_1013_garbage/2)*CG(3/2,1/2,3/2,1/2,2,1)
        ham_sq_mat[6][4] = (self.rabi_1013_garbage/2)*CG(3/2,1/2,3/2,1/2,3,1)

        ham_tq_mat = ham_tq_mat + np.kron(np.eye(7),ham_sq_mat)
        ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat,np.eye(7))

        return ham_tq_mat    


    def _decay_integrate(self, t_list, occ_list, decay_rate):
        poly_interpolation = interpolate.CubicSpline(t_list, occ_list)
        args = (poly_interpolation, decay_rate)
        def fun(t, y, poly_interpolation, decay_rate):
            diff = decay_rate*poly_interpolation.__call__(t)
            return np.array([diff])
        t_span = [0, t_list[-1]]
        result = integrate.solve_ivp(fun, t_span, np.array([0]), 
                                     t_eval=t_list, args=args,
                                     method='DOP853',
                                     # method = 'RK45',
                                     # method='BDF',
                                     rtol=1e-8,
                                     atol=1e-12)
        return np.array(result.y)

    # ------------------------------------------------------------------
    # --- TIME OPTIMAL (TO) STRATEGY: INTERNAL IMPLEMENTATIONS ---
    # ------------------------------------------------------------------
 
    def _get_gate_result_TO(self, phase_amp,omega, phase_init, delta, t_gate, state_mat) -> np.array:
        '''
        This function get the gate result y give all the parameters of time-optimal(TO) pulses
                                                for y, y[:,t] gives the state vector at time t
        no lase noise and no decay is considered. 

        This function is used in the pulse parameter optimiation functions below:
        avg_fidelity_TO();

        '''
        def fun(t, y, phase_init, omega, phase_amp, delta, t_gate):
            phase_420 = np.exp(-1j*(phase_amp*
                            np.cos(omega*t+phase_init)+delta*t))
            phase_420_conj = np.conjugate(phase_420)
            amplitude = blackman_pulse(t, self.t_rise, t_gate) if self.blackmanflag else 1
            ham_tq_mat = np.zeros((49,49),dtype=np.complex128)
            ham_tq_mat = ham_tq_mat + (self.tq_ham_const + 
                        amplitude*phase_420*self.tq_ham_420 + 
                        amplitude*phase_420_conj*self.tq_ham_420_conj+
                        self.tq_ham_1013 + 
                        self.tq_ham_1013_conj)
            diff = -1j*ham_tq_mat
            y_arr = np.reshape(np.array(y),(-1,1))
            return np.reshape(np.matmul(diff,y_arr),(-1))
        
        args = (phase_init, omega, phase_amp, delta, t_gate)
        t_span = [0, t_gate]
        t_eval = np.linspace(0,t_gate,1000)
        result = integrate.solve_ivp(fun, t_span, state_mat, t_eval=t_eval,
                        args=args,
                        method='DOP853',
                        # method = 'RK45',
                        # method='BDF',
                        rtol=1e-8,
                        atol=1e-12)
        res = np.array(result.y)
        # plotlist = [abs(x)**2 for x in res[1,:]]
        # tlist = np.linspace(0,t_gate,len(plotlist))
        # plt.plot(tlist,plotlist)
        # plt.show()
        return res

    def _avg_fidelity_TO(self,x) -> float:

        theta = x[4]
        #Find the a01
        ini_state = np.kron([1+0j,0,0,0,0,0,0],[0,1+0j,0,0,0,0,0])
        res = self._get_gate_result_TO(phase_amp=x[0],
                                    omega=x[1]*self.rabi_eff,  
                                    phase_init=x[2],
                                    delta= x[3]*self.rabi_eff,
                                    t_gate = x[5]*self.time_scale,
                                    state_mat = ini_state)[:,-1]
        a01 = np.exp(-1.0j*theta) * ini_state.conj().dot(res.T)

        #Find the a11
        ini_state = np.kron([0,1+0j,0,0,0,0,0],[0,1+0j,0,0,0,0,0])
        res = self._get_gate_result_TO(phase_amp=x[0],
                                    omega=x[1]*self.rabi_eff,  
                                    phase_init=x[2],
                                    delta= x[3]*self.rabi_eff,
                                    t_gate = x[5]*self.time_scale,
                                    state_mat = ini_state)[:,-1]
        a11 = np.exp(-2.0j*theta - 1.0j*np.pi) * ini_state.conj().dot(res.T)
        # print(np.angle(a11/a01),np.abs(a01)**2,np.abs(a11)**2)
        # #output average gate Fidelity 
        avg_F = (1/20)*(abs(1 + 2*a01+a11)**2 + 1 + 2*abs(a01)**2 + abs(a11)**2)
        print(1 - avg_F)
        return 1 - avg_F
    
    def _optimization_TO(self,x):
        def callback_func(x):
            with open('opt_hf_new.txt','a') as f:
                for var in x:
                    f.write("{:.9f},".format(var))
                f.write("\n")
            print("Current iteration parameters:", x)

        bounds = ((-np.pi,np.pi),(-10,10),(-np.pi, np.pi),
                        (-2,2),(-np.inf, np.inf),(-np.pi, np.pi))
        optimres = minimize(fun=self._avg_fidelity_TO,
                            x0=x,
                            method='Nelder-Mead',
                            options={'disp':True,'fatol':1e-9},
                            bounds=bounds,
                            callback=callback_func)
        return optimres   

    def _diagonise_run_TO(self,x,initial):
        '''
        The input form of x = x[0],...,x[7] gives 
        x[1] * sin(x[0] * self.rabi_eff * t + x[2]) + x[3] * sin(2x[0] * self.rabi_eff * t + x[4]) + x[5] * self.rabi_eff * t 
        '''
        #Find the a11
        if initial == '11':
            ini_state = np.kron([0,1+0j,0,0,0,0,0],[0,1+0j,0,0,0,0,0])
        elif initial == '10':
            ini_state = np.kron([0,1+0j,0,0,0,0,0],[1+0j,0,0,0,0,0,0])
        elif initial == '01':
            ini_state = np.kron([1+0j,0,0,0,0,0,0],[0,1+0j,0,0,0,0,0])
        else:
            raise ValueError("unsupport initial state!")
        # res_list[:,t]记录了t \in t_gate * [0,999]/1000时刻的量子态
        res_list = self._get_gate_result_TO(phase_amp=x[0],
                                    omega=x[1]*self.rabi_eff, 
                                    phase_init=x[2],
                                    delta= x[3]*self.rabi_eff,
                                    t_gate = x[5]*self.time_scale,
                                    state_mat = ini_state)
        mid_state_occ_oper = (self._occ_operator(2) + self._occ_operator(3) + 
                              self._occ_operator(4))
        ryd_state_occ_oper = self._occ_operator(5)
        ryd_garb_state_occ_oper = self._occ_operator(6)
        mid_state_list = []
        ryd_state_list = []
        ryd_garb_list = []
        for col in range(len(res_list[0,:])):
            state_temp = res_list[:,col]
            # print(len(state_temp))
            mid_state_occ_temp = np.dot(np.conjugate(state_temp), 
                                        np.reshape(np.matmul(mid_state_occ_oper,
                                            np.reshape(state_temp,(-1,1))),(-1)))
            ryd_state_occ_temp = np.dot(np.conjugate(state_temp), 
                                        np.reshape(np.matmul(ryd_state_occ_oper,
                                            np.reshape(state_temp,(-1,1))),(-1)))
            ryd_garb_occ_temp = np.dot(np.conjugate(state_temp), 
                                        np.reshape(np.matmul(ryd_garb_state_occ_oper,
                                            np.reshape(state_temp,(-1,1))),(-1)))
            mid_state_list.append(np.abs(mid_state_occ_temp))
            ryd_state_list.append(np.abs(ryd_state_occ_temp))
            ryd_garb_list.append(np.abs(ryd_garb_occ_temp))
        mid_state_arr = np.array(mid_state_list)
        ryd_state_arr = np.array(ryd_state_list)
        ryd_garb_state_arr = np.array(ryd_garb_list)

        # mid_decay_temp = self._decay_integrate(t_eval, mid_state_arr, 
        #                                            self.mid_state_decay_rate)
        # ryd_decay_temp = self._decay_integrate(t_eval, ryd_state_arr,
        #                                        self.ryd_state_decay_rate)
        # ryd_garb_decay_temp = self._decay_integrate(t_eval, ryd_garb_state_arr,
        #                                         self.ryd_garb_decay_rate)
        return [mid_state_arr, ryd_state_arr, ryd_garb_state_arr]

    def _diagonise_plot_TO(self,x, initial):
        population_evolution = self._diagonise_run_TO(x, initial)
        t_gate = x[5] * self.time_scale
        time_axis_ns = np.linspace(0, t_gate * 1e9, len(population_evolution[0]))
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_axis_ns, population_evolution[0], label=f'Intermediate states population for |{initial}⟩ state', lw=2)
        ax.plot(time_axis_ns, population_evolution[1], label='Rydberg state |r⟩ population', linestyle='--', lw=2)
        ax.plot(time_axis_ns, population_evolution[2], label="Unwanted Rydberg |r'⟩ population", linestyle=':', lw=2)
        ax.set_title('Population Evolution During CPHASE Gate (TO)', fontsize=16)
        ax.set_xlabel('Time (ns)', fontsize=12)
        ax.set_ylabel('Population Probability', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'population_TO_{initial}.png')
        plt.show()


    def _plotBloch_TO(self,x,saveflag = True):
        basis_01 = np.kron([1+0j,0,0,0,0,0,0],[0,1+0j,0,0,0,0,0])
        basis_0r = np.kron([1+0j,0,0,0,0,0,0],[0,0,0,0,0,1+0j,0])
        
        basis_01_mat = np.reshape(basis_01,(-1,1))
        basis_0r_mat = np.reshape(basis_0r,(-1,1))


        # First Bloch 01-0r
        sigmaz = basis_01_mat.dot(basis_01_mat.T)  - basis_0r_mat.dot(basis_0r_mat.T)
        sigmax = basis_0r_mat.dot(basis_01_mat.T)  + basis_01_mat.dot(basis_0r_mat.T)
        sigmay = -1j*basis_01_mat.dot(basis_0r_mat.T)  + 1j*basis_0r_mat.dot(basis_01_mat.T)
        zlist = []
        xlist = []
        ylist = []

        ini_state = basis_01
        res = self._get_gate_result_TO(phase_amp=x[0],
                                    omega=x[1]*self.rabi_eff,  
                                    phase_init=x[2],
                                    delta= x[3]*self.rabi_eff,
                                    t_gate = x[5]*self.time_scale,
                                    state_mat = ini_state)

        for t in range(len(res[0,:])):
            state = res[:,t]
            zlist.append(np.matmul(state.reshape(-1).conj(),sigmaz).dot(state.reshape(-1,1))[0])
            xlist.append(np.matmul(state.reshape(-1).conj(),sigmax).dot(state.reshape(-1,1))[0])
            ylist.append(np.matmul(state.reshape(-1).conj(),sigmay).dot(state.reshape(-1,1))[0])
        b = Bloch()
        b.zlabel = [r'$ |01 \rangle$ ', r'$|0r\rangle$']
        pnts = [np.array(xlist),np.array(ylist),np.array(zlist)]
        b.point_color=['#CC6600']
        b.point_size = [5]
        b.point_marker = ['^']
        b.add_points(pnts)
        b.make_sphere()

         # Second Bloch 11-W
        basis_11 = np.kron([0,1+0j,0,0,0,0,0],[0,1+0j,0,0,0,0,0])
        basis_W =np.sqrt(1/2)*( np.kron([0,1+0j,0,0,0,0,0],[0,0,0,0,0,1+0j,0]) + np.kron([0,0,0,0,0,1+0j,0],[0,1+0j,0,0,0,0,0]))
        basis_11_mat = np.reshape(basis_11,(-1,1))
        basis_W_mat = np.reshape(basis_W,(-1,1))

        sigmaz = basis_11_mat.dot(basis_11_mat.T)  - basis_W_mat.dot(basis_W_mat.T)
        sigmax = basis_W_mat.dot(basis_11_mat.T)  + basis_11_mat.dot(basis_W_mat.T)
        sigmay = -1j*basis_11_mat.dot(basis_W_mat.T)  + 1j*basis_W_mat.dot(basis_11_mat.T)
        zlist = []
        xlist = []
        ylist = []

        ini_state = basis_11
        res = self._get_gate_result_TO(phase_amp=x[0],
                                    omega=x[1]*self.rabi_eff,  
                                    phase_init=x[2],
                                    delta= x[3]*self.rabi_eff,
                                    t_gate = x[5]*self.time_scale,
                                    state_mat = ini_state)

        for t in range(len(res[0,:])):
            state = res[:,t]
            zlist.append(np.matmul(state.reshape(-1).conj(),sigmaz).dot(state.reshape(-1,1))[0])
            xlist.append(np.matmul(state.reshape(-1).conj(),sigmax).dot(state.reshape(-1,1))[0])
            ylist.append(np.matmul(state.reshape(-1).conj(),sigmay).dot(state.reshape(-1,1))[0])
        b2 = Bloch()
        b2.zlabel = [r'$ |11 \rangle$ ', r'$|W\rangle$']
        pnts = [np.array(xlist),np.array(ylist),np.array(zlist)]
        b2.point_color=['r']
        b2.point_size = [5]
        b2.point_marker = ['^']
        b2.add_points(pnts)
        b2.make_sphere()
        if saveflag:
            b.save("10-r0_Bloch")
            b2.save("11-W_Bloch")
        plt.show()
    # ------------------------------------------------------------------
    # --- AMPLITUDE ROBUST (AR) STRATEGY: INTERNAL IMPLEMENTATIONS ---
    # ------------------------------------------------------------------
    def _get_gate_result_AR(self, omega, phase_amp1,phase_init1,phase_amp2,phase_init2, delta, t_gate, state_mat) -> np.array:
        '''
        This function get the gate result y give all the parameters of time-optimal(TO) pulses
                                                for y, y[:,t] gives the state vector at time t
        Lase noise and decay is considered. 
        420 laser takes the phase function as 
        phase_amp1 * sin(omega * t + phase_init1) + phase_amp2 * sin(2 * omega * t + phase_init2) +   delta*t

        This function is used in the pulse parameter optimiation functions below:
        avg_fidelity_AR();

        '''
        def fun(t, y, omega, phase_init1, phase_amp1, phase_init2, phase_amp2, delta, t_gate):
            phase_420 = np.exp(-1j*(
                phase_amp1 * np.sin( omega * t + phase_init1) + 
                phase_amp2 * np.sin( 2 * omega * t + phase_init2) + delta*t))
            phase_420_conj = np.conjugate(phase_420)
            amplitude = blackman_pulse(t, self.t_rise, t_gate) if self.blackmanflag else 1
            # amplitude = 1
            ham_tq_mat = np.zeros((49,49),dtype=np.complex128)
            ham_tq_mat = ham_tq_mat + (self.tq_ham_const + 
                        amplitude*phase_420*self.tq_ham_420 + 
                        amplitude*phase_420_conj*self.tq_ham_420_conj+
                        self.tq_ham_1013 + 
                        self.tq_ham_1013_conj)
            diff = -1j*ham_tq_mat
            y_arr = np.reshape(np.array(y),(-1,1))
            return np.reshape(np.matmul(diff,y_arr),(-1))
        
        args = (omega, phase_init1, phase_amp1,phase_init2, phase_amp2, delta, t_gate)
        t_span = [0, t_gate]
        t_eval = np.linspace(0,t_gate,1000)
        result = integrate.solve_ivp(fun, t_span, state_mat, t_eval=t_eval,
                        args=args,
                        method='DOP853',
                        # method = 'RK45',
                        # method='BDF',
                        rtol=1e-8,
                        atol=1e-12)
        res = np.array(result.y)
        # plotlist = [abs(x)**2 for x in res[1,:]]
        # tlist = np.linspace(0,t_gate,len(plotlist))
        # plt.plot(tlist,plotlist)
        # plt.show()
        return res

    def _avg_fidelity_AR(self,x) -> float:
        '''
        The input form of x = x[0],...,x[7] gives 
        x[1] * sin(x[0] * self.rabi_eff * t + x[2]) + x[3] * sin(2x[0] * self.rabi_eff * t + x[4]) + x[5] * self.rabi_eff * t 
        '''
        theta = x[-1]
        #Find the a01
        ini_state = np.kron([1+0j,0,0,0,0,0,0],[0,1+0j,0,0,0,0,0])
        res = self._get_gate_result_AR(omega=x[0]*self.rabi_eff,
                                    phase_amp1=x[1],
                                    phase_init1=x[2],
                                    phase_amp2=x[3], 
                                    phase_init2=x[4],
                                    delta= x[5]*self.rabi_eff,
                                    t_gate = x[6]*self.time_scale,
                                    state_mat = ini_state)[:,-1]
        a01 = np.exp(-1.0j*theta) * ini_state.conj().dot(res.T)

        #Find the a11
        ini_state = np.kron([0,1+0j,0,0,0,0,0],[0,1+0j,0,0,0,0,0])
        res = self._get_gate_result_AR(omega=x[0]*self.rabi_eff,
                                    phase_amp1=x[1],
                                    phase_init1=x[2],
                                    phase_amp2=x[3], 
                                    phase_init2=x[4],
                                    delta= x[5]*self.rabi_eff,
                                    t_gate = x[6]*self.time_scale,
                                    state_mat = ini_state)[:,-1]
        a11 = np.exp(-2.0j * theta - 1.0j*np.pi) * ini_state.conj().dot(res.T)
        # print(np.angle(a11/a01),np.abs(a01)**2,np.abs(a11)**2)
        # #output average gate Fidelity 
        avg_F = (1/20)*(abs(1 + 2*a01+a11)**2 + 1 + 2*abs(a01)**2 + abs(a11)**2)
        # print(1 - avg_F)
        return 1 - avg_F
    
    def _optimization_AR(self,x):
        def callback_func(x):
            with open('opt_hf_new.txt','a') as f:
                for var in x:
                    f.write("{:.9f},".format(var))
                f.write("\n")
            print("parameters:",x,'Infidelity:',self._avg_fidelity_AR(x))

        bounds = ((-10,10),(-np.pi,np.pi),(-np.pi, np.pi),(-np.pi,np.pi),(-np.pi, np.pi),
                        (-2,2),(-np.inf, np.inf),(-np.pi, np.pi))
        optimres = minimize(fun=self._avg_fidelity_AR,
                            x0=x,
                            method='Nelder-Mead',
                            options={'disp':True,'fatol':1e-9},
                            bounds=bounds,
                            callback=callback_func)
        return optimres   

    def _diagonise_run_AR(self,x,initial):
        '''
        The input form of x = x[0],...,x[7] gives 
        x[1] * sin(x[0] * self.rabi_eff * t + x[2]) + x[3] * sin(2x[0] * self.rabi_eff * t + x[4]) + x[5] * self.rabi_eff * t 
        '''
        theta = x[-1]
        #Find the a11
        if initial == '11':
            ini_state = np.kron([0,1+0j,0,0,0,0,0],[0,1+0j,0,0,0,0,0])
        elif initial == '10':
            ini_state = np.kron([0,1+0j,0,0,0,0,0],[1+0j,0,0,0,0,0,0])
        elif initial == '01':
            ini_state = np.kron([1+0j,0,0,0,0,0,0],[0,1+0j,0,0,0,0,0])
        else:
            raise ValueError("unsupport initial state!")
        # res_list[:,t]记录了t \in t_gate * [0,999]/1000时刻的量子态
        res_list = self._get_gate_result_AR(omega=x[0]*self.rabi_eff,
                                    phase_amp1=x[1],
                                    phase_init1=x[2],
                                    phase_amp2=x[3], 
                                    phase_init2=x[4],
                                    delta= x[5]*self.rabi_eff,
                                    t_gate = x[6]*self.time_scale,
                                    state_mat = ini_state)
        mid_state_occ_oper = (self._occ_operator(2) + self._occ_operator(3) + 
                              self._occ_operator(4))
        ryd_state_occ_oper = self._occ_operator(5)
        ryd_garb_state_occ_oper = self._occ_operator(6)
        mid_state_list = []
        ryd_state_list = []
        ryd_garb_list = []
        for col in range(len(res_list[0,:])):
            state_temp = res_list[:,col]
            # print(len(state_temp))
            mid_state_occ_temp = np.dot(np.conjugate(state_temp), 
                                        np.reshape(np.matmul(mid_state_occ_oper,
                                            np.reshape(state_temp,(-1,1))),(-1)))
            ryd_state_occ_temp = np.dot(np.conjugate(state_temp), 
                                        np.reshape(np.matmul(ryd_state_occ_oper,
                                            np.reshape(state_temp,(-1,1))),(-1)))
            ryd_garb_occ_temp = np.dot(np.conjugate(state_temp), 
                                        np.reshape(np.matmul(ryd_garb_state_occ_oper,
                                            np.reshape(state_temp,(-1,1))),(-1)))
            mid_state_list.append(np.abs(mid_state_occ_temp))
            ryd_state_list.append(np.abs(ryd_state_occ_temp))
            ryd_garb_list.append(np.abs(ryd_garb_occ_temp))
        mid_state_arr = np.array(mid_state_list)
        ryd_state_arr = np.array(ryd_state_list)
        ryd_garb_state_arr = np.array(ryd_garb_list)

        # mid_decay_temp = self._decay_integrate(t_eval, mid_state_arr, 
        #                                            self.mid_state_decay_rate)
        # ryd_decay_temp = self._decay_integrate(t_eval, ryd_state_arr,
        #                                        self.ryd_state_decay_rate)
        # ryd_garb_decay_temp = self._decay_integrate(t_eval, ryd_garb_state_arr,
        #                                         self.ryd_garb_decay_rate)
        return [mid_state_arr, ryd_state_arr, ryd_garb_state_arr]

    def _diagonise_plot_AR(self,x, initial):
        population_evolution = self._diagonise_run_AR(x, initial)
        t_gate = x[6] * self.time_scale
        time_axis_ns = np.linspace(0, t_gate * 1e9, len(population_evolution[0]))
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_axis_ns, population_evolution[0], label=f'Intermediate states population for |{initial}⟩ state', lw=2)
        ax.plot(time_axis_ns, population_evolution[1], label='Rydberg state |r⟩ population', linestyle='--', lw=2)
        ax.plot(time_axis_ns, population_evolution[2], label="Unwanted Rydberg |r'⟩ population", linestyle=':', lw=2)
        ax.set_title('Population Evolution During CPHASE Gate (AR)', fontsize=16)
        ax.set_xlabel('Time (ns)', fontsize=12)
        ax.set_ylabel('Population Probability', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'population_AR_{initial}.png')
        plt.show()

