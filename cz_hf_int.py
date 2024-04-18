import numpy as np 
from arc import * 
import matplotlib.pyplot as plt 
from scipy import signal
from scipy import optimize
from qutip import *
from cmath import * 
import copy as copy
from scipy import integrate
from scipy import constants
from scipy import interpolate
import warnings

from blackman import blackman_pulse, blackman_pulse_sqrt
from utils import color_dict

class PauliTwo():

    def __init__(self, phase_0, phase_1, phase_2):
        self.phase_0 = phase_0
        self.phase_1 = phase_1
        self.phase_2 = phase_2
        self.phase_dict = {'00': self.phase_0,
                           '01': self.phase_1,
                           '10': self.phase_1,
                           '11': self.phase_2}

    def basis_vec_2d(self,index):
        if index == '00':
            return np.reshape(np.array([1,0,0,0]),
                              (-1,1))
        elif index == '01':
            return np.reshape(np.array([0,1,0,0]),
                              (-1,1))
        elif index == '10':
            return np.reshape(np.array([0,0,1,0]),
                              (-1,1))
        elif index == '11':
            return np.reshape(np.array([0,0,0,1]),
                              (-1,1))
    
    def basis_dual_vec_2d(self,index):
        if index == '00':
            return np.reshape(np.array([1,0,0,0]),
                              (1,-1))
        elif index == '01':
            return np.reshape(np.array([0,1,0,0]),
                              (1,-1))
        elif index == '10':
            return np.reshape(np.array([0,0,1,0]),
                              (1,-1))
        elif index == '11':
            return np.reshape(np.array([0,0,0,1]),
                              (1,-1))
    
    def basis_dm(self, index, index_dual):
        basis_2d = self.basis_vec_2d(index)
        basis_dual_2d = self.basis_dual_vec_2d(index_dual)
        return np.matmul(basis_2d,basis_dual_2d)
    

    def pp_dm(self):
        """
        DM for |++> state
        """
        basis_vec = np.array([0.5,0.5,0.5,0.5])
        basis_2d = np.reshape(basis_vec,(-1,1))
        basis_dual_2d = np.reshape(basis_vec,(1,-1))
        return np.matmul(basis_2d,basis_dual_2d)
    
    def bell_pp_dm(self):
        u_mat = np.diag([1, np.exp(1j*self.phase_1),
                         np.exp(1j*self.phase_1),
                         np.exp(1j*self.phase_2)])
        u_mat_dag = np.transpose(np.conjugate(u_mat))
        dm_temp = np.matmul(u_mat,self.pp_dm())
        dm_ret = np.matmul(dm_temp,u_mat_dag)
        return dm_ret

    def pp_state(self):
        pp_vec = np.array([0.5,0.5,0.5,0.5])
        return pp_vec
    
    def bell_pp_state(self):
        u_mat = np.diag([1, np.exp(1j*self.phase_1),
                         np.exp(1j*self.phase_1),
                         np.exp(1j*self.phase_2)])
        pp_col_vec = np.reshape(np.array([0.5,0.5,0.5,0.5]),(-1,1))
        return np.reshape(np.matmul(u_mat,pp_col_vec),(-1))







class CZ():
    """
    This class is used for handling calculations of CZ gate
    fidelities under various modelings of the atom (from more
    ideal settings to realistic ones). 
    -------------
    Current version supports the following: 
    -------------
    - Modelling of the atom: two ground states 5S_{1/2} 
    |0>=|F=1,m_F=0>, |1>=|F=2,m_F=0> for qubit encoding;
    two 6P_{3/2} states |m_J=3/2> and |m_J=1/2> to both of which
    |1> state can be excited via a sigma+ 420 laser(|m_J=3/2> is
    the good state and |m_J=1/2> is the garbage state.); two 
    Rydberg states at nS_{1/2} |m_J=1/2> and |m_J=-1/2>(garbage)
    to both of which |1> can be excited to via the two-photon 
    process. |m_J=-1/2> is detuned out of resonance by a static
    magetic field. 
    - Doppler effect and laser fluctuations are  
    not taken into account yet.
    - Vanilla CZ gate protocal which uses two pulses with 
    constant rabi and a phase jump in between. 
    - 8 states: 0, 1, hf_1,2,3, ryd, ryd_garb, grab
    - TODO: CZ gate protocal using a single pulse with smoothly
    modulated phases. (Lukin's, fidelity~99.5%)
    - CZ gate under a more idealistic setting where there is 
    no decay.
    - CZ gate under a more realistic setting where each 
    non-ground state will decay to its respective garbage 
    state and will not play a part in the evolution. 
    - TODO: A more realistic modeling of the decay process.
    - TODO: Take into account the hyperfine structure to extract
    the precise error channel for the CZ gate.
    - TODO: Write more TODOs. 
    ------------
    Methods: 
    ------------
    """
    def __init__(self, mid_state_decay, ryd_decay,
                       ryd_garbage, temp = 0) -> None:
        """
        Test.
        """
        self.atom = Rubidium87()
        self.ryd_level = 53
        self.rabi_420 = 2*np.pi*(237)*10**(6)
        self.rabi_1013 = 2*np.pi*(303)*10**(6)
        self.Delta = 2*np.pi*(7.8)*10**(9)  # E_mid - omega_420
        # rabi_eff is only heuristically used to estimate the
        # time/energy scale of our system and should not be used 
        # for precise calculations. We sometimes use it as a unit. 
        self.rabi_eff = self.rabi_420*self.rabi_1013/(2*self.Delta)
        self.time_scale = 2*np.pi/self.rabi_eff
        # More than one possible excitation pathways. 
        # Calculation of branching ratio. More accurately,
        # the ratio of the dipole moment between the garbage
        # transition and the good transition so that the rabi
        # frequency of the garbage transformation is simply the
        # ratio*rabi frequency for the good transformation. 
        self.d_mid_ratio = (self.atom.getDipoleMatrixElement(5,0,0.5,
                               -0.5,6,1,1.5,0.5,1)/
                            self.atom.getDipoleMatrixElement(5,0,0.5,
                                0.5,6,1,1.5,1.5,1))
        self.d_ryd_ratio = (self.atom.getDipoleMatrixElement(6,1,1.5,
                                0.5,self.ryd_level,0,0.5,-0.5,-1)/
                            self.atom.getDipoleMatrixElement(6,1,1.5,
                                1.5,self.ryd_level,0,0.5,0.5,-1))
        self.d_total_ratio = self.d_mid_ratio*self.d_ryd_ratio 
        self.rabi_420_garbage = self.rabi_420*self.d_mid_ratio
        self.rabi_1013_garbage = self.rabi_1013*self.d_ryd_ratio
        self.v_ryd = 2*np.pi*(450)*10**(6) # Ryd interaction 
        # ryd_zeeman_shift is the zeeman shift of the garbage
        # ryd state |m_J=-1/2> from the good ryd state 
        # |m_J=1/2>
        self.ryd_zeeman_shift = -2*np.pi*(24)*10**(6)
        # The following two detunings are for the good Ryd state
        # and the garbage Ryd state respectively. Notice that 
        # their exact values will be changed during the CZ gate
        # operation. We present them here just for book-keeping.
        self.delta = 0
        self.delta_m = self.ryd_zeeman_shift + self.delta
        self.room_temperature = 300 
        self.mid_state_decay = mid_state_decay
        self.ryd_decay = ryd_decay
        self.ryd_garbage = ryd_garbage
        # Compute the decay rate using ARC
        # self.mid_state_decay_rate = (1/(self.atom.getStateLifetime(6,
        #                                 1,1.5)))*self.mid_state_decay
        # self.mid_garb_decay_rate = (1/(self.atom.getStateLifetime(
        #                             6,1,0.5)))*self.mid_state_decay
        # self.ryd_state_decay_rate = (1/(self.atom.getStateLifetime(
        #     self.ryd_level, 0, 0.5, 300, self.ryd_level+30
        # )))*self.ryd_decay
        # 
        # Decay rate parameters in Lukin's paper:
        self.mid_state_decay_rate = (1/(110*10**(-9)))*self.mid_state_decay
        self.mid_garb_decay_rate = (1/(110*10**(-9)))*self.mid_state_decay
        self.ryd_state_decay_rate = (1/(88*10**(-6)))*self.ryd_decay
        self.ryd_garb_decay_rate = (1/(88*10**(-6)))*self.ryd_garbage

        # self.br_m_garb = 0.6142
        # self.br_m_1 = 0.2504
        # self.br_m_0 = 0.1354
        # self.br_r_garb = 0.894
        # self.br_r_1 = 0.053
        # self.br_r_0 = 0.053

        self.tq_ham_const = self._tq_ham_const()
        self.tq_ham_420 = self._tq_ham_420()
        self.tq_ham_420_conj = self._tq_ham_420_conj()

        self.t_rise = 20*10**(-9)   # Blackman pulse rise time 

        self.temp = temp

        pass


    def velocity_thermal_sample(self):
        std = np.sqrt(constants.k*self.temp/
                        self.atom.mass)
        return np.random.normal(0,std)
    
    def doppler_shift_Hz(self):
        k_420 = 2*np.pi/(420*10**(-9))
        k_1013 = 2*np.pi/(1013*10**(-9))
        shift_Hz = (k_420-k_1013)*self.velocity_thermal_sample()/(
                        2*np.pi)
        return shift_Hz
    
    def doppler_std_Hz(self):
        v_std = np.sqrt(constants.k*self.temp/self.atom.mass)
        k_420 = 2*np.pi/(420*10**(-9))
        k_1013 = 2*np.pi/(1013*10**(-9))
        shift_std_Hz = (k_420-k_1013)*v_std/(2*np.pi)
        return shift_std_Hz



    def Opt_phase_mod_bell(self):
        # if self.temp != 0:
        #     warnings.warn("Temperature is not zero! Unfit for optimization.")
        obj_fun = self._obj_fidel_bell_mod()
        # init_vals = [6.216e-01, 1.355e+00, -4.737e-01, 
        #                 1.218e+00, 1.707e+00]
        # init_vals = [6.375691e-01, 1.386582e+00, -6.569367e-01,
        #              1.232724e+00, 1.686851e+00, -1e-03]
        # init_vals = [6.5059e-01, 1.3990e+00, -1.2071e+00,
        #              1.3467e+00, 1.6081e+00, -4.4439e-03]
         
        # init_vals = [8.879389e-01, 1.39033e+00, -1.97138e+00,
        #              1.530272e+00, 1.71564e+00, -5.89933e-02]
        
        # optimize over all errs, at zero temp:
        init_vals = [8.90749e-01, 1.38770e+00, -1.97452e+00,
                     1.53386e+00, 1.71850e+00, -6.06313e-02]
        # Lukin:
        # init_vals = [0.0988*2*np.pi, 1.3629, 2.6082-np.pi,
        #              1.232724e+00, 1.686851e+00, 0.0187]
        print(init_vals)
        bounds = ((-np.inf,np.inf),(-10,10),(-np.inf, np.inf),
                  (0,5),(-np.inf, np.inf),(-np.inf, np.inf))
        
        # when delta = 0:
        # obj_fun = self._obj_fidel_bell_mod_no_delta()
        # init_vals = [6.5059e-01, 1.3990e+00, -1.2071e+00,
        #         1.3467e+00, 1.6081e+00]
        # print(init_vals)
        # bounds = ((-np.inf,np.inf),(-10,10),(-np.inf, np.inf),
        #           (0,5),(-np.inf, np.inf))
        set_arr = np.array([[self.mid_state_decay, self.ryd_decay,
                            self.ryd_garbage, self.temp]])
        with open('opt_hf_new.txt','a') as f:
                np.savetxt(f,set_arr)
    
        self.iter = 0

        def callback(x):
            self.iter = self.iter + 1
            obj_fun = self._obj_fidel_bell_mod()
            fun = 1 - obj_fun(x)
            x = x.tolist()
            x.append(fun)
            x = np.array(x)
            x = np.reshape(x,(1,-1))
            with open('opt_hf_new.txt','a') as f:
                np.savetxt(f,x)
            print('iter: ',self.iter, ' fidel: ',fun)
            # print('       t_gate:', x[0][3])
            pass 

        opt_result = optimize.minimize(obj_fun, init_vals, 
                            method='Nelder-Mead',
                            options={'disp':True,
                                     'fatol':1e-9},
                            callback=callback,
                            bounds=bounds)
        return [opt_result.x, 1-opt_result.fun]


    def Fidelity_bell_mod(self, phase_amp, omega, phase_init, delta,
                            t_gate, phase_1, phase_2, phase_0=0):
        """
        Test
        """
        if t_gate<2*self.t_rise:
            raise ValueError("t_gate is too small compared to t_rise.")
        pauli_two = PauliTwo(phase_0, phase_1, phase_2)
        state_init = self._state_embed(7,pauli_two.pp_state())
        cz_result = self.CZ_phase_modulation(phase_amp,
                        omega, phase_init, delta, t_gate, state_init)
        ideal_result = self._state_embed(7,
                            pauli_two.bell_pp_state())
        fidel = np.square(np.abs(np.dot(np.conjugate(ideal_result),
                            cz_result)))
        return fidel
    
    def Fidelity_bell_mod_thermal(self, phase_amp, omega, phase_init, delta,
                            t_gate, phase_1, phase_2, phase_0=0):
        """
        Test
        """
        n_shots = 500
        fidel_list = []
        if t_gate<2*self.t_rise:
            raise ValueError("t_gate is too small compared to t_rise.")
        pauli_two = PauliTwo(phase_0, phase_1, phase_2)
        state_init = self._state_embed(7,pauli_two.pp_state())
        ideal_result = self._state_embed(7,
                            pauli_two.bell_pp_state())
        for i in range(n_shots):
            cz_result = self.CZ_phase_modulation_thermal(phase_amp,
                omega, phase_init, delta, t_gate, state_init)
            fidel_temp = np.square(np.abs(np.dot(np.conjugate(ideal_result),
                            cz_result)))
            print("Progress: ", str(i+1)+'/'+str(n_shots), 
                  "     Fidel: ", fidel_temp)
            fidel_list.append(fidel_temp)
        return fidel_list



    def _obj_fidel_bell_mod_no_delta(self):
        def fun(x):
            phase_amp = x[0]
            omega = x[1]*self.rabi_eff
            phase_init = x[2]
            t_gate = x[3]*self.time_scale
            phase_1 = x[4]
            delta = 0
            ret = 1 - self.Fidelity_bell_mod(phase_amp, omega, 
                        phase_init, delta,
                        t_gate, phase_1, 2*phase_1+np.pi)
            return ret
        return fun 


    def _obj_fidel_bell_mod(self):
        def fun(x):
            phase_amp = x[0]
            omega = x[1]*self.rabi_eff
            phase_init = x[2]
            t_gate = x[3]*self.time_scale
            phase_1 = x[4]
            delta = x[5]*self.rabi_eff
            ret = 1 - self.Fidelity_bell_mod(phase_amp, omega, 
                        phase_init, delta,
                        t_gate, phase_1, 2*phase_1+np.pi)
            return ret
        return fun 
    
    def _tq_ham_const(self):
        ham_tq_mat = np.zeros((49,49),dtype=np.complex128)
        ham_sq_mat = np.zeros((7,7),dtype=np.complex128)
        delta = 0
        ham_sq_mat[2][2] = (self.Delta - 2*np.pi*51*10**(6) -
                            1j*self.mid_state_decay_rate/2)
        ham_sq_mat[3][3] = (self.Delta - 1j*
                            self.mid_state_decay_rate/2) 
        ham_sq_mat[4][4] = (self.Delta + 2*np.pi*87*10**(6) -
                            1j*self.mid_state_decay_rate/2)
        ham_sq_mat[5][5] = (delta - 1j*self.ryd_state_decay_rate/2) 
        ham_sq_mat[6][6] = (delta + self.ryd_zeeman_shift - 
                            1j*self.ryd_garb_decay_rate/2)
        # ham_sq_mat[2][1] = self.rabi_420*phase_420/2
        # ham_sq_mat[1][2] = np.conjugate(ham_sq_mat[2][1])
        # ham_sq_mat[3][1] = self.rabi_420_garbage*phase_420/2
        # ham_sq_mat[1][3] = np.conjugate(ham_sq_mat[3][1])
        ham_sq_mat[5][2] = (self.rabi_1013/2)*CG(3/2,3/2,3/2,-1/2,
                                                 1,1)
        ham_sq_mat[2][5] = np.conjugate(ham_sq_mat[5][2])
        ham_sq_mat[5][3] = (self.rabi_1013/2)*CG(3/2,3/2,3/2,-1/2,
                                                 2,1)
        ham_sq_mat[3][5] = np.conjugate(ham_sq_mat[5][3])
        ham_sq_mat[5][4] = (self.rabi_1013/2)*CG(3/2,3/2,3/2,-1/2,
                                                 3,1)
        ham_sq_mat[4][5] = np.conjugate(ham_sq_mat[5][4])
    
        ham_sq_mat[6][2] = (self.rabi_1013_garbage*
                            self.ryd_garbage/2)*CG(
                                3/2,1/2,3/2,1/2,1,1)
        ham_sq_mat[2][6] = np.conjugate(ham_sq_mat[6][2])
        ham_sq_mat[6][3] = (self.rabi_1013_garbage*
                            self.ryd_garbage/2)*CG(
                                3/2,1/2,3/2,1/2,2,1)
        ham_sq_mat[3][6] = np.conjugate(ham_sq_mat[6][3])
        ham_sq_mat[6][4] = (self.rabi_1013_garbage*
                            self.ryd_garbage/2)*CG(
                                3/2,1/2,3/2,1/2,3,1)
        ham_sq_mat[4][6] = np.conjugate(ham_sq_mat[6][4])

        ham_tq_mat = ham_tq_mat + np.kron(np.eye(7),ham_sq_mat)
        ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat,np.eye(7))
        ham_vdw_mat = np.zeros((7,7))
        ham_vdw_mat[5][5] = 1
        ham_tq_mat = ham_tq_mat + self.v_ryd*np.kron(ham_vdw_mat,
                                                ham_vdw_mat)
        return ham_tq_mat

    def _tq_ham_const_thermal(self):
        ham_tq_mat = np.zeros((49,49),dtype=np.complex128)
        ham_sq_mat = np.zeros((7,7),dtype=np.complex128)
        delta = 0
        doppler_shift = 2*np.pi*self.doppler_shift_Hz()
        ham_sq_mat[2][2] = (self.Delta - 2*np.pi*51*10**(6) -
                            1j*self.mid_state_decay_rate/2)
        ham_sq_mat[3][3] = (self.Delta - 1j*
                            self.mid_state_decay_rate/2) 
        ham_sq_mat[4][4] = (self.Delta + 2*np.pi*87*10**(6) -
                            1j*self.mid_state_decay_rate/2)
        ham_sq_mat[5][5] = (delta + doppler_shift - 
                            1j*self.ryd_state_decay_rate/2) 
        ham_sq_mat[6][6] = (delta + doppler_shift + 
                            self.ryd_zeeman_shift - 
                            1j*self.ryd_garb_decay_rate/2)
        # ham_sq_mat[2][1] = self.rabi_420*phase_420/2
        # ham_sq_mat[1][2] = np.conjugate(ham_sq_mat[2][1])
        # ham_sq_mat[3][1] = self.rabi_420_garbage*phase_420/2
        # ham_sq_mat[1][3] = np.conjugate(ham_sq_mat[3][1])
        ham_sq_mat[5][2] = (self.rabi_1013/2)*CG(3/2,3/2,3/2,-1/2,
                                                 1,1)
        ham_sq_mat[2][5] = np.conjugate(ham_sq_mat[5][2])
        ham_sq_mat[5][3] = (self.rabi_1013/2)*CG(3/2,3/2,3/2,-1/2,
                                                 2,1)
        ham_sq_mat[3][5] = np.conjugate(ham_sq_mat[5][3])
        ham_sq_mat[5][4] = (self.rabi_1013/2)*CG(3/2,3/2,3/2,-1/2,
                                                 3,1)
        ham_sq_mat[4][5] = np.conjugate(ham_sq_mat[5][4])
    
        ham_sq_mat[6][2] = (self.rabi_1013_garbage*
                            self.ryd_garbage/2)*CG(
                                3/2,1/2,3/2,1/2,1,1)
        ham_sq_mat[2][6] = np.conjugate(ham_sq_mat[6][2])
        ham_sq_mat[6][3] = (self.rabi_1013_garbage*
                            self.ryd_garbage/2)*CG(
                                3/2,1/2,3/2,1/2,2,1)
        ham_sq_mat[3][6] = np.conjugate(ham_sq_mat[6][3])
        ham_sq_mat[6][4] = (self.rabi_1013_garbage*
                            self.ryd_garbage/2)*CG(
                                3/2,1/2,3/2,1/2,3,1)
        ham_sq_mat[4][6] = np.conjugate(ham_sq_mat[6][4])

        ham_tq_mat = ham_tq_mat + np.kron(np.eye(7),ham_sq_mat)
        ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat,np.eye(7))
        ham_vdw_mat = np.zeros((7,7))
        ham_vdw_mat[5][5] = 1
        ham_tq_mat = ham_tq_mat + self.v_ryd*np.kron(ham_vdw_mat,
                                                ham_vdw_mat)
        return ham_tq_mat

    def _tq_ham_420(self):
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

    def _tq_ham_420_conj(self):
        ham_tq_mat = np.zeros((49,49),dtype=np.complex128)
        ham_sq_mat = np.zeros((7,7),dtype=np.complex128)
        ham_sq_mat[1][2] = np.conjugate((self.rabi_420*
                                CG(3/2,3/2,3/2,-1/2,1,1) + 
                            self.rabi_420_garbage*
                                CG(3/2,1/2,3/2,1/2,1,1)
                            )/2)
        ham_sq_mat[1][3] = np.conjugate((self.rabi_420*
                                CG(3/2,3/2,3/2,-1/2,2,1) + 
                            self.rabi_420_garbage*
                                CG(3/2,1/2,3/2,1/2,2,1)
                            )/2)
        ham_sq_mat[1][4] = np.conjugate((self.rabi_420*
                                CG(3/2,3/2,3/2,-1/2,3,1) + 
                            self.rabi_420_garbage*
                                CG(3/2,1/2,3/2,1/2,3,1)
                            )/2)
        ham_tq_mat = ham_tq_mat + np.kron(np.eye(7),ham_sq_mat)
        ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat,np.eye(7))
        return ham_tq_mat
    
    def _occ_operator(self, index):
        oper_tq = np.zeros((49,49),dtype=np.complex128)
        oper_sq = np.zeros((7,7),dtype=np.complex128)
        oper_sq[index][index] = 1
        oper_tq = oper_tq + np.kron(np.eye(7),oper_sq)
        oper_tq = oper_tq + np.kron(oper_sq, np.eye(7))
        return oper_tq


    
    def CZ_phase_modulation(self, phase_amp, omega,
                            phase_init, delta, t_gate, state_mat):
        """
        Test
        """
        def ham_td(t, phase_init, omega, phase_amp, delta, t_gate):
            phase_420 = np.exp(-1j*(phase_amp*
                            np.cos(omega*t+phase_init)+delta*t))
            phase_420_conj = np.conjugate(phase_420)
            amplitude = blackman_pulse_sqrt(t, self.t_rise, t_gate)
            # amplitude = 1
            ham_tq_mat = np.zeros((49,49),dtype=np.complex128)
            ham_tq_mat = ham_tq_mat + (self.tq_ham_const + 
                        amplitude*phase_420*self.tq_ham_420 + 
                        amplitude*phase_420_conj*self.tq_ham_420_conj)
            return ham_tq_mat
        
        def fun(t, y, phase_init, omega, phase_amp, delta, t_gate):
            diff = -1j*ham_td(t, phase_init, omega, phase_amp, 
                                delta, t_gate)
            y_arr = np.reshape(np.array(y),(-1,1))
            return np.reshape(np.matmul(diff,y_arr),(-1))
        
        # print(state_mat)
        args = (phase_init, omega, phase_amp, delta, t_gate)
        t_span = [0, t_gate]
        result = integrate.solve_ivp(fun, t_span, state_mat,
                        args=args,
                        method='DOP853',
                        # method = 'RK45',
                        # method='BDF',
                        rtol=1e-8,
                        atol=1e-12)
        ret_state_vec = np.array(result.y)[:,-1]
        # print(np.dot(np.conjugate(ret_state_vec),ret_state_vec))
        return np.array(ret_state_vec)
    
    def CZ_phase_modulation_thermal(self, phase_amp, omega, phase_init,
                                    delta, t_gate, state_mat):
        """
        Test
        """
        self.tq_ham_const_temp = self._tq_ham_const_thermal()
        def ham_td(t, phase_init, omega, phase_amp, delta, t_gate):
            phase_420 = np.exp(-1j*(phase_amp*
                            np.cos(omega*t+phase_init)+delta*t))
            phase_420_conj = np.conjugate(phase_420)
            amplitude = blackman_pulse_sqrt(t, self.t_rise, t_gate)
            # amplitude = 1
            ham_tq_mat = np.zeros((49,49),dtype=np.complex128)
            ham_tq_mat = ham_tq_mat + (self.tq_ham_const_temp + 
                        amplitude*phase_420*self.tq_ham_420 + 
                        amplitude*phase_420_conj*self.tq_ham_420_conj)
            return ham_tq_mat
        
        def fun(t, y, phase_init, omega, phase_amp, delta, t_gate):
            diff = -1j*ham_td(t, phase_init, omega, phase_amp, 
                                delta, t_gate)
            y_arr = np.reshape(np.array(y),(-1,1))
            return np.reshape(np.matmul(diff,y_arr),(-1))
        
        # print(state_mat)
        args = (phase_init, omega, phase_amp, delta, t_gate)
        t_span = [0, t_gate]
        result = integrate.solve_ivp(fun, t_span, state_mat,
                        args=args,
                        method='DOP853',
                        # method = 'RK45',
                        # method='BDF',
                        rtol=1e-8,
                        atol=1e-12)
        ret_state_vec = np.array(result.y)[:,-1]
        # print(np.dot(np.conjugate(ret_state_vec),ret_state_vec))
        return np.array(ret_state_vec)

    # def Diagnosis(self, phase_amp, omega, phase_init,
    #                 delta, t_gate, state_mat):
    #     tq_ham_const_doppler = self._tq_ham_const_thermal()
    #     state_arr = self.Diagnosis_run(tq_ham_const_doppler, 
    #                                     phase_amp, omega, phase_init,
    #                                     delta, t_gate, state_mat)
    #     state_occ_arr = np.square(np.abs(state_arr)) 
    #     pass

    def Diagnosis_run(self, tq_ham_const_doppler, phase_amp,
                      omega, phase_init, delta, t_gate, state_mat):
        self.tq_ham_const_temp = tq_ham_const_doppler
        def ham_td(t, phase_init, omega, phase_amp, delta, t_gate):
            phase_420 = np.exp(-1j*(phase_amp*
                            np.cos(omega*t+phase_init)+delta*t))
            phase_420_conj = np.conjugate(phase_420)
            amplitude = blackman_pulse_sqrt(t, self.t_rise, t_gate)
            # amplitude = 1
            ham_tq_mat = np.zeros((49,49),dtype=np.complex128)
            ham_tq_mat = ham_tq_mat + (self.tq_ham_const_temp + 
                        amplitude*phase_420*self.tq_ham_420 + 
                        amplitude*phase_420_conj*self.tq_ham_420_conj)
            return ham_tq_mat
        
        def fun(t, y, phase_init, omega, phase_amp, delta, t_gate):
            diff = -1j*ham_td(t, phase_init, omega, phase_amp, 
                                delta, t_gate)
            y_arr = np.reshape(np.array(y),(-1,1))
            return np.reshape(np.matmul(diff,y_arr),(-1))
        
        # print(state_mat)
        # print("right answer:", 0.9976861840575)
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
        # ret_state_arr = np.array(result.y)
        # ret_state_occ_arr = np.square(np.abs(ret_state_arr)) 
        ret_state_occ_arr = np.array(result.y)
        # print(np.dot(np.conjugate(ret_state_occ_arr[:,-100]),ret_state_occ_arr[:,-100]))
        mid_state_occ_oper = (self._occ_operator(2) + self._occ_operator(3) + 
                              self._occ_operator(4))
        ryd_state_occ_oper = self._occ_operator(5)
        ryd_garb_state_occ_oper = self._occ_operator(6)
        mid_state_list = []
        ryd_state_list = []
        ryd_garb_list = []
        for col in range(len(ret_state_occ_arr[0,:])):
            state_temp = ret_state_occ_arr[:,col]
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
            # print(mid_state_occ_temp)
            # print(ryd_state_occ_temp)
            # print(ryd_garb_occ_temp)
            mid_state_list.append(np.abs(mid_state_occ_temp))
            ryd_state_list.append(np.abs(ryd_state_occ_temp))
            ryd_garb_list.append(np.abs(ryd_garb_occ_temp))
        mid_state_arr = np.array(mid_state_list)
        ryd_state_arr = np.array(ryd_state_list)
        ryd_garb_state_arr = np.array(ryd_garb_list)
        # mid_state_arr = ret_state_occ_arr[2:5,:]
        # ryd_state_arr = ret_state_occ_arr[5,:]
        # ryd_garb_state_arr = ret_state_occ_arr[6,:]
        # mid_decay_list = []
        # for occ_list in mid_state_arr:
        #     mid_decay_temp = self._decay_integrate(t_eval, occ_list, 
        #                                            self.mid_state_decay_rate)
        #     mid_decay_list.append(mid_decay_temp[0])
        mid_decay_temp = self._decay_integrate(t_eval, mid_state_arr, 
                                                   self.mid_state_decay_rate)
        ryd_decay_temp = self._decay_integrate(t_eval, ryd_state_arr,
                                               self.ryd_state_decay_rate)
        ryd_garb_decay_temp = self._decay_integrate(t_eval, ryd_garb_state_arr,
                                                self.ryd_garb_decay_rate)
        return [mid_state_arr, ryd_state_arr, ryd_garb_state_arr,
                mid_decay_temp[0], ryd_decay_temp[0],
                ryd_garb_decay_temp[0], t_eval]
    
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
    






    def _sq_mat_gen(self, num_states, row_i, col_j) ->np.ndarray:
        mat_temp = np.zeros((num_states, num_states),
                            dtype=np.complex128)
        mat_temp[row_i][col_j] = 1
        return mat_temp
    
    def _matrix_embed(self, num_states, 
                      matrix : np.ndarray) -> np.ndarray:
        """
        Used to embed a small 2-qubit system to a larger 
        system with more levels taken into account. 
        This method is quite dangerous because we tacitly 
        assumed |0> and |1> state in the small system is also
        the |0> and |1> state in the larger system which is 
        composed of states |0>,|1>,|2>,..., and |n>.
        """
        mat_temp = np.zeros((num_states**2, num_states**2),
                            dtype=np.complex128)
        for i in range(0,2):
            for j in range(0,2):
                for k in range(0,2):
                    for m in range(0,2):
                        row = 2*i+j
                        col = 2*k+m
                        mat_1 = self._sq_mat_gen(num_states,i,k)
                        mat_2 = self._sq_mat_gen(num_states,j,m)
                        mat_temp = mat_temp + np.kron(mat_1,
                                        mat_2)*matrix[row][col]
        return mat_temp

    def _sq_state_gen(self, num_states, row) ->np.ndarray:
        state_temp = np.zeros(num_states, dtype=np.complex128)
        state_temp[row] = 1
        return state_temp
    
    def _state_embed(self, num_states, state : np.array) -> np.array:
        """
        Used to embed a small 2-qubit system to a larger 
        system with more levels taken into account. 
        This method is quite dangerous because we tacitly 
        assumed |0> and |1> state in the small system is also
        the |0> and |1> state in the larger system which is 
        composed of states |0>,|1>,|2>,..., and |n>.
        """
        state_temp = np.zeros(num_states**2, dtype=np.complex128)
        for i in range(0,2):
            for j in range(0,2):
                index = 2*i + j
                sq_state_1 = self._sq_state_gen(num_states,i)
                sq_state_2 = self._sq_state_gen(num_states,j)
                state_temp = state_temp + np.kron(sq_state_1,
                                    sq_state_2)*state[index]
        return state_temp

    
    def _two_qubit_state_haar_dm(self):
        """
        Output the density matrix of a Haar random two-qubit
        pure state. 
        """
        haar_mat = rand_unitary_haar(4).data.toarray()
        state_init = self._sq_mat_gen(4,0,0)
        state_inter = np.matmul(haar_mat, state_init)
        state_end = np.matmul(state_inter, 
                        np.conjugate(np.transpose(haar_mat)))
        return state_end




# 0.6172 1.3374 -0.3475 1.2043 1.6869 fidel: 99.461  all err

# 6.5059e-01 1.3990e+00 -1.2071e+00 1.3467e+00 1.6081e+00 -4.4439e-03
            
temp = 30*10**(-6)
cz_test = CZ(mid_state_decay=1, ryd_decay=1, ryd_garbage=1, temp=temp)

# param_opt, fidel_bell = cz_test.Opt_phase_mod_bell()

# print(fidel_bell)


fidel_list = cz_test.Fidelity_bell_mod_thermal(8.958024327e-01,
                1.38171690e+00*cz_test.rabi_eff,
                -1.86536976e+00, -6.43545721e-02*cz_test.rabi_eff,
                1.5152934673e+00*cz_test.time_scale,
                1.710546596e+00,(2*1.710546596e+00)+np.pi)

print('avg fidel: ', np.mean(np.array(fidel_list)))



# Test all processes during gate protocol 
n_shots = 500
mid_state_occ_all = []
ryd_state_occ_all = []
ryd_garb_occ_all = []
mid_decay_all = []
ryd_decay_all = []
ryd_garb_all = []
for index in range(n_shots):
    tq_ham_const_therm = cz_test._tq_ham_const_thermal()
    phase_0 = 0
    phase_1 = 1.71054659e+00
    phase_2 = (2*1.71054659e+00)+np.pi
    pauli_two = PauliTwo(phase_0, phase_1, phase_2)
    state_init = cz_test._state_embed(7,pauli_two.pp_state())
    all_shit_together = cz_test.Diagnosis_run(tq_ham_const_therm, 
                            8.95802432e-01, 1.3817169e+00*cz_test.rabi_eff,
                            -1.8653697e+00, -6.43545721e-02*cz_test.rabi_eff,
                            1.515293467e+00*cz_test.time_scale,
                            state_init)

    t_eval = all_shit_together[-1]*10**(9)

    mid_state_occ_all.append(np.array(all_shit_together[0]))
    ryd_state_occ_all.append(np.array(all_shit_together[1]))
    ryd_garb_occ_all.append(np.array(all_shit_together[2]))
    mid_decay_all.append(np.array(all_shit_together[3]))
    ryd_decay_all.append(np.array(all_shit_together[4]))
    ryd_garb_all.append(np.array(all_shit_together[5]))
    print('process: ',str(index)+'/'+str(n_shots))

mid_state_occ_all = np.array(mid_state_occ_all)
ryd_state_occ_all = np.array(ryd_state_occ_all)
ryd_garb_occ_all = np.array(ryd_garb_occ_all)
mid_decay_all = np.array(mid_decay_all)
ryd_decay_all = np.array(ryd_decay_all)
ryd_garb_all = np.array(ryd_garb_all)

plt.figure()
plt.plot(t_eval, np.mean(mid_state_occ_all, axis=0), 
         color = color_dict['Teal'], 
         label='Mid-states')
plt.fill_between(t_eval, 
                 (np.mean(mid_state_occ_all, axis=0)-
                  np.std(mid_state_occ_all, axis=0)),
                 (np.mean(mid_state_occ_all, axis=0)+
                  np.std(mid_state_occ_all, axis=0)), 
                  color = color_dict['Teal'],
                  alpha=0.5)

plt.plot(t_eval, np.mean(ryd_garb_occ_all, axis=0),
         color = color_dict['Red'],
         label='Unwanted Rydberg excitation')
plt.fill_between(t_eval, 
                 (np.mean(ryd_garb_occ_all, axis=0)-
                  np.std(ryd_garb_occ_all, axis=0)),
                 (np.mean(ryd_garb_occ_all, axis=0)+
                  np.std(ryd_garb_occ_all, axis=0)), 
                  color = color_dict['Red'],
                  alpha=0.5)

plt.xlabel('time (ns)')
plt.ylabel('State occupation probability')
plt.legend()
plt.grid(True)
plt.savefig('cz_mid_occ_bsq_30_micro_K_500_shots.pdf')

plt.figure()
plt.plot(t_eval, np.mean(ryd_state_occ_all, axis=0), 
         color = color_dict['Blue'], 
         label='Rydberg')
plt.fill_between(t_eval, 
                 (np.mean(ryd_state_occ_all, axis=0)-
                  np.std(ryd_state_occ_all, axis=0)),
                 (np.mean(ryd_state_occ_all, axis=0)+
                  np.std(ryd_state_occ_all, axis=0)), 
                  color = color_dict['Blue'],
                  alpha=0.5)
plt.xlabel('time (ns)')
plt.ylabel('State occupation probability')
# plt.legend()
plt.grid(True)
plt.savefig('cz_ryd_occ_bsq_30_micro_K_500_shots.pdf')



plt.figure()
plt.plot(t_eval, np.mean(ryd_decay_all, axis=0), 
         color = color_dict['Blue'], 
         label='Rydberg decay')
plt.fill_between(t_eval, 
                 (np.mean(ryd_decay_all, axis=0)-
                  np.std(ryd_decay_all, axis=0)),
                 (np.mean(ryd_decay_all, axis=0)+
                  np.std(ryd_decay_all, axis=0)), 
                  color = color_dict['Blue'],
                  alpha=0.5)
plt.plot(t_eval, np.mean(mid_decay_all, axis=0), 
         color = color_dict['Teal'], 
         label='Mid-state decay')
plt.fill_between(t_eval, 
                 (np.mean(mid_decay_all, axis=0)-
                  np.std(mid_decay_all, axis=0)),
                 (np.mean(mid_decay_all, axis=0)+
                  np.std(mid_decay_all, axis=0)), 
                  color = color_dict['Teal'],
                  alpha=0.5)
plt.plot(t_eval, np.mean(ryd_garb_all, axis=0),
         color = color_dict['Red'],
         label='Unwanted Rydberg decay')
plt.fill_between(t_eval, 
                 (np.mean(ryd_garb_all, axis=0)-
                  np.std(ryd_garb_all, axis=0)),
                 (np.mean(ryd_garb_all, axis=0)+
                  np.std(ryd_garb_all, axis=0)), 
                  color = color_dict['Red'],
                  alpha=0.5)


plt.xlabel('time (ns)')
plt.ylabel('Error probability')
plt.legend()
plt.grid(True)
plt.savefig('cz_error_bsq_30_micro_K_500_shots.pdf')









# tq_ham_const_therm = cz_test._tq_ham_const_thermal()
# phase_0 = 0
# phase_1 = 1.7193658e+00
# phase_2 = (2*1.7193658e+00)+np.pi
# pauli_two = PauliTwo(phase_0, phase_1, phase_2)
# state_init = cz_test._state_embed(7,pauli_two.pp_state())
# all_shit_together = cz_test.Diagnosis_run(tq_ham_const_therm, 
#                         8.91542996e-01, 1.387220109e+00*cz_test.rabi_eff,
#                         -1.975647263e+00, -6.0834125323e-02*cz_test.rabi_eff,
#                         1.534627726e+00*cz_test.time_scale,
#                         state_init)

# t_eval = all_shit_together[-1]*10**(9)

# mid_state_occ = all_shit_together[0]
# ryd_state_occ = all_shit_together[1]
# ryd_garb_occ = all_shit_together[2]
# mid_decay = all_shit_together[3]
# ryd_decay = all_shit_together[4]
# ryd_garb = all_shit_together[5]

# plt.figure()
# plt.plot(t_eval, mid_state_occ, color = color_dict['Teal'])
# # # plt.plot(t_eval, ryd_state_occ)
# plt.figure()
# plt.plot(t_eval, ryd_garb_occ, color = color_dict['Red'])

# plt.figure()
# plt.plot(t_eval, ryd_state_occ)



# plt.figure()
# plt.plot(t_eval, mid_decay)
# plt.plot(t_eval, ryd_decay)
# plt.figure()
# plt.plot(t_eval, ryd_garb)



# fidel_list = cz_test.Fidelity_bell_mod_thermal(8.879389e-01,
#                 1.39033e+00*cz_test.rabi_eff,
#                 -1.97138e+00, -5.89933e-02*cz_test.rabi_eff,
#                 1.530272e+00*cz_test.time_scale,
#                 1.71564e+00,(2*1.71564e+00)+np.pi)

# fidel_arr = np.array(fidel_list)
# fidel_mean = np.mean(fidel_arr)
# fidel_std = np.std(fidel_arr)
# print("Avg: ", fidel_mean)
# print("Std: ", fidel_std)

