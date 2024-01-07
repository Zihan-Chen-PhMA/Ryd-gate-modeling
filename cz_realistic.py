import numpy as np 
from arc import * 
import matplotlib.pyplot as plt 
from scipy import signal
from scipy import optimize
from qutip import *
from cmath import * 
import copy as copy



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
    CZ_real(delta_ratio, state_mat) -> np.ndarray:
        Assumption: decay is taken into account.
        Detuning of the two-photon process is given by
        delta_ratio*self.rabi_eff. (Here the detuning is defined 
        as the 1013 laser freq + 420 laser freq - (energy 
        difference between ryd and |1>))
        Once the detuning is given, pulse time and phase jump
        will be obtained and optimized under the no decay 
        situation.  
        Input state(density matrix)
        is given by state_mat which is a numpy 2D-array. 
        The output is the resulting state(density matrix).  
        *Notice that if the delta_ratio is not the correct one
        for the CZ gate then the gate implemented may not be 
        close to the ideal CZ gate. 

    CZ_ptcol_ideal_phase(delta_ratio):
        Assumption: decay is ignored.
        Given a detuning for the two-photon excitation, 
        find the optimzied pulse time and phase jump parameters
        so that at the end |11> and |01> can both return to 
        themselves respectively. The output is the corresponding
        single-qubit phase change on |01>, the two-qubit 
        phase change on |11> and also the respective
        probabilities of |01> and |11> returning back to 
        themselves. 
    """
    def __init__(self) -> None:
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
        self.temperature = 300 


        pass

    def CZ_real(self, delta_ratio, state_mat) -> np.ndarray:
        """
        Test
        """
        t_gate, phase_opt, fid = self._gate_params(delta_ratio)
        delta = -delta_ratio*self.rabi_eff
        exp_phase = np.exp(1j*phase_opt)

        mid_state_decay = np.sqrt(1/(self.atom.getStateLifetime(6,
                                        1,1.5)))
        mid_garb_decay = np.sqrt(1/(self.atom.getStateLifetime(
                                    6,1,0.5)))
        ryd_state_decay = np.sqrt(1/(self.atom.getStateLifetime(
            self.ryd_level, 0, 0.5, 300, self.ryd_level+30
        )))

        ham_tq_mat = np.zeros((100,100),dtype=np.complex128)
        ham_sq_mat = np.zeros((10,10),dtype=np.complex128)
        ham_sq_mat[2][2] = self.Delta
        ham_sq_mat[3][3] = self.Delta
        ham_sq_mat[4][4] = delta
        ham_sq_mat[5][5] = delta + self.ryd_zeeman_shift
        ham_sq_mat[2][1] = self.rabi_420/2
        ham_sq_mat[1][2] = np.conjugate(ham_sq_mat[2][1])
        ham_sq_mat[3][1] = self.rabi_420_garbage/2
        ham_sq_mat[1][3] = np.conjugate(ham_sq_mat[3][1])
        ham_sq_mat[4][2] = self.rabi_1013/2
        ham_sq_mat[2][4] = np.conjugate(ham_sq_mat[4][2])
        ham_sq_mat[5][3] = self.rabi_1013_garbage/2
        ham_sq_mat[3][5] = np.conjugate(ham_sq_mat[5][3])
        ham_tq_mat = ham_tq_mat + np.kron(np.eye(10),ham_sq_mat)
        ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat,np.eye(10))
        ham_vdw_mat = np.zeros((10,10))
        ham_vdw_mat[4][4] = 1
        ham_tq_mat = ham_tq_mat + self.v_ryd*np.kron(ham_vdw_mat,
                                                ham_vdw_mat)
        ham_tq_oper = Qobj(ham_tq_mat)

        mid_state_decay_mat = self._sq_mat_gen(
            10,6,2)*mid_state_decay
        mid_garb_decay_mat = self._sq_mat_gen(
            10,7,3)*mid_garb_decay
        ryd_state_decay_mat = self._sq_mat_gen(
            10,8,4)*ryd_state_decay
        ryd_garb_decay_mat = self._sq_mat_gen(
            10,9,5)*ryd_state_decay

        c_ops = []  # Used for decays. 
 
        for i in range(0,10):
            mid_decay_temp = np.kron(self._sq_mat_gen(10,i,i),
                                     mid_state_decay_mat)
            mid_decay_oper = Qobj(mid_decay_temp)
            c_ops.append(mid_decay_oper)
            mid_garb_temp = np.kron(self._sq_mat_gen(10,i,i),
                                    mid_garb_decay_mat)
            mid_garb_oper = Qobj(mid_garb_temp)
            c_ops.append(mid_garb_oper)
            ryd_temp = np.kron(self._sq_mat_gen(10,i,i),
                               ryd_state_decay_mat)
            ryd_oper = Qobj(ryd_temp)
            c_ops.append(ryd_oper)
            ryd_garb_temp = np.kron(self._sq_mat_gen(10,i,i),
                                    ryd_garb_decay_mat)
            ryd_garb_oper = Qobj(ryd_garb_temp)
            c_ops.append(ryd_garb_oper)

            mid_decay_temp = np.kron(mid_state_decay_mat,
                                     self._sq_mat_gen(10,i,i))
            mid_decay_oper = Qobj(mid_decay_temp)
            c_ops.append(mid_decay_oper)
            mid_garb_temp = np.kron(mid_garb_decay_mat,
                                    self._sq_mat_gen(10,i,i)
                                    )
            mid_garb_oper = Qobj(mid_garb_temp)
            c_ops.append(mid_garb_oper)
            ryd_temp = np.kron(ryd_state_decay_mat,
                               self._sq_mat_gen(10,i,i)
                               )
            ryd_oper = Qobj(ryd_temp)
            c_ops.append(ryd_oper)
            ryd_garb_temp = np.kron(ryd_garb_decay_mat,
                                    self._sq_mat_gen(10,i,i)
                                    )
            ryd_garb_oper = Qobj(ryd_garb_temp)
            c_ops.append(ryd_garb_oper)

        state_init_oper = Qobj(state_mat)

        t_list = np.linspace(0,t_gate,5000)

        result_mid = mesolve(ham_tq_oper, state_init_oper,
                             t_list,c_ops=c_ops)
        mid_state_oper = result_mid.states[-1]

        ham_tq_jump_mat = np.zeros((100,100),dtype=np.complex128)
        ham_sq_mat = np.zeros((10,10),dtype=np.complex128)
        ham_sq_mat[2][2] = self.Delta
        ham_sq_mat[3][3] = self.Delta
        ham_sq_mat[4][4] = delta
        ham_sq_mat[5][5] = delta + self.ryd_zeeman_shift
        ham_sq_mat[2][1] = self.rabi_420*exp_phase/2
        ham_sq_mat[1][2] = np.conjugate(ham_sq_mat[2][1])
        ham_sq_mat[3][1] = self.rabi_420_garbage*exp_phase/2
        ham_sq_mat[1][3] = np.conjugate(ham_sq_mat[3][1])
        ham_sq_mat[4][2] = self.rabi_1013/2
        ham_sq_mat[2][4] = np.conjugate(ham_sq_mat[4][2])
        ham_sq_mat[5][3] = self.rabi_1013_garbage/2
        ham_sq_mat[3][5] = np.conjugate(ham_sq_mat[5][3])
        ham_tq_jump_mat = ham_tq_jump_mat + np.kron(np.eye(10),
                                                    ham_sq_mat)
        ham_tq_jump_mat = ham_tq_jump_mat + np.kron(ham_sq_mat,
                                                    np.eye(10))
        ham_vdw_mat = np.zeros((10,10))
        ham_vdw_mat[4][4] = 1
        ham_tq_jump_mat = ham_tq_jump_mat + self.v_ryd*np.kron(
                                                ham_vdw_mat,
                                                ham_vdw_mat)
        ham_tq_jump_oper = Qobj(ham_tq_jump_mat)

        result_end = mesolve(ham_tq_jump_oper, mid_state_oper,
                             t_list)
        end_state = result_end.states[-1]
        return end_state.data.toarray()



    def CZ_ptcol_ideal_phase(self, delta_ratio) -> list[float]:
        t_gate, phase_opt, fid = self._gate_params(delta_ratio)
        delta = -delta_ratio*self.rabi_eff
        exp_phase = np.exp(1j*phase_opt)

        ham_tq_mat = np.zeros((36,36),dtype=np.complex128)
        ham_sq_mat = np.zeros((6,6),dtype=np.complex128)
        ham_sq_mat[2][2] = self.Delta
        ham_sq_mat[3][3] = self.Delta
        ham_sq_mat[4][4] = delta
        ham_sq_mat[5][5] = delta + self.ryd_zeeman_shift
        ham_sq_mat[2][1] = self.rabi_420/2
        ham_sq_mat[1][2] = np.conjugate(ham_sq_mat[2][1])
        ham_sq_mat[3][1] = self.rabi_420_garbage/2
        ham_sq_mat[1][3] = np.conjugate(ham_sq_mat[3][1])
        ham_sq_mat[4][2] = self.rabi_1013/2
        ham_sq_mat[2][4] = np.conjugate(ham_sq_mat[4][2])
        ham_sq_mat[5][3] = self.rabi_1013_garbage/2
        ham_sq_mat[3][5] = np.conjugate(ham_sq_mat[5][3])
        ham_tq_mat = ham_tq_mat + np.kron(np.eye(6),ham_sq_mat)
        ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat,np.eye(6))
        ham_vdw_mat = np.zeros((6,6))
        ham_vdw_mat[4][4] = 1
        ham_tq_mat = ham_tq_mat + self.v_ryd*np.kron(ham_vdw_mat,
                                                ham_vdw_mat)
        sq_0_mat = np.zeros((6,6))
        sq_0_mat[0][0] = 1
        sq_1_mat = np.zeros((6,6))
        sq_1_mat[1][1] = 1
        sq_01_mat = np.zeros((6,6))
        sq_01_mat[0][1] = 1
        sq_10_mat = np.zeros((6,6))
        sq_10_mat[1][0] =1

        state_tq_init = np.zeros((36,36))
        state_tq_init = (0.5*np.kron(sq_0_mat,sq_0_mat) + 
                        0.5*np.kron(sq_0_mat,sq_1_mat) + 
                        0.5*np.kron(sq_0_mat,sq_10_mat) + 
                        0.5*np.kron(sq_0_mat,sq_01_mat))

        ham_tq_oper = Qobj(ham_tq_mat)
        state_tq_oper = Qobj(state_tq_init)

        t_list = np.linspace(0,t_gate,5000)
        result_mid_1 = mesolve(ham_tq_oper, state_tq_oper, 
                            t_list)
        state_mid_oper_1 = result_mid_1.states[-1]


        ham_tq_jump_mat = np.zeros((36,36),dtype=np.complex128)
        ham_sq_mat = np.zeros((6,6),dtype=np.complex128)
        ham_sq_mat[2][2] = self.Delta
        ham_sq_mat[3][3] = self.Delta
        ham_sq_mat[4][4] = delta
        ham_sq_mat[5][5] = delta + self.ryd_zeeman_shift
        ham_sq_mat[2][1] = self.rabi_420*exp_phase/2
        ham_sq_mat[1][2] = np.conjugate(ham_sq_mat[2][1])
        ham_sq_mat[3][1] = self.rabi_420_garbage*exp_phase/2
        ham_sq_mat[1][3] = np.conjugate(ham_sq_mat[3][1])
        ham_sq_mat[4][2] = self.rabi_1013/2
        ham_sq_mat[2][4] = np.conjugate(ham_sq_mat[4][2])
        ham_sq_mat[5][3] = self.rabi_1013_garbage/2
        ham_sq_mat[3][5] = np.conjugate(ham_sq_mat[5][3])
        ham_tq_jump_mat = ham_tq_jump_mat + np.kron(np.eye(6),ham_sq_mat)
        ham_tq_jump_mat = ham_tq_jump_mat + np.kron(ham_sq_mat,np.eye(6))
        ham_vdw_mat = np.zeros((6,6))
        ham_vdw_mat[4][4] = 1
        ham_tq_jump_mat = ham_tq_jump_mat + self.v_ryd*np.kron(
                                                ham_vdw_mat,
                                                ham_vdw_mat)
        ham_tq_jump_oper = Qobj(ham_tq_jump_mat)

        expect_op_1 = np.kron(sq_0_mat,sq_01_mat)
        # expect_op_1 = Qobj(expect_op_1)

        result_phase_1 = mesolve(ham_tq_jump_oper, state_mid_oper_1,
                                t_list)
        end_state = result_phase_1.states[-1].data.toarray()
        end_phase_1_exp = np.trace(np.matmul(expect_op_1,end_state))
        end_phase_1 = phase(complex(np.real(end_phase_1_exp),
                                    np.imag(end_phase_1_exp)))



        state_tq_init_2 = np.zeros((36,36))
        state_tq_init_2 = (0.5*np.kron(sq_0_mat,sq_0_mat) + 
                        0.5*np.kron(sq_1_mat,sq_1_mat) + 
                        0.5*np.kron(sq_01_mat,sq_01_mat) + 
                        0.5*np.kron(sq_10_mat,sq_10_mat))
        state_tq_oper_2 = Qobj(state_tq_init_2)
        result_mid_2 = mesolve(ham_tq_oper, state_tq_oper_2, 
                            t_list)
        state_mid_oper_2 = result_mid_2.states[-1]
        result_2 = mesolve(ham_tq_jump_oper, state_mid_oper_2,
                        t_list)
        end_state_2 = result_2.states[-1].data.toarray()
        expect_op_2 = np.kron(sq_01_mat,sq_01_mat)
        end_phase_2_exp = np.trace(np.matmul(expect_op_2,
                                            end_state_2))
        end_phase_2 = phase(complex(np.real(end_phase_2_exp),
                                    np.imag(end_phase_2_exp)))
        # print(end_state_2)

        return [end_phase_1, end_phase_2, fid]

    def _gate_params(self, delta_ratio) -> list[float]:
        """
        Given delta_ratio, obtain the correct pulse time s.t. 
        |11> returns to itself. Moreover, given
        an input state |11>, record its return probability
        as occ_11.

        Then optimize the phase jump parameter to make sure
        |01> returns to itself at the end of the gate protocal.
        Record its return probability as occ_01.

        It goes without saying that both occ_01 and occ_11
        should be very close to 1.

        Notice that given an arbitrary delta_ratio, we may not 
        be able to obtain a CZ gate. (But it should be some
        sort of two-qubit phase gate.) 
        """
        t_gate, occ_11 = self._gate_time(delta_ratio)
        phase_opt, occ_01 = self._gate_phase_opt(delta_ratio,
                                                 t_gate)
        return [t_gate, phase_opt, [occ_01, occ_11]]


    def _gate_time(self, delta_ratio, disp = 0) -> list[float]:
        delta = -delta_ratio*self.rabi_eff
        time = 2*self.time_scale
        ham_11_mat = np.zeros((25,25))
        ham_sq_mat = np.zeros((5,5))
        ham_sq_mat[1][1] = self.Delta
        ham_sq_mat[2][2] = self.Delta
        ham_sq_mat[3][3] = delta
        ham_sq_mat[4][4] = delta + self.ryd_zeeman_shift
        ham_sq_mat[1][0] = self.rabi_420/2
        ham_sq_mat[0][1] = np.conjugate(ham_sq_mat[1][0])
        ham_sq_mat[2][0] = self.rabi_420_garbage/2
        ham_sq_mat[0][2] = np.conjugate(ham_sq_mat[2][0])
        ham_sq_mat[3][1] = self.rabi_1013/2
        ham_sq_mat[1][3] = np.conjugate(ham_sq_mat[3][1])
        ham_sq_mat[4][2] = self.rabi_1013_garbage/2
        ham_sq_mat[2][4] = np.conjugate(ham_sq_mat[4][2])
        ham_11_mat = ham_11_mat + np.kron(np.eye(5),ham_sq_mat)
        ham_11_mat = ham_11_mat + np.kron(ham_sq_mat,np.eye(5))
        ham_vdw_mat = np.zeros((5,5))
        ham_vdw_mat[3][3] = 1
        ham_11_mat = ham_11_mat + self.v_ryd*np.kron(ham_vdw_mat,
                                                ham_vdw_mat)

        state_11_init = np.zeros((25,25))
        state_11_init[0][0] = 1 

        ham_11_oper = Qobj(ham_11_mat)
        state_11_oper = Qobj(state_11_init)
        time_list = np.linspace(0,time,300*200)

        result = mesolve(ham_11_oper, state_11_oper, 
                        time_list)
        state_list = result.states

        gs_occ_list = []
        for i in range(len(time_list)):
            state_temp = state_list[i]
            state_temp_mat = state_temp.data.toarray()
            gs_occ_list.append(np.real_if_close(state_temp_mat[0][0]))
        # peaks, _ = signal.find_peaks(gs_occ_list)
        # peaks = peaks.tolist()
        # peaks = peaks
        vert_span = max(gs_occ_list) - min(gs_occ_list)
        threshold_b = max(gs_occ_list) - 0.3*vert_span
        threshold_t = max(gs_occ_list) - 0.2*vert_span
        for i in range(len(time_list)):
            if gs_occ_list[i]<threshold_b:
                index_below_1 = i
                break
        for i in range(len(time_list)):
            if i>index_below_1:
                if gs_occ_list[i]>threshold_t:
                    index_start = i
                    break
        for i in range(len(time_list)):
            if i>index_start:
                if gs_occ_list[i]<threshold_b:
                    index_end = i
                    break

        max_to_1 = max(gs_occ_list[index_start:index_end])
        for i in range(len(time_list)):
            if i>index_start:
                if i<index_end:
                    if gs_occ_list[i] == max_to_1:
                        index_max = i
                        break

        if disp != 0:
            plt.figure(figsize=(16,5))
            # peaks_time = [t_list[i] for i in peaks]
            # peaks_val = [gs_occ_list[i] for i in peaks]
            plt.plot(time_list,gs_occ_list)
            plt.plot(time_list[index_max],gs_occ_list[index_max],'x')
            # plt.plot(peaks_time, peaks_val,'x')
        t_gate = time_list[index_max]
        occ_11 = gs_occ_list[index_max]

        return [t_gate, occ_11] 
        

    def _gate_phase_opt(self, delta_ratio, t_gate) -> float:
        delta = -delta_ratio*self.rabi_eff

        ham_sq_mat = np.zeros((5,5),dtype=np.complex128)
        ham_sq_mat[1][1] = self.Delta
        ham_sq_mat[2][2] = self.Delta
        ham_sq_mat[3][3] = delta
        ham_sq_mat[4][4] = delta + self.ryd_zeeman_shift
        ham_sq_mat[1][0] = self.rabi_420/2
        ham_sq_mat[0][1] = np.conjugate(ham_sq_mat[1][0])
        ham_sq_mat[2][0] = self.rabi_420_garbage/2
        ham_sq_mat[0][2] = np.conjugate(ham_sq_mat[2][0])
        ham_sq_mat[3][1] = self.rabi_1013/2
        ham_sq_mat[1][3] = np.conjugate(ham_sq_mat[3][1])
        ham_sq_mat[4][2] = self.rabi_1013_garbage/2
        ham_sq_mat[2][4] = np.conjugate(ham_sq_mat[4][2])

        state_1r_init = np.zeros((5,5))
        state_1r_init[0][0] = 1

        ham_1r_oper = Qobj(ham_sq_mat)
        state_1r_oper = Qobj(state_1r_init)

        t_list = np.linspace(0,t_gate,5000)
        result_1r = mesolve(ham_1r_oper, state_1r_oper, 
                            t_list)
        state_mid_oper = result_1r.states[-1]

        def phase_res(phase, state_mid_oper):
            exp_phase = np.exp(1j*phase)
            ham_sq_mat = np.zeros((5,5),dtype=np.complex128)
            ham_sq_mat[1][1] = self.Delta
            ham_sq_mat[2][2] = self.Delta
            ham_sq_mat[3][3] = delta
            ham_sq_mat[4][4] = delta + self.ryd_zeeman_shift
            ham_sq_mat[1][0] = self.rabi_420*exp_phase/2
            ham_sq_mat[0][1] = np.conjugate(ham_sq_mat[1][0])
            ham_sq_mat[2][0] = self.rabi_420_garbage*exp_phase/2
            ham_sq_mat[0][2] = np.conjugate(ham_sq_mat[2][0])
            ham_sq_mat[3][1] = self.rabi_1013/2
            ham_sq_mat[1][3] = np.conjugate(ham_sq_mat[3][1])
            ham_sq_mat[4][2] = self.rabi_1013_garbage/2
            ham_sq_mat[2][4] = np.conjugate(ham_sq_mat[4][2])

            ham_1r_oper = Qobj(ham_sq_mat)
            result_scan = mesolve(ham_1r_oper, state_mid_oper,
                                    t_list)
            result_fin_mat = result_scan.states[-1].data.toarray()
            return 1-np.real_if_close(result_fin_mat[0][0])


        rabi_init = self.rabi_eff + 0 
        theta_half = 0.5*np.sqrt(rabi_init**2+delta**2)*t_gate
        y_eff = np.sqrt(rabi_init**2+delta**2)
        exp_phase = (-1)*((y_eff**2*np.cos(theta_half)**2
                    +delta**2*np.sin(theta_half)**2)/
                    (y_eff*np.cos(theta_half)
                    -1j*delta*np.sin(theta_half))**2)
        phase_xi = phase(complex(np.real(exp_phase),
                                np.imag(exp_phase)))


        opt_result = optimize.minimize(phase_res, phase_xi, 
                                       args=(state_mid_oper),
                                    method='trust-constr',
                                    options={'verbose':0,
                                                'gtol':1e-9})

        phase_optimal = opt_result.x[0]
        return [phase_optimal, 1-opt_result.fun]


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
                        mat_1 = self._sq_mat_gen(10,i,k)
                        mat_2 = self._sq_mat_gen(10,j,m)
                        mat_temp = mat_temp + np.kron(mat_1,
                                        mat_2)*matrix[row][col]
        return mat_temp
    
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







            
cz_test = CZ()



# Scanning delta_ratio to approximately locate 
# the sweet spot for the CZ gate. 

# delta_ratio_list = np.linspace(0,3,150)
# phase_1_list = []
# phase_2_list = []
# fid_list = []
# for delta in delta_ratio_list:
#     phase_1, phase_2, fid = cz_test.CZ_ptcol_ideal_phase(delta)
#     phase_1_list.append(phase_1)
#     phase_2_list.append(phase_2)
#     fid_list.append(fid)
#     print('status:',round(100*delta/(3)))

# phase_1_arr = np.array(phase_1_list)
# phase_2_arr = np.array(phase_2_list)

# phase_diff =np.remainder(phase_2_arr-2*phase_1_arr,2*np.pi)
# phase_2_mod = np.remainder(phase_2_arr,2*np.pi)
# phase_1_2t_pi = np.remainder(2*phase_1_arr+np.pi,2*np.pi)

# plt.figure(figsize=(8,4))
# plt.plot(delta_ratio_list,phase_diff)



# Optimization for the delta_ratio. 

# def gate_opt(delta_ratio):
#     phase_1, phase_2 , fid = cz_test.CZ_ptcol_ideal_phase(
#                                             delta_ratio)
#     phase_diff = np.abs(phase_2-2*phase_1-np.pi)
#     return phase_diff

# delta_ratio_init = 0.25

# opt_delta_res = optimize.minimize(gate_opt, delta_ratio_init,
#                                   method='trust-constr',
#                                   options={'verbose':2,
#                                            'gtol':1e-9})

# opt_delta = opt_delta_res.x

# 0.24999835  optimal delta_ratio for the CZ gate. 



# Do some samplings to estimate the gate fidelity.
# TODO: average fidelity given by averaging over 
# Haar random input states have a nice analytic expression
# so that we can simply use a specific input state 
# to obtain the average fidelity. (This procedure seems 
# neccessary if we want to optimize parameters according to
# the gate fidelity since the sampling method is way too
# time consuming.) 

delta_opt = 0.24999835
phase_1, phase_2, fid = cz_test.CZ_ptcol_ideal_phase(delta_opt)
U_ideal_small = np.zeros((4,4),dtype=np.complex128)
U_ideal_small[0][0] = 1
U_ideal_small[1][1] = np.exp(1j*phase_1)
U_ideal_small[2][2] = np.exp(1j*phase_1)
U_ideal_small[3][3] = np.exp(1j*(2*phase_1+np.pi))

U_ideal = cz_test._matrix_embed(10, U_ideal_small)


# A special case for state_haar_small, which is invoked 
# later. 
# state_haar_small = np.array([
#     [0.5,0.5,0,0],
#     [0.5,0.5,0,0],
#     [0,0,0,0],
#     [0,0,0,0]
# ])


num_rounds = 200

fidel_list = []

for i in range(0,num_rounds):
    state_haar_small = cz_test._two_qubit_state_haar_dm()
    state_haar = cz_test._matrix_embed(10, state_haar_small)
    state_real_out = cz_test.CZ_real(delta_opt, state_haar)
    state_ideal_mid = np.matmul(U_ideal,state_haar)
    state_ideal_out = np.matmul(state_ideal_mid, 
                        np.conjugate(np.transpose(U_ideal)))

    fidel = np.trace(np.matmul(state_real_out, state_ideal_out))
    fidel_real = np.real_if_close(fidel)
    print(fidel_real)
    fidel_list.append(fidel_real)

avg_fidelity = np.sum(fidel_list)/num_rounds
min_fidelity = min(fidel_list)


# Avg fidelity: 0.9934
# Minimal fidelity: 0.9845
# test test