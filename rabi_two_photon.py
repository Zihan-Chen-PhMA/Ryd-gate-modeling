import numpy as np 
from arc import * 
import matplotlib.pyplot as plt 
from scipy import signal
from scipy import optimize
from scipy import constants
from qutip import *


atom = Rubidium87()

class Rabi():

    def __init__(self, rabi_420, rabi_1013,
                 temp, ryd_level = 53) -> None:
        self.rabi_420 = rabi_420
        self.rabi_1013 = rabi_1013
        self.rabi_gm = self.rabi_420
        self.rabi_gmgarb = self.rabi_gm/np.sqrt(3)
        self.rabi_mr = self.rabi_1013
        self.rabi_mrgarb = self.rabi_mr/np.sqrt(3)
        self.temp = temp 
        self.ryd_level = 53
        self.Delta = 2*np.pi*(7.8)*10**(9)  # E_mid - omega_420
        self.delta = 0
        self.delta_m = -2*np.pi*(24)*10**(6) + self.delta
        self.atom = Rubidium87()
        self.num_states = 1 + 2*2 + 2*2
        self.rabi_eff = self.rabi_420*self.rabi_1013/(
                            2*self.Delta)
        self.time_scale = 2*np.pi/self.rabi_eff
        self.time_list = np.linspace(0,self.time_scale*5,300*5)

        
        mid_decay_rate = 1/(self.atom.getStateLifetime(6,1,0.5))

        ryd_decay_rate = 1/(self.atom.getStateLifetime(
                                    self.ryd_level,
                                    0,0.5,300,self.ryd_level+30))


        mid_e_decay_mat = np.zeros((self.num_states,
                                    self.num_states))
        mid_e_decay_mat[5][1] = np.sqrt(mid_decay_rate)
        mid_e_decay_oper = Qobj(mid_e_decay_mat)

        mid_e_garb_decay_mat = np.zeros((self.num_states,
                                         self.num_states))
        mid_e_garb_decay_mat[6][2] = np.sqrt(mid_decay_rate)
        mid_e_garb_decay_oper = Qobj(mid_e_garb_decay_mat)

        ryd_decay_mat = np.zeros((self.num_states,
                                  self.num_states))
        ryd_decay_mat[7][3] = np.sqrt(ryd_decay_rate)
        ryd_decay_oper = Qobj(ryd_decay_mat)

        ryd_garb_decay_mat = np.zeros((self.num_states,
                                       self.num_states))
        ryd_garb_decay_mat[8][4] = np.sqrt(ryd_decay_rate)
        ryd_garb_decay_oper = Qobj(ryd_garb_decay_mat)

        self.c_ops = [mid_e_decay_oper, mid_e_garb_decay_oper,
                        ryd_decay_oper, ryd_garb_decay_oper]

        


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
    
    def rabi_shot(self):
        velocity = self.velocity_thermal_sample()
        doppler_shift_Hz = self.doppler_shift_Hz()
        doppler_shift = 2*np.pi*doppler_shift_Hz
        ham_mat = np.zeros((self.num_states, self.num_states))
        ham_diag = np.diag([0, self.Delta, self.Delta, 
                            self.delta + doppler_shift,
                            self.delta_m + doppler_shift,
                            0, 0, 0, 0])
        ham_mat = ham_mat + ham_diag
        # 420 
        ham_mat[1][0] = self.rabi_gm/2
        ham_mat[0][1] = np.conjugate(ham_mat[1][0])
        ham_mat[2][0] = self.rabi_gmgarb/2
        ham_mat[0][2] = np.conjugate(ham_mat[2][0])
        # 1013 
        ham_mat[3][1] = self.rabi_mr/2
        ham_mat[1][3] = np.conjugate(ham_mat[3][1])
        ham_mat[4][2] = self.rabi_mrgarb/2
        ham_mat[2][4] = np.conjugate(ham_mat[4][2])

        hamiltonian = Qobj(ham_mat)

        init_mat = np.zeros((self.num_states, 
                             self.num_states))
        init_mat[0][0] = 1
        rho_init = Qobj(init_mat)

        result = mesolve(hamiltonian, rho_init, 
                         self.time_list,
                         c_ops=self.c_ops)
        state_list = result.states
        state_arr_list = []
        for state_temp in state_list:
            state_temp_mat = state_temp.data.toarray()
            state_arr_list.append(state_temp_mat)
        return state_arr_list



rabi_420 = 2*np.pi*(237)*10**(6)
rabi_1013 = 2*np.pi*(303)*10**(6)
temp = 10*10**(-6)

trial_list = []
trial_gs_occ_list = []
trial_mid_state_scatter_list = []
trial_ryd_decay_list = []
trial_ryd_err_list = []

# rabi_zero = Rabi(rabi_420, rabi_1013, 0)
# state_arr_list = rabi_temp.rabi_shot()
# gs_occ_list = []
# for state_arr in state_arr_list:
#      gs_occ_list.append(np.real_if_close(
#                             state_arr[3][3]))

# t_list = [t for t in rabi_zero.time_list]
# plt.figure(figsize=(10,8))
# plt.plot(t_list,gs_occ_list,'--')


rabi_temp = Rabi(rabi_420, rabi_1013, temp)

for i in range(100):
    state_arr_list = rabi_temp.rabi_shot()
    # trial_list.append(state_arr_list)
    gs_occ_list = []
    mid_state_scatter_list = []
    ryd_decay_list = []
    ryd_err_list = []
    for state_arr in state_arr_list:
        gs_occ_list.append(np.real_if_close(
                            state_arr[3][3]))
        mid_state_scatter_list.append(np.real_if_close(
                state_arr[5][5] + state_arr[6][6]))
        ryd_decay_list.append(np.real_if_close(
            state_arr[7][7]))
        ryd_err_list.append(np.real_if_close(state_arr[4][4]+
            state_arr[8][8]))
    trial_gs_occ_list.append(gs_occ_list)
    trial_mid_state_scatter_list.append(mid_state_scatter_list)
    trial_ryd_decay_list.append(ryd_decay_list)
    trial_ryd_err_list.append(ryd_err_list)


trial_gs_occ_arr = np.array(trial_gs_occ_list)
trial_mid_state_scatter_list = np.array(trial_mid_state_scatter_list)
trial_ryd_decay_list = np.array(trial_ryd_decay_list)
trial_ryd_err_list = np.array(trial_ryd_err_list)


t_list = [t for t in rabi_temp.time_list]
t_list = 10**(6)*np.array(t_list)

gs_occ_mean_list = []
gs_occ_std_list = []
mid_state_scatter_mean_list = []
mid_state_scatter_std_list = []
ryd_decay_mean_list = []
ryd_decay_std_list = []
ryd_err_mean_list = []
ryd_err_std_list = []

for index, time in zip(range(len(t_list)),t_list):
    gs_occ_trials = trial_gs_occ_arr[:,index]
    gs_occ_mean_t = np.mean(gs_occ_trials)
    gs_occ_std_t = np.std(gs_occ_trials)
    gs_occ_mean_list.append(gs_occ_mean_t)
    gs_occ_std_list.append(gs_occ_std_t)
    mid_state_scatter_t = trial_mid_state_scatter_list[:,index]
    mid_state_scatter_mean_t = np.mean(mid_state_scatter_t)
    mid_state_scatter_std_t = np.std(mid_state_scatter_t)
    mid_state_scatter_mean_list.append(mid_state_scatter_mean_t)
    mid_state_scatter_std_list.append(mid_state_scatter_std_t)
    ryd_decay_t = trial_ryd_decay_list[:,index]
    ryd_decay_mean_t = np.mean(ryd_decay_t)
    ryd_decay_std_t = np.std(ryd_decay_t)
    ryd_decay_mean_list.append(ryd_decay_mean_t)
    ryd_decay_std_list.append(ryd_decay_std_t)
    ryd_err_t = trial_ryd_err_list[:,index]
    ryd_err_mean_t = np.mean(ryd_err_t)
    ryd_err_std_t = np.std(ryd_err_t)
    ryd_err_mean_list.append(ryd_err_mean_t)
    ryd_err_std_list.append(ryd_err_std_t)



gs_occ_mean_list = np.array(gs_occ_mean_list)
gs_occ_std_list = np.array(gs_occ_std_list)
mid_state_scatter_mean_list = np.array(mid_state_scatter_mean_list)
mid_state_scatter_std_list = np.array(mid_state_scatter_std_list)
ryd_decay_mean_list = np.array(ryd_decay_mean_list)
ryd_decay_std_list = np.array(ryd_decay_std_list)
ryd_err_mean_list = np.array(ryd_err_mean_list)
ryd_err_std_list = np.array(ryd_err_std_list)


plt.figure(figsize=(10,8))
plt.plot(t_list,gs_occ_mean_list,
           color='#3366CC',lw=0.5)
plt.fill_between(t_list,gs_occ_mean_list-gs_occ_std_list,
                 gs_occ_mean_list+gs_occ_std_list,color='#3366CC',
                 alpha=0.5)
plt.xlabel(r"time $(\mu s)$")
plt.ylabel(r"Rydberg state occupation probability")
# plt.legend()
plt.grid(True)
plt.savefig('rabi_10_micro_K_100_shots.pdf')


plt.figure(figsize=(10,8))
plt.plot(t_list,mid_state_scatter_mean_list,
           color='#E67300',lw=0.5,
           label=r"Mid-state scattering")
plt.fill_between(t_list,
                 mid_state_scatter_mean_list-
                            mid_state_scatter_std_list,
                 mid_state_scatter_mean_list+
                            mid_state_scatter_std_list,
                 color='#E67300',
                 alpha=0.5)
plt.plot(t_list,ryd_decay_mean_list,
           color='#109618',lw=0.5,
           label=r"Rydberg decay")
plt.fill_between(t_list,
                 ryd_decay_mean_list-
                            ryd_decay_std_list,
                 ryd_decay_mean_list+
                            ryd_decay_std_list,
                 color='#109618',
                 alpha=0.5)
plt.plot(t_list,ryd_err_mean_list,
           color='#B82E2E',lw=0.5,
           label=r"Unwanted Rydberg excitation")
plt.fill_between(t_list,
                 ryd_err_mean_list-
                            ryd_err_std_list,
                 ryd_err_mean_list+
                            ryd_err_std_list,
                 color='#B82E2E',
                 alpha=0.5)
plt.xlabel(r"time $(\mu s)$")
plt.ylabel(r"Error probability")
plt.legend()
plt.grid(True)
plt.savefig('errors_of_rabi_10_micro_K_100_shots.pdf')


# ryd decay rate: array([0.00675864])
# mid scatter rate: array([0.00273223])


# temp = 20*10**(-6)
# rabi_temp = Rabi(rabi_420, rabi_1013, temp)
# trial_list = []
# trial_gs_occ_list = []

# for i in range(5):
#     state_arr_list = rabi_temp.rabi_shot()
#     # trial_list.append(state_arr_list)
#     gs_occ_list = []
#     for state_arr in state_arr_list:
#         gs_occ_list.append(np.real_if_close(
#                             state_arr[3][3]))
#     trial_gs_occ_list.append(gs_occ_list)

# trial_gs_occ_arr = np.array(trial_gs_occ_list)

# t_list = [t for t in rabi_temp.time_list]

# gs_occ_mean_list = []
# gs_occ_std_list = []

# for index, time in zip(range(len(t_list)),t_list):
#     gs_occ_trials = trial_gs_occ_arr[:,index]
#     gs_occ_mean_t = np.mean(gs_occ_trials)
#     gs_occ_std_t = np.std(gs_occ_trials)
#     gs_occ_mean_list.append(gs_occ_mean_t)
#     gs_occ_std_list.append(gs_occ_std_t)

# gs_occ_mean_list = np.array(gs_occ_mean_list)
# gs_occ_std_list = np.array(gs_occ_std_list)


# plt.plot(t_list,gs_occ_mean_list,
#            color='#109618',lw=0.5)
# plt.fill_between(t_list,gs_occ_mean_list-gs_occ_std_list,
#                  gs_occ_mean_list+gs_occ_std_list,color='#109618',
#                  alpha=0.5)

# rabi_420 = 2*np.pi*(237)*10**(6)
# rabi_gm = rabi_420*np.sqrt(3)/2
# rabi_1013 = 2*np.pi*(303)*10**(6)
# rabi_mr =  rabi_1013*2/np.sqrt(3)
# Delta = 2*np.pi*(7.8)*10**(9)  # E_mid - omega_420
# time_scale = 2*np.pi/(rabi_420*rabi_1013/
#                       (2*Delta))
# delta = 0  # E_ryd_+ - omega_420 - omega_1013
# delta_m = -2*np.pi*(24)*10**(6) + delta  # E_ryd_- - (...)
# temperature = 300
# ryd_level = 53

# d_mid_ratio = (atom.getDipoleMatrixElement(5,0,0.5,
#                     -0.5,6,1,1.5,0.5,1)/
#                atom.getDipoleMatrixElement(5,0,0.5,
#                     0.5,6,1,1.5,1.5,1))

# d_ryd_ratio = (atom.getDipoleMatrixElement(6,1,1.5,
#                     0.5,53,0,0.5,-0.5,-1)/
#                atom.getDipoleMatrixElement(6,1,1.5,
#                     1.5,53,0,0.5,0.5,-1))

# rabi_420_garbage = rabi_420*d_mid_ratio
# rabi_1013_garbage = rabi_1013*d_ryd_ratio


# num_states = 1 + 2*2 + 2*2 
# one |1> = |F=2,mF=0> state,
# two 6P3/2 states and each one of them is associated
# with a garbage decay state.
# two Rydberg S states and each one of them is 
# associated with a garbage decay state.
# Check the overleaf document for more info on
# energy levels. 


# ham_mat = np.zeros((num_states, num_states))
# ham_diag = np.diag([0, Delta, Delta, delta, delta_m,
#                     0, 0, 0, 0])
# ham_mat = ham_mat + ham_diag
# # 420 
# ham_mat[1][0] = rabi_420/2
# ham_mat[0][1] = np.conjugate(ham_mat[1][0])
# ham_mat[2][0] = rabi_420_garbage/2
# ham_mat[0][2] = np.conjugate(ham_mat[2][0])
# # 1013 
# ham_mat[3][1] = rabi_1013/2
# ham_mat[1][3] = np.conjugate(ham_mat[3][1])
# ham_mat[4][2] = rabi_1013_garbage/2
# ham_mat[2][4] = np.conjugate(ham_mat[4][2])

# hamiltonian = Qobj(ham_mat)

# init_mat = np.zeros((num_states, num_states))
# init_mat[0][0] = 1
# rho_init = Qobj(init_mat)

# mid_decay_rate = 1/(atom.getStateLifetime(6,1,0.5))

# ryd_decay_rate = 1/(atom.getStateLifetime(ryd_level,
#                                 0,0.5,300,ryd_level+30))

# mid_e_decay_mat = np.zeros((num_states,num_states))
# mid_e_decay_mat[5][1] = np.sqrt(mid_decay_rate)
# mid_e_decay_oper = Qobj(mid_e_decay_mat)

# mid_e_garb_decay_mat = np.zeros((num_states,num_states))
# mid_e_garb_decay_mat[6][2] = np.sqrt(mid_decay_rate)
# mid_e_garb_decay_oper = Qobj(mid_e_garb_decay_mat)

# ryd_decay_mat = np.zeros((num_states,num_states))
# ryd_decay_mat[7][3] = np.sqrt(ryd_decay_rate)
# ryd_decay_oper = Qobj(ryd_decay_mat)

# ryd_garb_decay_mat = np.zeros((num_states,num_states))
# ryd_garb_decay_mat[8][4] = np.sqrt(ryd_decay_rate)
# ryd_garb_decay_oper = Qobj(ryd_garb_decay_mat)

# c_ops = [mid_e_decay_oper, mid_e_garb_decay_oper,
#          ryd_decay_oper, ryd_garb_decay_oper]


# time_list = np.linspace(0,time_scale*20,300*20)

# result = mesolve(hamiltonian, rho_init, time_list,
#                  c_ops=c_ops)

# state_list = result.states


def exp_decay(x,a):
    return np.exp(-a*x)



# Ground state occupation at each time point during
# Rabi oscillation. 

# gs_occ_list = []
# for i in range(len(time_list)):
#     state_temp = state_list[i]
#     state_temp_mat = state_temp.data.toarray()
#     gs_occ_list.append(np.real_if_close(
#                         state_temp_mat[3][3]))
# peaks, _ = signal.find_peaks(gs_occ_list)
# peaks = peaks.tolist()
# peaks = peaks
# plt.figure(figsize=(16,5))
# t_list = [i for i in time_list]
# peaks_time = [t_list[i] for i in peaks]
# peaks_val = [gs_occ_list[i] for i in peaks]
# plt.plot(t_list,gs_occ_list,
#           color='#3366CC',lw=2.7)

# plt.plot(peaks_time, peaks_val,'x',
#           color='#B82E2E',lw=3,markersize=8)

# decay_param, _ = optimize.curve_fit(exp_decay, 
#                     peaks_time, peaks_val)

# plt.plot(time_list, exp_decay(time_list, decay_param),
#          '--',lw=2.7,color='#109618')



# gs_occ_list = []
# for state_arr in state_arr_list:
#     gs_occ_list.append(np.real_if_close(
#                         state_arr[3][3]))
# peaks, _ = signal.find_peaks(gs_occ_list)
# peaks = peaks.tolist()
# peaks = peaks
# plt.figure(figsize=(16,5))
# t_list = [i for i in rabi_temp.time_list]
# peaks_time = [t_list[i] for i in peaks]
# peaks_val = [gs_occ_list[i] for i in peaks]
# plt.plot(t_list,gs_occ_list,
#           color='#3366CC',lw=2.7)

# plt.plot(peaks_time, peaks_val,'x',
#           color='#B82E2E',lw=3,markersize=8)

# decay_param, _ = optimize.curve_fit(exp_decay, 
#                     peaks_time, peaks_val)

# plt.plot(t_list, exp_decay(t_list, decay_param),
#          '--',lw=2.7,color='#109618')