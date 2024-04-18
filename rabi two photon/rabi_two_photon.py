import numpy as np 
from arc import * 
import matplotlib.pyplot as plt 
from scipy import signal
from scipy import optimize
from qutip import *


atom = Rubidium87()

rabi_420 = 2*np.pi*(237)*10**(6)
rabi_1013 = 2*np.pi*(303)*10**(6)
Delta = 2*np.pi*(7.8)*10**(9)  # E_mid - omega_420
time_scale = 2*np.pi/(rabi_420*rabi_1013/
                      (2*Delta))
delta = 0  # E_ryd_+ - omega_420 - omega_1013
delta_m = -2*np.pi*(24)*10**(6) + delta  # E_ryd_- - (...)
temperature = 300
ryd_level = 53

d_mid_ratio = (atom.getDipoleMatrixElement(5,0,0.5,
                    -0.5,6,1,1.5,0.5,1)/
               atom.getDipoleMatrixElement(5,0,0.5,
                    0.5,6,1,1.5,1.5,1))

d_ryd_ratio = (atom.getDipoleMatrixElement(6,1,1.5,
                    0.5,53,0,0.5,-0.5,-1)/
               atom.getDipoleMatrixElement(6,1,1.5,
                    1.5,53,0,0.5,0.5,-1))

rabi_420_garbage = rabi_420*d_mid_ratio
rabi_1013_garbage = rabi_1013*d_ryd_ratio


num_states = 1 + 2*2 + 2*2 
# one |1> = |F=2,mF=0> state,
# two 6P3/2 states and each one of them is associated
# with a garbage decay state.
# two Rydberg S states and each one of them is 
# associated with a garbage decay state.
# Check the overleaf document for more info on
# energy levels. 


ham_mat = np.zeros((num_states, num_states))
ham_diag = np.diag([0, Delta, Delta, delta, delta_m,
                    0, 0, 0, 0])
ham_mat = ham_mat + ham_diag
# 420 
ham_mat[1][0] = rabi_420/2
ham_mat[0][1] = np.conjugate(ham_mat[1][0])
ham_mat[2][0] = rabi_420_garbage/2
ham_mat[0][2] = np.conjugate(ham_mat[2][0])
# 1013 
ham_mat[3][1] = rabi_1013/2
ham_mat[1][3] = np.conjugate(ham_mat[3][1])
ham_mat[4][2] = rabi_1013_garbage/2
ham_mat[2][4] = np.conjugate(ham_mat[4][2])

hamiltonian = Qobj(ham_mat)

init_mat = np.zeros((num_states, num_states))
init_mat[0][0] = 1
rho_init = Qobj(init_mat)

mid_decay_rate = 1/(atom.getStateLifetime(6,1,0.5))

ryd_decay_rate = 1/(atom.getStateLifetime(ryd_level,
                                0,0.5,300,ryd_level+30))

mid_e_decay_mat = np.zeros((num_states,num_states))
mid_e_decay_mat[5][1] = np.sqrt(mid_decay_rate)
mid_e_decay_oper = Qobj(mid_e_decay_mat)

mid_e_garb_decay_mat = np.zeros((num_states,num_states))
mid_e_garb_decay_mat[6][2] = np.sqrt(mid_decay_rate)
mid_e_garb_decay_oper = Qobj(mid_e_garb_decay_mat)

ryd_decay_mat = np.zeros((num_states,num_states))
ryd_decay_mat[7][3] = np.sqrt(ryd_decay_rate)
ryd_decay_oper = Qobj(ryd_decay_mat)

ryd_garb_decay_mat = np.zeros((num_states,num_states))
ryd_garb_decay_mat[8][4] = np.sqrt(ryd_decay_rate)
ryd_garb_decay_oper = Qobj(ryd_garb_decay_mat)

c_ops = [mid_e_decay_oper, mid_e_garb_decay_oper,
         ryd_decay_oper, ryd_garb_decay_oper]


time_list = np.linspace(0,time_scale*20,300*20)

result = mesolve(hamiltonian, rho_init, time_list,
                 c_ops=c_ops)

state_list = result.states


def exp_decay(x,a):
    return np.exp(-a*x)



# Ground state occupation at each time point during
# Rabi oscillation. 

gs_occ_list = []
for i in range(len(time_list)):
    state_temp = state_list[i]
    state_temp_mat = state_temp.data.toarray()
    gs_occ_list.append(np.real_if_close(
                        state_temp_mat[3][3]))
peaks, _ = signal.find_peaks(gs_occ_list)
peaks = peaks.tolist()
peaks = peaks
plt.figure(figsize=(16,5))
t_list = [i for i in time_list]
t_list = np.array(t_list)*10**6
peaks_time = [t_list[i] for i in peaks]
peaks_val = [gs_occ_list[i] for i in peaks]
peaks_time_arr = np.array(peaks_time)
peaks_time_arr = peaks_time_arr - peaks_time[0]
peaks_val_arr = np.array(peaks_val)
peaks_val_arr = peaks_val_arr/peaks_val[0]
plt.plot(t_list,gs_occ_list,
          color='#3366CC',lw=2.7)

plt.plot(peaks_time, peaks_val,'x',
          color='#B82E2E',lw=3,markersize=8)

decay_param, _ = optimize.curve_fit(exp_decay, 
                    peaks_time_arr,
                    peaks_val_arr)

plt.plot(t_list, (peaks_val[0]*
          exp_decay(t_list-peaks_time[0], decay_param)),
         '--',lw=2.7,color='#109618')
plt.xlabel(r"time $(\mu s)$")
plt.ylabel(r"Rydberg state occupation probability")
plt.grid(True)
plt.savefig('rabi_decay.pdf')

print(peaks_val[0])

# decay rate is array([0.00928182])