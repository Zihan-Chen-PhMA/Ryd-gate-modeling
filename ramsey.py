import numpy as np 
from arc import * 
import matplotlib.pyplot as plt 
from scipy import signal
from scipy import optimize
from scipy import constants
from scipy import integrate
from qutip import *
from utils import color_dict


atom = Rubidium87()

class Rabi():

    def __init__(self, temp, t_gate, ryd_level = 53) -> None:
        self.temp = temp 
        self.ryd_level = 53
        self.t_gate = t_gate
        self.Delta = 2*np.pi*(7.8)*10**(9)  # E_mid - omega_420
        self.delta = 0
        self.delta_m = -2*np.pi*(24)*10**(6) + self.delta
        self.atom = Rubidium87()


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
    
    def Ramsey_thermal_shot(self):
        """
        Test
        """
        self.doppler_shift_temp = 2*np.pi*self.doppler_shift_Hz()
        def fun(t, y):
            ham_temp = np.array(
                [[0, 0], 
                [0, self.doppler_shift_temp]]
            ,dtype=np.complex64)
            diff = -1j*ham_temp
            y_arr = np.reshape(np.array(y),(-1,1))
            return np.reshape(np.matmul(diff,y_arr),(-1))
        
        # print(state_mat)
        state_mat = np.array([1/np.sqrt(2), 1j*1/np.sqrt(2)],dtype=np.complex64)
        state_check = np.array([1, 0],dtype=np.complex64)
        # t_gate = 15*10**(-6)
        t_span = [0, self.t_gate]
        t_evals = np.linspace(0, self.t_gate, 2000)
        result = integrate.solve_ivp(fun, t_span, state_mat, t_eval=t_evals)
        u_matrix = np.array([
            [1/np.sqrt(2), 1j/np.sqrt(2)],
            [1j/np.sqrt(2), 1/np.sqrt(2)]
        ])
        result_state = np.matmul(u_matrix,np.array(result.y))
        fidel_list = np.reshape(np.matmul(
                                    np.conjugate(np.transpose(result_state)),
                                    np.reshape(state_check,(-1,1))),(-1))
        fidel_list = np.square(np.abs(fidel_list))
        return [t_evals, fidel_list]

temp = 40*10**(-6)
rabi_ramsey = Rabi(temp, t_gate=25*10**(-6))
# t_evals, fidel_list = rabi_ramsey.Ramsey_thermal_shot()


# plt.figure()
# plt.plot(t_evals, fidel_list)

def gauss_inv_decay(x,a):
    return 0.5*(1-np.exp(-(x/a)**2))


fidel_result_list = []
fidel_mean_list = []
n_shots = 50000


temp_arr = 10**(-6)*np.array([
    0.5, 1, 3, 5, 9, 13, 17, 21, 25, 30, 35, 40
])

plt.figure()
decay_param_list = []
print("Started")
for index, temperature in zip(range(len(temp_arr)),temp_arr):
    fidel_result_list = []
    fidel_mean_list = []
    rabi_ramsey = Rabi(temperature, t_gate=25*10**(-6))
    for i in range(n_shots):
        t_evals, fidel_temp = rabi_ramsey.Ramsey_thermal_shot()
        fidel_result_list.append(fidel_temp)
    fidel_result_arr = np.array(fidel_result_list)
    fidel_mean_list = np.mean(fidel_result_arr, axis=0)
    plt.plot(t_evals, fidel_mean_list, color=color_dict['Orange'],
             alpha=1-0.5*index/len(temp_arr))
    decay_param, _ = optimize.curve_fit(gauss_inv_decay, 
                    t_evals,
                    fidel_mean_list,p0=3*10**(-6))
    plt.plot(t_evals, gauss_inv_decay(t_evals,decay_param),
             color=color_dict['Blue'], alpha=1-0.5*index/len(temp_arr))
    decay_param_list.append(decay_param)
    print("progress: ", str(index+1)+'/'+str(len(temp_arr)))


plt.figure()
plt.plot(temp_arr*10**(6), np.array(decay_param_list)*10**(6))

    



# for i in range(n_shots):
#     t_evals, fidel_temp = rabi_ramsey.Ramsey_thermal_shot()
#     fidel_result_list.append(fidel_temp)


# fidel_result_arr = np.array(fidel_result_list)
# fidel_mean_list = np.mean(fidel_result_arr, axis=0)


# plt.figure()
# plt.plot(t_evals, fidel_mean_list)


# decay_param, _ = optimize.curve_fit(gauss_inv_decay, 
#                     t_evals,
#                     fidel_mean_list,p0=3*10**(-6))


# plt.plot(t_evals, gauss_inv_decay(t_evals,decay_param))