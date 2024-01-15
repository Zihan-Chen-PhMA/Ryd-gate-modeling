from qutip import *
import numpy as np
import random
import matplotlib.pyplot as plt

fwidth = 1 #MHz, the bandwidth of power spectrum of noise phase
N = 1000 # cut 0-1 MHz into 1000 parts
deltaf = fwidth/N #MHz
I = 0.004 # total Amplitude of phase noise
PSD = I / fwidth # phase noise power evenly distributed among 0-1MHz
tau = 6 #second simulation time
ptnum = 1000 # time segments

Omega = 4.29268 #Rabi frequency
Delta = 0 # detuning

def noise_coeff(t,args):
    phase_noise = 0
    phi_vec = args['phi']
    for i in range(N):
        phase_noise = phase_noise + 2*np.sqrt(PSD * deltaf) * np.cos(deltaf*i*2*np.pi*t + phi_vec[i])
    return np.exp(1.0j*phase_noise)

def noise_coeff_dag(t,args):
    phase_noise = 0
    phi_vec = args['phi']
    for i in range(N):
        phase_noise = phase_noise + 2*np.sqrt(PSD * deltaf) * np.cos(deltaf*i*2*np.pi*t + phi_vec[i])
    return np.exp(-1.0j*phase_noise)

# Sampling phase noise.
phi_vec = 2*np.pi * np.random.random(size = N)

#defined time-dependent Hamiltonian given sampled noise
args = {'phi':phi_vec}
H = [[0.5 * Omega * sigmap(), noise_coeff],[0.5 * Omega * sigmam(),noise_coeff_dag]] 

time = np.linspace(0,tau,ptnum)
psi0 = basis(2, 0)
result = sesolve(H, psi0, time, args = args)
p = np.zeros(ptnum)
for i in range(ptnum):
    p[i] = np.abs(result.states[i][0,0])**2

#Plot Prob(ground state) changed with time
plt.figure()
plt.grid()
plt.plot(time,p,label = 'ground state')
plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0)
plt.show()



# plt.figure()
# plt.grid()
# plt.plot(time,phase_noise,label = 'Noise phase')
# plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0)
# plt.show()

