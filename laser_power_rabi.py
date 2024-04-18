import numpy as np 
from arc import * 
import matplotlib.pyplot as plt 
from scipy import signal
from scipy import optimize
from qutip import *
from scipy import constants
import matplotlib.colors as colors

atom = Rubidium87()

# SI units
power = 1
width = 10*10**(-6)    # This can be kept constant
                        # because we only have a single
                        # sheet of atoms
height = 35*10**(-6)     # This should grow if we want
                        # to do more CZ gates at the same
                        # time. (Each CZ gate site should 
                        # be separated by 10um.)
# area = height*width
# e_field = np.sqrt(2*power/(
#                     constants.epsilon_0*constants.c*
#                     area))

class Laser():

    def __init__(self, power, height, width) -> None:
        self.power =  power
        self.height = height
        self.width = width 
        self.area = self.height*self.width
        self.e_field = np.sqrt(2*self.power/(
                            constants.epsilon_0*constants.c*
                            self.area))
        self.prefactor = 2*self.e_field/constants.hbar


laser_1013 = Laser(power, height, width)


# prefactor = 2*e_field/constants.hbar    
# factor of 2 in prefactor is the correction factor
# for lukin's convention. 
# Originally omega = <d \cdot E>/hbar which is exact
# the non-diagonal elements of the Hamiltonian. 
# However, Lukin set those non-diagonal elements as
# Omega/2. 
gs_factor = np.sqrt(3)/(2*np.sqrt(2))
dp_moment_au_to_si = (constants.e*
            constants.physical_constants['Bohr radius'][0])
# In ARC, dipole moment has unit [e*r_bohr]

omega = (laser_1013.prefactor*dp_moment_au_to_si*gs_factor*
            atom.getDipoleMatrixElement(5,0,0.5,0.5,
                                        6,1,1.5,1.5,
                                        1))
omega_MHz = omega/(2*np.pi*10**6)
print(omega_MHz)







levs = range(40,100)
omega_list = []
dp_list = []

for n_lev in levs:
    dp_moment_au_n = np.abs(atom.getDipoleMatrixElement(6,1,1.5,1.5,
                        n_lev,0,0.5,0.5,-1))
    # dp_moment_au_n = atom.getDipoleMatrixElement(5,0,0.5,0.5,
    #                     n_lev,1,0.5,-0.5,-1)
    omega_n = (laser_1013.prefactor*dp_moment_au_to_si*
                    dp_moment_au_n)
    omega_n_MHz = np.abs(omega_n/(2*np.pi*10**6))
    dp_list.append(dp_moment_au_n)
    omega_list.append(omega_n_MHz)

plt.plot(levs,omega_list,color='#3366CC',lw=2,
            marker='o',
            markeredgecolor='#3366CC',
            markerfacecolor=colors.to_rgba('#3366CC', 0.8),
            markersize=6
            )
plt.xlabel("n")
plt.ylabel(r"Rabi frequency (MHz)")
# plt.legend()
plt.grid(True)
plt.savefig('ryd_rabi_freq_1013_1W_laser_power.pdf')



color_list = [
    '#3366CC', # Blue
    '#0099C6', # SkyBlue
    '#22AA99', # Teal
    '#DC3912', # Red
    '#B82E2E', # FireBrick
    '#DD4477', # Pink
    '#FF9900', # Orange
    '#E67300', # DeepOrange
    '#109618', # Green
    '#66AA00', # LightGreen
    '#990099', # Purple
]




power = 0.03
width = 10*10**(-6)    
height = 35*10**(-6) 

laser_297 = Laser(power, height, width)

sg_factor = 1/np.sqrt(2)

levs = range(40,100)
omega_list = []
dp_list = []

for n_lev in levs:
    dp_moment_au_n = np.abs(atom.getDipoleMatrixElement(5,0,0.5,-0.5,
                        n_lev,1,0.5,0.5,1))
    # dp_moment_au_n = atom.getDipoleMatrixElement(5,0,0.5,0.5,
    #                     n_lev,1,0.5,-0.5,-1)
    omega_n = (laser_297.prefactor*dp_moment_au_to_si*
                    dp_moment_au_n*sg_factor)
    omega_n_MHz = np.abs(omega_n/(2*np.pi*10**6))
    dp_list.append(dp_moment_au_n)
    omega_list.append(omega_n_MHz)

plt.figure()
plt.plot(levs,omega_list,color='#109618',lw=2,
            marker='o',
            markeredgecolor='#109618',
            markerfacecolor=colors.to_rgba('#109618', 0.8),
            markersize=6
            )
plt.xlabel("n")
plt.ylabel(r"Rabi frequency (MHz)")
# plt.legend()
plt.grid(True)
plt.savefig('ryd_rabi_freq_297_30mW_laser_power.pdf')

