import numpy as np 
from arc import * 
import matplotlib.pyplot as plt 
import matplotlib.colors as colors

atom = Rubidium87()



def ev_to_MHz(energy_ev): 
    return energy_ev*(2.418*10**8)

def ev_to_GHz(energy_ev):
    return energy_ev*(2.418*10**5)

n_list = [n for n in range(39,100)]
energy_MHz_nS_list = []
energy_GHz_nS_list = []
energy_MHz_nP1_list = []    # J = 1/2
energy_GHz_nP1_list = []
energy_MHz_nP3_list = []    # J = 3/2
energy_GHz_nP3_list = []



for n in n_list:
    energy_MHz_nS_list.append(ev_to_MHz(
                                atom.getEnergy(n,0,0.5)))
    energy_GHz_nS_list.append(ev_to_GHz(
                                atom.getEnergy(n,0,0.5)))
    energy_MHz_nP1_list.append(ev_to_MHz(
                                atom.getEnergy(n,1,0.5)))
    energy_GHz_nP1_list.append(ev_to_GHz(
                                atom.getEnergy(n,1,0.5)))
    energy_MHz_nP3_list.append(ev_to_MHz(
                                atom.getEnergy(n,1,1.5)))
    energy_GHz_nP3_list.append(ev_to_GHz(
                                atom.getEnergy(n,1,1.5)))

n_plot_list = n_list[1:]
energy_gap_GHz_nS_list = []
energy_gap_GHz_nP1_list = []
energy_gap_GHz_nP3_list = []
energy_PS_GHz_nSP1_list = []
energy_PS_GHz_nSP3_list = []
energy_FS_GHz_nP13_list = []

for i in range(0,len(n_plot_list)):
    energy_gap_GHz_nS_list.append((energy_GHz_nS_list[i+1]-
                                   energy_GHz_nS_list[i]))
    energy_gap_GHz_nP1_list.append((energy_GHz_nP1_list[i+1]-
                                    energy_GHz_nP1_list[i]))
    energy_gap_GHz_nP3_list.append((energy_GHz_nP3_list[i+1]-
                                    energy_GHz_nP3_list[i]))
    energy_PS_GHz_nSP1_list.append((energy_GHz_nP1_list[i+1]-
                                    energy_GHz_nS_list[i+1]))
    energy_PS_GHz_nSP3_list.append((energy_GHz_nP3_list[i+1]-
                                    energy_GHz_nS_list[i+1]))
    energy_FS_GHz_nP13_list.append((energy_GHz_nP3_list[i+1]-
                                    energy_GHz_nP1_list[i+1]))

plt.figure()
# In this plot, I used Google's color chart, which
# looks really nice. 
plt.plot(n_plot_list,energy_gap_GHz_nS_list,
            label=r'$E(nS_{1/2})-E([n-1]S_{1/2})$',
            marker='o',
            color='#FF9900',lw=2,
            markeredgecolor='#FF9900',
            markerfacecolor=colors.to_rgba('#FF9900',0.8))
            # 0.8 is the transparency of the marker.

plt.plot(n_plot_list,energy_gap_GHz_nP1_list,
            label=r'$E(nP_{1/2})-E([n-1]P_{1/2})$',
            marker='o',
            color='#3366CC',lw=2,
            markeredgecolor='#3366CC',
            markerfacecolor=colors.to_rgba('#3366CC',0.8))
plt.xlabel("n")
plt.ylabel("Energy gap (GHz)")
plt.legend()
plt.grid(True)
plt.savefig('ryd_energy_gap.pdf')


plt.figure()
# In the following two plots, I used built-in colors of
# matplotlib.
plt.plot(n_plot_list,energy_PS_GHz_nSP1_list,
            marker='o',
            color='firebrick', lw=2,
            markeredgecolor='firebrick',
            markerfacecolor=colors.to_rgba('firebrick',0.6))
plt.xlabel("n")
plt.ylabel(r"$\mathrm{E}(nP_{1/2})-\mathrm{E}(nS_{1/2})$ (GHz)")
plt.legend()
plt.grid(True)
plt.savefig('ryd_energy_diff_PS.pdf')

plt.figure()
plt.plot(n_plot_list,energy_FS_GHz_nP13_list,
            marker='o',
            color='darkorange', lw=2,
            markeredgecolor='darkorange',
            markerfacecolor=colors.to_rgba('darkorange', 0.8))
plt.xlabel("n")
plt.ylabel(r"$\mathrm{E}(nP_{3/2})-\mathrm{E}(nP_{1/2})$ (GHz)")
plt.legend()
plt.grid(True)
plt.savefig('ryd_energy_diff_P31.pdf')



