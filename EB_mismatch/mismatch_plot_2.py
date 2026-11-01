import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

t_acc = 0.34*1e-3
n_turn_rcs = 17
injection_energy = 63e9
extraction_energy = 313.83e9
extraction_B = 1.8
injection_B = injection_energy/extraction_energy*extraction_B
circumference = 5990
rho = extraction_energy/sc.c/extraction_B

n_point = 20
nb_arc = np.array([8, 16, 32])
color = ['tab:orange', 'tab:green', 'tab:purple']

n_turn_arc = n_turn_rcs * nb_arc

res = {}
energy_values_list = []
t_arc_list = []

t_arc = np.array([[]])

for n_arc, n_turn in zip(nb_arc, n_turn_arc):
    res_key = str(n_arc)
    res[res_key] = {}

    t_1turn = np.zeros((n_point + 1) * n_arc)
    energy_values = np.zeros((n_point + 1) * n_arc)

    t_arc = np.linspace(0, 1, n_point + 1) / n_turn

    energy_increment_per_arc = (
        extraction_energy - injection_energy
    ) / n_turn

    energy_loc = injection_energy
    i0 = 0
    i1 = n_point + 1
    dt = 0.5/n_turn
    for i in np.arange(n_arc):
        t_1turn[i0: i1] = t_arc + i / n_turn
        energy_values[i0: i1] = energy_loc
        energy_loc += energy_increment_per_arc
        i0 = i1
        i1 += n_point + 1

    B_plot = injection_B + (extraction_B-injection_B)*(t_1turn-dt)
    rho_ramp = energy_values / B_plot / sc.c

    res[res_key]['E'] = energy_values
    res[res_key]['t'] = t_1turn
    res[res_key]['rho'] = rho_ramp
    res[res_key]['t_arc'] = t_1turn
    res[res_key]['B'] = B_plot

res_keys = [str(key) for key in nb_arc]
fig1 = plt.figure()
for ik, key in enumerate(res_keys):
    plt.plot(
        res[key]['t'], 
        res[key]['E'] * 1e-9, 
        label=f'E ({key} RF sectionvs)', 
        color=color[ik],
        )
    plt.plot(
        res[key]['t'], 
        res[key]['B'] * rho * sc.c * 1e-9, 
        label=r'B*$\rho$*c ' + f'({key} RF sectionvs)', 
        color=color[ik],
        linestyle="--"
        )

plt.xlabel(r'$t / t_{acc}$')
plt.ylabel(r'$E$ [GeV]')
plt.legend()
plt.show()

fig2 = plt.figure(figsize=(6.4,4.8))
for ik, key in enumerate(res.keys()):
    dE = (res[key]['E'] - res[key]['B']*rho*sc.c)/res[key]['E']
    plt.plot(
        res[key]['t'], 
        dE * 100, 
        label=f'E ({key} RF sections)', 
        color=color[ik])
    print(key, np.min(dE)*100, np.max(dE)*100)
plt.ylabel(r'$\Delta E / E$ [%]')
plt.xlabel(r'$t \ / \ t_{acc}$')
plt.ylim(-2,2)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
for ik, key in enumerate(res.keys()):
    plt.plot(
        res[key]['t'][:-1], 
        1/res[key]['rho'][:-1] - 1/res[key]['rho'][0], 
        color=color[ik]
        )
plt.xlabel(r'$t / t_{acc}$')
plt.ylabel(r'$1/\rho$ [1/m]')
plt.show()
