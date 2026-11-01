import numpy as np
import matplotlib.pyplot as plt
from class_geometry.ring_geometry.ring_geometry import RingGeometry
import scipy.constants as sc

file = 'class_geometry/parameter_files/para_RCS_LE.txt'
rcs = RingGeometry(file)
t_acc = 0.34*1e-3
n_turn_rcs = 17

n_point = 20
nb_arc = np.array([8, 16, 32])
color = ['tab:orange', 'tab:green', 'tab:purple']

n_turn_arc = n_turn_rcs * nb_arc

res = {}
energy_values_list = []
t_arc_list = []

for n_arc, n_turn in zip(nb_arc, n_turn_arc):
    res[str(n_arc)] = {}

    energy_increment_per_arc = (
        rcs.extraction_energy - rcs.injection_energy
    ) / n_turn

    energy_values = (
        rcs.injection_energy
        + np.arange(n_arc) * energy_increment_per_arc
    )
    energy_values = np.repeat(energy_values, n_point)
    # energy_values += 0.5 * energy_increment_per_arc
    energy_values = np.append(
        energy_values,
        energy_values[-1] + 1 * energy_increment_per_arc
    )

    t_arc = np.linspace(0, 1 / n_turn_rcs, n_arc * n_point)
    t_arc = np.append(t_arc, 1 / n_turn_rcs)
    t_arc -= 0.5/(n_turn_rcs*n_arc)

    B_plot = rcs.dipole_families['BNC']['B'](t_arc)
    rho = energy_values / B_plot / sc.c

    res[str(n_arc)]['E'] = energy_values
    res[str(n_arc)]['t'] = t_arc
    res[str(n_arc)]['rho'] = rho
    res[str(n_arc)]['t_arc'] = t_arc
    res[str(n_arc)]['B'] = B_plot

fig1 = plt.figure(figsize=(6.4,4.8))
plt.plot(t_arc, B_plot*rcs.rho*sc.c * 1e-9, label=r'B*$\rho$*c', color='k')
for ik, key in enumerate(res.keys()):
    plt.plot(res[key]['t'], res[key]['E'] * 1e-9, label='E (' + key +' RF sections)', color=color[ik])
plt.xlabel('$t \ / \ t_{acc}$')
plt.ylabel('GeV')
plt.legend()
plt.tight_layout()
plt.show()