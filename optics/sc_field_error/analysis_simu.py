import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xpart as xp
import json

from scipy.interpolate import CubicSpline

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize

with open('/mnt/c/muco/code/data_saved/dipole_design_results/tuning_chormacorrect_no_errors.json', 'r') as file:
    data = json.load(file)
ex=np.array(data['eps_x'])
ey=np.array(data['eps_y'])
es=np.array(data['eps_s'])

with open('/mnt/c/muco/code/data_saved/dipole_design_results/tuning_chormacorrect_medV_2RT.json', 'r') as file:
    data = json.load(file)
eps_x=np.array(data['eps_x'])
eps_y=np.array(data['eps_y'])
eps_s=np.array(data['eps_s'])
transmission=np.array(data['tr'])

turn=np.linspace(0,55,1430)

plt.figure(figsize=(7, 5))
ax1 = plt.subplot(2, 1, 1)
plt.ylabel(r'$\Delta \epsilon / \epsilon_{x,0}$')
plt.plot(turn[-200:], (ex[-200:] - ex[0]) / ex[0], alpha=0.7, label='no error', color='tab:orange')
plt.plot(turn[-200:], (eps_x[-200:] - eps_x[0]) / eps_x[0], alpha=0.8, label='with errors', color='tab:blue')
plt.legend()
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.ylabel(r'$\Delta \epsilon /\epsilon_{y,0}$')
plt.plot(turn[-200:], (ey[-200:] - ey[0]) / ey[0], alpha=0.7, label='no error', color='tab:orange')
plt.plot(turn[-200:], (eps_y[-200:] - eps_y[0]) / eps_y[0], alpha=0.8, label='with errors', color='tab:blue')
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
ax1 = plt.subplot(2, 1, 1)
plt.ylabel('$ \Delta \epsilon_s / \epsilon_s$ ')
plt.plot(turn, (es - es[0]) / es[0], label='no error', color='tab:orange')
plt.plot(turn, (eps_s - eps_s[0]) / eps_s[0], label='with errors', color='tab:blue')
plt.legend() 
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.ylabel('tr')
plt.plot(turn, np.ones(n_turns), label='no error', color='tab:orange')
plt.plot(turn, transmission, label='with errors', color='tab:blue')
plt.xlabel('Turn')
plt.legend()  
plt.tight_layout()
plt.show()


