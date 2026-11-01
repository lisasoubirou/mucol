import json
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/mnt/c/muco/rcsparameters/rcsparameters')
from ring_geometry.ring_geometry import RingGeometry

file_input='/mnt/c/muco/mucol/class_geometry/parameter_files/para_RCS_CERN_RCS3.txt'
RCS = RingGeometry(file_input)
t_print=0.5
y_min, y_max,width,max_apert_noshift,max_apert=RCS.extrema(t_print)
print('Results for t=',t_print)
print('Theta',RCS.theta(t_print))
print('Width', width)
print('Max apert no shift',max_apert_noshift)
print('Max width from t_inj to t_ext',np.max(RCS.extrema(RCS.t_ext)[1])-np.min(RCS.extrema(RCS.t_inj)[0]))
print('Path length',RCS.path_length(t_print))
print('Path length tot',RCS.path_length_tot(t_print))

# Plot n_traj between injection and extraction
n_traj=8
t_traj=np.linspace(RCS.t_inj,RCS.t_ext,n_traj)
RCS.plot_traj(t_traj)

#Scan in Bnc
B_nc_max=3.0
B_nc_min=1.8
B_nc_scan=np.linspace(B_nc_min,B_nc_max,10)
E_ext=RCS.extraction_energy
E_inj=RCS.injection_energy
dE=E_ext-E_inj
E_ext_scan=B_nc_scan/B_nc_min*dE+E_inj

dBrho=RCS.ext_Brho-RCS.inj_Brho
aBrho=RCS.ext_Brho+RCS.inj_Brho
L_tot=RCS.dipole_families['BNC']['tot_length']+RCS.dipole_families['BSC']['tot_length']
B_sc_scan=B_nc_scan*aBrho/(L_tot*B_nc_scan/np.pi-dBrho)

t_np=np.linspace(RCS.t_inj,RCS.t_ext,11)
Dpath=np.ones((2,len(B_nc_scan)))
Dwidth_nc=np.ones((2,len(B_nc_scan)))
Dwidth_sc=np.ones((2,len(B_nc_scan)))
L_nc=np.ones((2,len(B_nc_scan)))
L_sc=np.ones((2,len(B_nc_scan)))
pat=[['BSC','BNC','BSC'],['BSC','BNC','BSC','BNC','BSC']]

scan_bnc=False
scan_bnc_bsc=False
scan_bnc_e=True

if scan_bnc==True:
    for i, B_nc in enumerate(B_nc_scan):
        for i_pat,pattern in enumerate(pat):
            RCS.pattern=pattern
            RCS.dipoles={"BSC": 16., "BNC": (B_nc, "linear")}
            mask_nc=[item == 'BNC' for item in RCS.pattern]
            mask_sc=[item == 'BSC' for item in RCS.pattern]
            print('energy', RCS.extraction_energy)
            print('B_nc',B_nc, RCS.dipoles['BNC'])
            # RCS.plot_traj(t_traj)
            Dpath[i_pat,i]=RCS.max_path_diff
            Dwidth_nc[i_pat,i]=RCS.arc_length
            max_width_dip=RCS.extrema(RCS.t_ext)[1]-RCS.extrema(RCS.t_inj)[0]
            Dwidth_sc[i_pat,i]=np.max(max_width_dip[mask_sc])
            Dwidth_nc[i_pat,i]=np.max(max_width_dip[mask_nc])
            L_nc[i_pat,i]=RCS.dipole_families['BNC']['length']
            L_sc[i_pat,i]=RCS.dipole_families['BSC']['length']
    plt.figure()
    plt.plot(B_nc_scan, Dwidth_nc[0] * 1e3, label='3 dip: Max width NC',color='tab:blue',linestyle='dashed')
    plt.plot(B_nc_scan, Dwidth_sc[0] * 1e3, label='3 dip: Max width SC',color='tab:orange',linestyle='dashed')
    plt.plot(B_nc_scan, Dwidth_nc[1] * 1e3, label='5 dip: Max width NC',color='tab:blue')
    plt.plot(B_nc_scan, Dwidth_sc[1] * 1e3, label='5 dip: Max width SC',color='tab:orange')
    plt.xlabel('$B_{NC} [T]$')
    plt.ylabel('Max width in dipole [mm]')
    plt.legend()
    plt.show()    
    plt.figure()
    plt.plot(B_nc_scan, Dpath[0]*1e3, label='3 dip',color='tab:green',linestyle='dashed')
    plt.plot(B_nc_scan, Dpath[1]*1e3, label='5 dip',color='tab:green')
    plt.xlabel('$B_{NC} [T]$')
    plt.ylabel('Max path length difference [mm]')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(B_nc_scan, L_nc[0], label='3 dip: Length NC dipole',color='tab:blue',linestyle='dashed')
    plt.plot(B_nc_scan, L_sc[0], label='3 dip: Length SC dipole',color='tab:orange',linestyle='dashed')
    plt.plot(B_nc_scan, L_nc[1], label='5 dip: Length NC dipole',color='tab:blue')
    plt.plot(B_nc_scan, L_sc[1], label='5 dip: Length SC dipole',color='tab:orange')
    plt.xlabel('$B_{NC} [T]$')
    plt.ylabel('Dipole length [m]')
    plt.legend()
    plt.show()

if scan_bnc_e==True:
    for i, (energy, B_nc) in enumerate(zip(E_ext_scan,B_nc_scan)):
        for i_pat,pattern in enumerate(pat):
            RCS=RingGeometry(file_input,pattern=pattern,extraction_energy=energy, dipoles={"BSC": 16., 
                                                        "BNC": (B_nc, "linear")},)
            mask_nc=[item == 'BNC' for item in RCS.pattern]
            mask_sc=[item == 'BSC' for item in RCS.pattern]
            print('energy',energy, RCS.extraction_energy)
            print('B_nc',B_nc, RCS.dipoles['BNC'])
            # RCS.plot_traj(t_traj)
            path = [RCS.path_length_tot(t) for t in np.linspace(0,1,11)]
            Dpath[i_pat,i]=np.max(path)-np.min(path)
            Dwidth_nc[i_pat,i]=RCS.arc_length
            max_width_dip=RCS.extrema(RCS.t_ext)[1]-RCS.extrema(RCS.t_inj)[0]
            Dwidth_sc[i_pat,i]=np.max(max_width_dip[mask_sc])
            Dwidth_nc[i_pat,i]=np.max(max_width_dip[mask_nc])
            L_nc[i_pat,i]=RCS.dipole_families['BNC']['length']
            L_sc[i_pat,i]=RCS.dipole_families['BSC']['length']
    B_nc_ticks=np.linspace(B_nc_min, B_nc_max, 7)
    E_ticks=(B_nc_ticks/B_nc_min*dE+E_inj)*1e-12
    fig, ax1 = plt.subplots()
    ax1.plot(B_nc_scan, Dwidth_nc[0] * 1e3, label='3 dip: Max width NC',color='tab:blue',linestyle='dashed')
    ax1.plot(B_nc_scan, Dwidth_sc[0] * 1e3, label='3 dip: Max width SC',color='tab:orange',linestyle='dashed')
    ax1.plot(B_nc_scan, Dwidth_nc[1] * 1e3, label='5 dip: Max width NC',color='tab:blue')
    ax1.plot(B_nc_scan, Dwidth_sc[1] * 1e3, label='5 dip: Max width SC',color='tab:orange')
    ax1.set_xlabel('$B_{NC}$ [T]')
    ax1.set_ylabel('Max width in dipole [mm]')
    ax1.legend()
    ax1.set_xticks(B_nc_ticks)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(B_nc_ticks)
    ax2.set_xticklabels(np.round(E_ticks, 2))
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 40))
    ax2.set_xlabel('$E_{ext}$ [TeV]')
    plt.show()
    fig, ax1 = plt.subplots()
    ax1.plot(B_nc_scan, Dpath[0]*1e3, label='3 dip',color='tab:green',linestyle='dashed')
    ax1.plot(B_nc_scan, Dpath[1]*1e3, label='5 dip',color='tab:green')
    ax1.set_xlabel('$B_{NC}$ [T]')
    ax1.set_ylabel('Max path length difference [mm]')
    ax1.set_xticks(B_nc_ticks)
    ax1.legend()
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(B_nc_ticks)
    ax2.set_xticklabels(np.round(E_ticks, 2))
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 40))
    ax2.set_xlabel('$E_{ext}$ [TeV]')
    plt.show()
    fig, ax1 = plt.subplots()
    ax1.plot(B_nc_scan, L_nc[0], label='3 dip: Length NC dipole',color='tab:blue',linestyle='dashed')
    ax1.plot(B_nc_scan, L_sc[0], label='3 dip: Length SC dipole',color='tab:orange',linestyle='dashed')
    ax1.plot(B_nc_scan, L_nc[1], label='5 dip: Length NC dipole',color='tab:blue')
    ax1.plot(B_nc_scan, L_sc[1], label='5 dip: Length SC dipole',color='tab:orange')
    ax1.set_xlabel('$B_{NC}$ [T]')
    ax1.set_ylabel('Dipole length [m]')
    ax1.legend()
    ax1.set_xticks(B_nc_ticks)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(B_nc_ticks)
    ax2.set_xticklabels(np.round(E_ticks, 2))
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 40))
    ax2.set_xlabel('$E_{ext}$ [TeV]')
    plt.show()

if scan_bnc_bsc==True:
    for i, (B_sc, B_nc) in enumerate(zip(B_sc_scan,B_nc_scan)):
        for i_pat,pattern in enumerate(pat):
            RCS=RingGeometry(file_input,pattern=pattern, dipoles={"BSC": B_sc, 
                                                        "BNC": (B_nc, "linear")},)
            mask_nc=[item == 'BNC' for item in RCS.pattern]
            mask_sc=[item == 'BSC' for item in RCS.pattern]
            print('energy', RCS.extraction_energy)
            print('B_nc',B_nc, RCS.dipoles['BNC'])
            print('B_sc',B_sc, RCS.dipoles['BSC'])
            # RCS.plot_traj(t_traj)
            Dpath[i_pat,i]=RCS.max_path_diff
            Dwidth_nc[i_pat,i]=RCS.arc_length
            max_width_dip=RCS.extrema(RCS.t_ext)[1]-RCS.extrema(RCS.t_inj)[0]
            Dwidth_sc[i_pat,i]=np.max(max_width_dip[mask_sc])
            Dwidth_nc[i_pat,i]=np.max(max_width_dip[mask_nc])
            L_nc[i_pat,i]=RCS.dipole_families['BNC']['length']
            L_sc[i_pat,i]=RCS.dipole_families['BSC']['length']
        
    B_nc_ticks=np.linspace(B_nc_min, B_nc_max, 7)
    B_sc_ticks=B_nc_ticks*aBrho/(L_tot*B_nc_ticks/np.pi-dBrho)
    fig, ax1 = plt.subplots()
    ax1.plot(B_nc_scan, Dwidth_nc[0] * 1e3, label='3 dip: Max width NC',color='tab:blue',linestyle='dashed')
    ax1.plot(B_nc_scan, Dwidth_sc[0] * 1e3, label='3 dip: Max width SC',color='tab:orange',linestyle='dashed')
    ax1.plot(B_nc_scan, Dwidth_nc[1] * 1e3, label='5 dip: Max width NC',color='tab:blue')
    ax1.plot(B_nc_scan, Dwidth_sc[1] * 1e3, label='5 dip: Max width SC',color='tab:orange')
    ax1.set_xlabel('$B_{NC} [T]$')
    ax1.set_ylabel('Max width in dipole [mm]')
    ax1.legend()
    ax1.set_xticks(B_nc_ticks)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(B_nc_ticks)
    ax2.set_xticklabels(np.round(B_sc_ticks, 2))
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 40))
    ax2.set_xlabel('$B_{SC} [T]$')
    plt.show()
    fig, ax1 = plt.subplots()
    ax1.plot(B_nc_scan, Dpath[0]*1e3, label='3 dip',color='tab:green',linestyle='dashed')
    ax1.plot(B_nc_scan, Dpath[1]*1e3, label='5 dip',color='tab:green')
    ax1.set_xlabel('$B_{NC} [T]$')
    ax1.set_ylabel('Max path length difference [mm]')
    ax1.set_xticks(B_nc_ticks)
    ax1.legend()
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(B_nc_ticks)
    ax2.set_xticklabels(np.round(B_sc_ticks, 2))
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 40))
    ax2.set_xlabel('$B_{SC} [T]$')
    plt.show()
    fig, ax1 = plt.subplots()
    ax1.plot(B_nc_scan, L_nc[0], label='3 dip: Length NC dipole',color='tab:blue',linestyle='dashed')
    ax1.plot(B_nc_scan, L_sc[0], label='3 dip: Length SC dipole',color='tab:orange',linestyle='dashed')
    ax1.plot(B_nc_scan, L_nc[1], label='5 dip: Length NC dipole',color='tab:blue')
    ax1.plot(B_nc_scan, L_sc[1], label='5 dip: Length SC dipole',color='tab:orange')
    ax1.set_xlabel('$B_{NC} [T]$')
    ax1.set_ylabel('Dipole length [m]')
    ax1.legend()
    ax1.set_xticks(B_nc_ticks)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(B_nc_ticks)
    ax2.set_xticklabels(np.round(B_sc_ticks, 2))
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 40))
    ax2.set_xlabel('$B_{SC} [T]$')
    plt.show()