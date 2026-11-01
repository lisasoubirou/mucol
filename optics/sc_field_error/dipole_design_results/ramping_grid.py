# -*- coding: utf-8 -*-
"""
Created on March 28

@author: LS276867
"""

import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xpart as xp
import sys
sys.path.append('/mnt/c/muco')
from class_geometry.ring_geometry.ring_geometry import RingGeometry
from interpolation import calc_dip_coef,eval_horner
from track_function import compute_emit_x_n, compute_emit_y_n, compute_emit_s, compute_sigma_z, compute_x_mean, compute_y_mean,calculate_transmission
import json

from scipy.interpolate import CubicSpline

plt.rc('axes', labelsize=13)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize

#Constants
muon_mass_kg=sc.physical_constants['muon mass'][0]
e_charge=sc.e
c=sc.c

#Importing line
method='var_k'
n_slice=21
t_ref=0.5
t_acc=1.09704595*1e-3
n_part=5000

#Parameters
file_input='/mnt/c/muco/code/class_geometry/para_RCS_ME.txt'
RCS = RingGeometry(file_input)
C=RCS.C
nb_arc=RCS.nb_arc
n_cav_arc = 2 
E_inj=RCS.E_inj
E_ext=RCS.E_ext
frequency_rf=RCS.RF_freq
N_turn_rcs=55
n_turns = N_turn_rcs*nb_arc
energy_increment_per_arc = (E_ext-E_inj)/n_turns
sync_phase=45
phase=180-sync_phase
volt_arc=energy_increment_per_arc/np.sin(phase*np.pi/180)
volt_cav=volt_arc/n_cav_arc
gamma_0=RCS.inj_gamma
beta_0=np.sqrt(1-1/gamma_0**2)

file_seq = '/mnt/c/muco/code/lattice_disp_suppr_6d_dq_5.json'

#Create grid
n_grid=10
b3=np.linspace(-10,3,n_grid)
b5=np.linspace(-7,0,n_grid)
bb3, bb5 = np.meshgrid(b3, b5)
bb3=np.ravel(bb3)
bb5=np.ravel(bb5)

list_tr=[]
list_emx=[]
list_emy=[]
list_ems=[]

for para3, para5 in zip(bb3,bb5):
    line=xt.Line.from_json(file_seq)

    line.particle_ref = xt.Particles(mass0=xp.MUON_MASS_EV, q0=1.,
                                        energy0=RCS.E_inj,
                                    )

    arr_delta = np.linspace(-0.02, 0.02, 41)
    arr_twiss = np.zeros((41, 4))
    for i, delta in enumerate(arr_delta):
        tw_6d = line.twiss(method='6d', matrix_stability_tol=5e-3, delta0=delta)
        arr_twiss[i, 0] = tw_6d.betx[0]
        arr_twiss[i, 1] = tw_6d.alfx[0]
        arr_twiss[i, 2] = tw_6d.bety[0]
        arr_twiss[i, 3] = tw_6d.alfy[0]

    twiss_delta = CubicSpline(arr_delta, arr_twiss, bc_type='not-a-knot')

    tw_6d=line.twiss(method='6d', matrix_stability_tol=5e-3)

    tw_6d=line.twiss(method='6d', matrix_stability_tol=5e-3, 
                    compute_chromatic_properties=True)
    h_rf = np.round(frequency_rf*tw_6d.T_rev0*nb_arc)
    MCF=tw_6d['momentum_compaction_factor']

    tab=line.get_table()
    line.discard_tracker()
    pattern=RCS.pattern 
    i_BNC=pattern.index('BNC')
    i_BSC=pattern.index('BSC')

    line._init_var_management()
    line.vars['t_turn_s']=0.

    coef=calc_dip_coef(file_input,t_ref)
    line.functions.h_nc_pol= lambda x: eval_horner(coef[0],x)
    line.functions.h_sc_pol= lambda x: eval_horner(coef[1],x)
    line.functions.l_nc_pol= lambda x: eval_horner(coef[2],x)
    line.functions.l_sc_pol= lambda x: eval_horner(coef[3],x)
    line.functions.path_diff_1cav= lambda x: -eval_horner(coef[4],x)/2

    line.vars['h_NC']=line.functions.h_nc_pol(line.vars['t_turn_s']/t_acc)
    line.vars['h_SC']=line.functions.h_sc_pol(line.vars['t_turn_s']/t_acc)
    line.vars['l_dip_NC']=line.functions.l_nc_pol(line.vars['t_turn_s']/t_acc)
    line.vars['l_dip_SC']=line.functions.l_sc_pol(line.vars['t_turn_s']/t_acc)
    line.vars['dz_rf']=line.functions.path_diff_1cav(line.vars['t_turn_s']/t_acc)

    line.element_refs['dz_acc'].dzeta=line.vars['dz_rf']
    line.element_refs['dz_acc_1'].dzeta=line.vars['dz_rf']

    h_ref_NC=RCS.hn(t_ref)[i_BNC]
    h_ref_SC=RCS.hn(t_ref)[i_BSC]
    l_dip_ref_NC=RCS.L_dip_path(t_ref)[i_BNC]
    l_dip_ref_SC=RCS.L_dip_path(t_ref)[i_BSC]

    #Errors dipole
    b3=para3
    b5=para5
    r_rad=0.01

    # Time dependent knobs on dipole parameters
    for el in tab.rows[tab.element_type == 'Bend'].name:
        if 'BSC' in el:
            line.element_refs[el].knl[2]=2*10e-4*b3*r_rad**(-2)*line.vars['h_SC']
            line.element_refs[el].knl[4]=24*10e-4*b5*r_rad**(-4)*line.vars['h_SC']
        if method == 'var_ref':
            if 'BSC' in el:
                line.element_refs[el].k0=line.vars['h_SC']
                line.element_refs[el].h=line.vars['h_SC']
                line.element_refs[el].length=line.vars['l_dip_SC']
            elif 'BNC' in el:
                line.element_refs[el].k0=line.vars['h_NC']
                line.element_refs[el].h=line.vars['h_NC']
                line.element_refs[el].length=line.vars['l_dip_NC']
        elif method == 'var_k':
            if 'BSC' in el:  
                line.element_refs[el].k0=line.vars['h_SC']
                line.element_refs[el].h=h_ref_SC
                line.element_refs[el].length=l_dip_ref_SC
            elif 'BNC' in el:
                line.element_refs[el].k0=line.vars['h_NC']
                line.element_refs[el].h=h_ref_NC
                line.element_refs[el].length=l_dip_ref_NC
        else:
            raise ValueError("Invalid option: {}".format(method))
    #Time dependent knobs on dipole edge strength    
    for el in tab.rows[tab.element_type == 'DipoleEdge'].name:
        if 'en1' or 'ex1' or 'en3' or 'ex3' in el:
                line.element_refs[el].k=line.vars['h_SC']
        elif 'en2' or 'ex2' in el:
                line.element_refs[el].k=line.vars['h_NC']

    line.vars['t_turn_s']=0.5*t_acc
    f=RCS.cell_length/(4*np.sin(np.pi/2/2)) #Focusing strength of quadrupole
    line.vars['s_d'] = -1/(f*0.5)*8
    line.vars['s_f'] = 1/(f*0.25)*8
    line.element_refs['sxt_d_1'].knl[2] = line.vars['s_d']
    line.element_refs['sxt_f_1'].knl[2] = line.vars['s_f']
    line.element_refs['sxt_d_2'].knl[2] = line.vars['s_d']
    line.element_refs['sxt_f_2'].knl[2] = line.vars['s_f']
    dqx_goal_arc=5/nb_arc
    dqy_goal_arc=5/nb_arc
    match_ds_6d_sxt=line.match(vary=xt.VaryList(['s_d','s_f'],
                                step=1e-5,
                                tag='sxt'),
                    targets=[
                            xt.TargetSet(dqx=dqx_goal_arc, dqy=dqy_goal_arc, tol=1e-6, tag='chroma')],
                    solve=False,
                    method='6d',
                    matrix_stability_tol=5e-3,
                    # verbose=True
                    )
    match_ds_6d_sxt.step(20)
    print('RESULTS SXT DS MATCH 6D')
    match_ds_6d_sxt.target_status()
    match_ds_6d_sxt.vary_status()
    line.vars['t_turn_s']=0.
    line.discard_tracker()
    line.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(n_slice))])
    print('slicing done')
    tab_sliced=line.get_table()

    #Tracking parameters
    nemitt_x_0 = 25e-6
    nemitt_y_0 = 25e-6
    emit_s=0.025 #eVs

    print('go to tracker')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))
    # line.build_tracker()
    print('tracker built')
    particles = xp.generate_matched_gaussian_bunch(line=line,
                                                matrix_stability_tol=5e-3,
                                                num_particles=n_part,
                                                nemitt_x=nemitt_x_0, 
                                                nemitt_y=nemitt_y_0, 
                                                sigma_z=0,
                                                )

    rfbucket = xp.longitudinal.rf_bucket.RFBucket(
                                circumference=C/nb_arc,
                                gamma=gamma_0,
                                mass_kg=muon_mass_kg,
                                charge_coulomb=e_charge,
                                alpha_array=np.atleast_1d(MCF),
                                harmonic_list=np.atleast_1d(h_rf/nb_arc),
                                voltage_list=np.atleast_1d(volt_arc), #*nb_arc
                                phi_offset_list=np.atleast_1d((sync_phase)*np.pi/180),
                                p_increment=energy_increment_per_arc * e_charge / c) #*nb_arc

    matcher = xp.longitudinal.rfbucket_matching.RFBucketMatcher(rfbucket=rfbucket,
                            distribution_type=xp.longitudinal.rfbucket_matching.ThermalDistribution,
                            #   sigma_z=sigma_z,
                            epsn_z=4*np.pi*emit_s
                            )

    z_particles, delta_particles, _, _ = matcher.generate(macroparticlenumber=n_part)
    particles.zeta=z_particles
    particles.delta=delta_particles
    tw_loc = twiss_delta(particles.delta)
    Ax = np.sqrt(tw_loc[:, 0]/tw_6d.betx[0])
    Bx = (tw_6d.alfx[0]-tw_loc[:,1])/np.sqrt(tw_6d.betx[0]*tw_loc[:,0])
    Ay = np.sqrt(tw_loc[:, 2]/tw_6d.bety[0])
    By = (tw_6d.alfy[0]-tw_loc[:,3])/np.sqrt(tw_6d.bety[0]*tw_loc[:,2])
    particles.px = -Bx*particles.x + particles.px/Ax + tw_6d.ddpx[0]*particles.delta**2
    particles.x = Ax*particles.x + tw_6d.ddx[0]*particles.delta**2
    particles.py = -By*particles.y + particles.py/Ay + tw_6d.ddpy[0]*particles.delta**2
    particles.y = Ay*particles.y + tw_6d.ddy[0]*particles.delta**2
    particles0=particles.copy()
    print('Particles ok')

    track_log = xt.Log(x_mean=compute_x_mean,y_mean=compute_y_mean, 
                    eps_x=compute_emit_x_n, eps_y=compute_emit_y_n,
                    eps_s=compute_emit_s, sig_z=compute_sigma_z)
    line.enable_time_dependent_vars = True
    line.track(particles, log=track_log, num_turns=n_turns, 
            with_progress=True,turn_by_turn_monitor=True)
    rec=line.record_last_track
    particles.sort(interleave_lost_particles=True)
    x_mean = line.log_last_track['x_mean']
    y_mean = line.log_last_track['y_mean']
    eps_x=line.log_last_track['eps_x']
    var_eps_x=(eps_x - eps_x[0]) / eps_x[0]
    eps_y=line.log_last_track['eps_y']
    var_eps_y=(eps_y - eps_y[0]) / eps_y[0]
    eps_s=line.log_last_track['eps_s']
    var_eps_s=(eps_s - eps_s[0]) / eps_s[0]
    sig_z=line.log_last_track['sig_z']
    lost_part=particles.at_turn[particles.state<0]
    transmission=calculate_transmission(sorted(lost_part),n_part,n_turns)
    list_emx.append(np.mean(var_eps_x[-200:]))
    list_emy.append(np.mean(var_eps_y[-200:]))
    list_ems.append(np.mean(var_eps_s[-200:]))
    list_tr.append(transmission[-1])

data_grid = {
    "bb3": bb3.tolist(),
    "bb5": bb5.tolist(),
    "list_emx": list_emx,
    "list_emy": list_emy,
    "list_ems": list_ems,
    "list_tr": list_tr
}

file_path='grid_100_scan_5000p_chromacorrect.json'
try:
    with open(file_path, "x") as json_file:
        json.dump(data_grid, json_file)
    print("Lists saved to", file_path)
except FileExistsError:
    print("File", file_path, "already exists. Cannot overwrite.")

geo_coils={ 'Min V(2RT)': [-9.8,-6.3],
            'Med V(2RT)': [2.5, -3.0],
            'Best B(2RT)': [0.3, -0.8],
            'Min V(AD)': [-1.6,-6.5],
            'Med V(AD)': [-1.8, -2.5],
            'Best B(AD)': [0.3, -0.7]
}

def add_points(color):
    for name, (x, y) in geo_coils.items():
        if name=='Best B (2RT)':
            plt.scatter(x, y, color='red', marker='o')
            plt.text(x, y-0.2, name, fontsize=12,color=color)
        elif name=='Med V (2RT)':
            plt.scatter(x, y, color='red', marker='o')
            plt.text(x-1.5, y+0.1, name, fontsize=12,color=color)
        else:
            plt.scatter(x, y, color='red', marker='o')
            plt.text(x, y+0.1, name, fontsize=12,color=color)


plt.figure(figsize=(12, 8))
plt.scatter(bb3, bb5, c=list_tr, s=100)
plt.xlabel('b3 [units]')
plt.ylabel('b5 [units]')
cb = plt.colorbar()  
cb.set_label('Transmission')
plt.show() 

bb3_mesh, bb5_mesh = np.meshgrid(b3, b5)
levels=[0,0.20,0.40,0.60,0.80,0.95,0.9998,1]
list_tr_resh=np.array(list_tr).reshape(10,10)
plt.figure(figsize=(12, 8))
plt.xlabel('b3 [units]')
plt.ylabel('b5 [units]')
# Plot the filled contour using meshgrid and the data
contourf = plt.contourf(bb3_mesh, bb5_mesh, list_tr_resh, levels=levels, cmap='viridis')
# Add contour lines for better visualization
plt.contour(bb3_mesh, bb5_mesh, list_tr_resh, levels=levels, colors='k')
cb = plt.colorbar(contourf)
cb.set_ticks(levels)
cb.set_label('Transmission')
add_points('k')
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(bb3, bb5, c=list_emx, s=100)
plt.xlabel('b3 [units]')
plt.ylabel('b5 [units]')
cb = plt.colorbar()  
cb.set_label(r'$\Delta \epsilon_x / \epsilon_{x,0}$')
plt.show() 

levels=[-0.025,0,0.025,0.05,0.10]
list_emx_resh=np.array(list_emx).reshape(10,10)
plt.figure(figsize=(12, 8))
plt.xlabel('b3 [units]')
plt.ylabel('b5 [units]')
# Plot the filled contour using meshgrid and the data
contourf = plt.contourf(bb3_mesh, bb5_mesh, list_emx_resh, levels=levels, cmap='viridis',extend='both')
# Add contour lines for better visualization
plt.contour(bb3_mesh, bb5_mesh, list_emx_resh, levels=levels, colors='k')
cb = plt.colorbar(contourf)
cb.set_ticks(levels)
cb.set_label(r'$\Delta \epsilon_x / \epsilon_{x,0}$')
add_points('white')
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(bb3, bb5, c=list_emy, s=100)
plt.xlabel('b3 [units]')
plt.ylabel('b5 [units]')
cb = plt.colorbar()  
cb.set_label(r'$\Delta \epsilon_y / \epsilon_{y,0}$')
plt.show() 

levels=[0,0.025,0.05,0.10]
list_emy_resh=np.array(list_emy).reshape(10,10)
plt.figure(figsize=(12, 8))
plt.xlabel('b3 [units]')
plt.ylabel('b5 [units]')
# Plot the filled contour using meshgrid and the data
contourf = plt.contourf(bb3_mesh, bb5_mesh, list_emy_resh, levels=levels, cmap='viridis',extend='max')
# Add contour lines for better visualization
plt.contour(bb3_mesh, bb5_mesh, list_emy_resh, levels=levels, colors='k')
cb = plt.colorbar(contourf)
cb.set_ticks(levels)
cb.set_label(r'$\Delta \epsilon_y / \epsilon_{y,0}$')
add_points('white')
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(bb3, bb5, c=list_ems, s=100)
plt.xlabel('b3 [units]')
plt.ylabel('b5 [units]')
cb = plt.colorbar()  
cb.set_label(r'$\Delta \epsilon_s / \epsilon_{s,0}$')
plt.show() 


