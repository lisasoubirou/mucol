# -*- coding: utf-8 -*-
"""
Created on April 12
@author: LS276867
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.constants as sc
sys.path.append('/mnt/c/muco')

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

def track_cell(line, num_turns=1):
    """Tracking in the 1st cell of the typical line_ds
    Args:
        line (xt.line object): line containing 1 cell
        num_turns (int, optional): number of turns to track, if = 1 , tracking data is taken
        at each element of the cell.
    Returns:
        rec : line.record_last_track (contains the tracking data)
    """    
    line.build_tracker()
    part=line.build_particles()
    if num_turns==1:
        line.track(part,num_turns=1, turn_by_turn_monitor='ONE_TURN_EBE',ele_start='en1',ele_stop='ex3')
    else:
        line.track(part,num_turns=num_turns, turn_by_turn_monitor=True)
    rec=line.record_last_track
    return (rec)

def track(line, var, num_turns=1):
    """ Tracking on a line for a particle with IC x=px=y=py=delta=zeta=0
    Args:
        line (xt.line object): line to perform tracking on
        num_turns (int, optional): number of turns to track, if = 1 , tracking data is taken
        at each element of the line.
        var (bool) :  if True, variables allowed to be time dependent
    Returns:
        rec : line.record_last_track (contains the tracking data)
    """    
    line.build_tracker()
    part=line.build_particles()
    if var == True:
        line.enable_time_dependent_vars = True
    if num_turns==1:
        line.track(part,num_turns=1, turn_by_turn_monitor='ONE_TURN_EBE')
    else:
        line.track(part,num_turns=num_turns, turn_by_turn_monitor=True)
    rec=line.record_last_track
    part.sort(interleave_lost_particles=True)
    return (rec,part)


def distribution_lost_turn(particles0,particles):
    plt.figure(figsize=(12, 8))
    plt.scatter(particles0.x, particles0.y, c=particles.at_turn)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    cb = plt.colorbar()  
    cb.set_label('Lost at turn')
    plt.show()

def distribution_lost_turn_long(particles0,particles):
    plt.figure(figsize=(12, 8))
    plt.scatter(particles0.zeta, particles0.delta, c=particles.at_turn)
    plt.xlabel('zeta [m]')
    plt.ylabel('$\delta$')
    cb = plt.colorbar()  
    cb.set_label('Lost at turn')
    plt.show() 

def distribution_loss(particles0_u,particles0_v,particles,u,v):
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 2)
    scatter = ax1.scatter(particles0_u, particles0_v, c=particles.at_turn)
    ax1.set_xlabel(u)
    ax1.set_ylabel(v)
    cb = plt.colorbar(scatter, ax=ax1)  # Adjust the pad value as needed
    # cb.set_label('Lost at turn')
    ax2 = plt.subplot(2, 2, 4, sharex=ax1)
    ax2.hist(particles0_u[particles.state == -1], bins=20, color='blue', alpha=0.7, orientation='vertical')
    ax2.set_xlabel(u)
    ax2.set_ylabel('# lost')
    ax3 = plt.subplot(2, 2, 1, sharey=ax1)
    ax3.hist(particles0_v[particles.state == -1], bins=20, color='blue', alpha=0.7, orientation='horizontal')
    ax3.set_xlabel("# lost")
    ax3.set_ylabel(v)
    # plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.1, wspace=0.3, hspace=0.3)
    plt.show()

def distribution_loss_x_px(particles0,particles):
    distribution_loss(particles0.x,particles0.px,particles, 'x [m]',"x'")

def distribution_loss_y_py(particles0,particles):
    distribution_loss(particles0.y,particles0.py,particles,'y [m]',"y'")

def distribution_loss_x_y(particles0,particles):
    distribution_loss(particles0.x,particles0.y,particles,'x [m]','y [m]')

def compute_emit_x_n(line,particles):
    particles.hide_lost_particles()
    cov=np.cov(particles.x,particles.px)
    eps_x=np.sqrt(np.linalg.det(cov))
    particles.unhide_lost_particles()
    return(eps_x*particles._gamma0[0]*particles._beta0[0])

def compute_emit_y_n(line,particles):
    particles.hide_lost_particles()
    cov=np.cov(particles.y,particles.py)
    eps_y=np.sqrt(np.linalg.det(cov))
    particles.unhide_lost_particles()
    return(eps_y*particles._gamma0[0]*particles._beta0[0])

def compute_emit_s(line,particles):
    particles.hide_lost_particles()
    cov=np.cov(particles.zeta,particles.delta*particles.p0c)
    eps_s=np.sqrt(np.linalg.det(cov))/sc.c
    particles.unhide_lost_particles()
    return(eps_s)

def compute_sigma_z(line,particles):
    particles.hide_lost_particles()
    sigma_z=np.std(particles.zeta)
    particles.unhide_lost_particles()
    return(sigma_z)

def compute_x_mean(line,particles):
    particles.hide_lost_particles()
    x_ave = np.mean(particles.x)
    particles.unhide_lost_particles()
    return x_ave

def compute_y_mean(line,particles):
    particles.hide_lost_particles()
    y_ave = np.mean(particles.y)
    particles.unhide_lost_particles()
    return y_ave

def compute_betx(line,particles):
    # particles.hide_lost_particles()
    cov=np.cov(particles.x,particles.px)
    eps_x=np.sqrt(np.linalg.det(cov))
    beta_x = cov[0, 0] / eps_x
    # particles.unhide_lost_particles()
    return beta_x

def compute_bety(line,particles):
    # particles.hide_lost_particles()
    cov=np.cov(particles.y,particles.py)
    eps_y=np.sqrt(np.linalg.det(cov))
    beta_y = cov[0, 0] / eps_y
    # particles.unhide_lost_particles()
    return beta_y

def calculate_transmission(turns, total_particles, n_turn):
    transmissions = []
    lost_particles = 0
    for turn in range(1, n_turn + 1):
        if turn in turns:
            lost_particles += turns.count(turn)
        transmission = (total_particles - lost_particles) / total_particles
        transmissions.append(transmission)
    return transmissions

def compute_action_list(u, v):
    cov=np.cov(u, v)
    eps=np.sqrt(np.linalg.det(cov))

    bet = cov[0, 0] / eps       # Twiss functions from beam parameters
    alf = - cov[0, 1] / eps
    gam = cov[1, 1] / eps
    
    return 0.5 * (gam * u**2 + 2 * alf * u * v + bet * v**2)

def compute_action_perc(u, v, n_perc):
    action_list = compute_action_list(u ,v)
    action = np.percentile(action_list, n_perc) # 95th percentile 
    mask_action = action_list < action

    u_mask, v_mask= u[mask_action], v[mask_action]
    action_list_mask = compute_action_list(u_mask, v_mask)
    action = np.percentile(action_list_mask, n_perc)
    
    return action


def emit_x_per(mon):
    n_perc = 95
    emit = []

    for i in range(len(mon.x[0, :])): 
        mask_lost = mon.state[:, i] == 1  # Filter out lost particles

        # Apply the mask to x, px, gamma, and beta
        x = mon.x[:, i][mask_lost]
        px = mon.px[:, i][mask_lost]
        gamma = np.mean(mon.gamma0[:, i][mask_lost])  # Mean gamma over non-lost particles
        beta = np.mean(mon.beta0[:, i][mask_lost])    # Mean beta over non-lost particles
        
        action = compute_action_perc(x, px, n_perc)
        emit.append(action / 2 * gamma * beta) # Normalized emittance

    return np.array(emit)

def emit_y_per(mon):
    n_perc = 95
    emit = []

    for i in range(len(mon.y[0, :])): 
        mask_lost = mon.state[:, i] == 1  # Filter out lost particles
        
        # Apply the mask to y, py, gamma, and beta
        y = mon.y[:, i][mask_lost]
        py = mon.py[:, i][mask_lost]
        gamma = np.mean(mon.gamma0[:, i][mask_lost])  # Mean gamma over non-lost particles
        beta = np.mean(mon.beta0[:, i][mask_lost])    # Mean beta over non-lost particles

        action = compute_action_perc(y, py, n_perc)
        emit.append(action / 2 * gamma * beta)  # Normalized emittance
    
    return np.array(emit)

def emit_s_per(mon):
    n_perc = 95
    emit = []

    for i in range(len(mon.y[0, :])): 
        mask_lost = mon.state[:, i] == 1  # Filter out lost particles

        zeta = mon.zeta[:, i][mask_lost]  # Time from ref. particle 
        delta = mon.delta[:, i][mask_lost]
        p0c = np.mean(mon.p0c[:, i][mask_lost])

        action = compute_action_perc(zeta, delta, n_perc)
        emit.append(action/2*p0c/sc.c)  # Normalized emittance
    
    return np.array(emit)

def x_mean_mon(mon):
    x_mean = []
    for i in range(len(mon.x[0, :])):
        mask_lost = mon.state[:, i] == 1 # Filter out lost particles
        x_filtered = mon.x[:, i][mask_lost]
        x_mean.append(np.mean(x_filtered))
    return np.array(x_mean)

def y_mean_mon(mon):
    y_mean = []
    for i in range(len(mon.y[0, :])):
        mask_lost = mon.state[:, i] == 1 # Filter out lost particles
        y_filtered = mon.y[:, i][mask_lost]
        y_mean.append(np.mean(y_filtered))
    return np.array(y_mean)

def sigma_z_mon(mon):
    sig_z = []
    for i in range(len(mon.zeta[0, :])):
        mask_lost = mon.state[:, i] == 1 # Filter out lost particles
        z_filtered = mon.zeta[:, i][mask_lost]
        sig_z.append(np.std(z_filtered))
    return np.array(sig_z)

def emit_x_mon(mon):
    emit = []
    for i in range( len(mon.x[0, :]) ): 
        mask_lost = mon.state[:, i] == 1  # Mask to filter out lost particles

        x = mon.x[:, i][mask_lost]
        px = mon.px[:, i][mask_lost]
        gamma = np.mean(mon.gamma0[:, i][mask_lost])  # Mean gamma 
        beta = np.mean(mon.beta0[:, i][mask_lost])    # Mean beta
        
        cov = np.cov(x,px) # Covariant matrix 
        eps_x = np.sqrt(np.linalg.det(cov))
        emit.append(eps_x*gamma*beta)
    return emit

def emit_y_mon(mon):
    emit = []
    for i in range( len(mon.y[0, :]) ): 
        mask_lost = mon.state[:, i] == 1  # Mask to filter out lost particles

        y = mon.y[:, i][mask_lost]
        py = mon.py[:, i][mask_lost]
        gamma = np.mean(mon.gamma0[:, i][mask_lost])  # Mean gamma 
        beta = np.mean(mon.beta0[:, i][mask_lost])    # Mean beta
        
        cov = np.cov(y,py) # Covariant matrix 
        eps_y = np.sqrt(np.linalg.det(cov))
        emit.append(eps_y*gamma*beta)
    return emit

def emit_s_mon(mon):
    emit = []
    for i in range( len(mon.zeta[0, :]) ): 
        mask_lost = mon.state[:, i] == 1  # Mask to filter out lost particles
        zeta = mon.zeta[:, i][mask_lost]
        delta = mon.delta[:, i][mask_lost]
        p0c = np.mean(mon.p0c[:, i][mask_lost]) # Mean energy

        cov=np.cov(zeta,delta*p0c) # Covariant matrix
        eps_s=np.sqrt(np.linalg.det(cov))/sc.c
    
        emit.append(eps_s)
    return emit