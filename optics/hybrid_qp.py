# -*- coding: utf-8 -*-
"""
Created on Dec 2024

@author: Lisa Soubirou
"""

import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import xpart as xp
import json
import copy
from class_geometry.ring_geometry.ring_geometry import RingGeometry, load_file
from pathlib import Path

plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

T_REF = 0.5
MU = np.pi/2
LC = 50
N_SIGMA = 3
TR_EMIT_N = 25e-6
DX = 10e-3
L_QP_MIN_SC = 0.75
L_QP_MIN_NC = 0.75
ALPHA = 7/1
MIN_DIP_LENGTH = {'BSC': 2, 'BNC': 2}
N_START = 50
N_CELLS_PLOT = 50

def length_qp_hybrid (rcs : RingGeometry, sigma_delta: float, n_cell = None, 
                     L_qp_min_nc : float = L_QP_MIN_NC, L_qp_min_sc : float = L_QP_MIN_SC, 
                     L_dip_min : dict = MIN_DIP_LENGTH, alpha : float = ALPHA, mu : float = MU):
    """Computes the quadrupole parameter for a hybrid quadrupole cell. 

    Args:
        rcs (RingGeometry): RCS object defining the geometry. 
        n_cell (_type_, optional): Total number of cells in a RCS. Defaults to None.
        L_qp_min_nc (float, optional): Minimal length for NC quadrupoles [m]. Defaults to L_QP_MIN_NC.
        L_qp_min_sc (float, optional): Minimal length for SC quadrupoles [m]. Defaults to L_QP_MIN_SC.
        L_dip_min (dict, optional): Minimal lengths for SC and NC dipoles [m]. Defaults to MIN_DIP_LENGTH.
        alpha (float, optional): SC/NC Technological ratio. Defaults to ALPHA.
        mu (float, optional): Cell phase advance [rad]. Defaults to MU.

    Returns:
        n_cell : number of cell of study
        cell_length : cell length of study [m]
        qp_para : qp parameters (length [m], strength [1/m²], gradient[T/m])
        r_qp : radial aperture of QP [m]
    """
    mid_Brho = (rcs.ext_Brho + rcs.inj_Brho) / 2
    r_sum_diff = (rcs.ext_Brho + rcs.inj_Brho) / (rcs.ext_Brho - rcs.inj_Brho)
    rcs = copy.deepcopy(rcs)
    rcs.length_short_insertion = L_QP_MIN_SC + L_QP_MIN_NC + rcs.dipole_spacing

    if n_cell == None:
        print('Limit nc max is ', rcs.nc_best*rcs.nb_arc)

        # Determine maximum number of cells per arc from minimum QP and dipole lengths
        nc_max_sc = int(rcs.tot_arc_length-rcs.dipole_length_tot)/2/(L_qp_min_sc*(1+r_sum_diff/alpha)*alpha/r_sum_diff+(len(rcs.pattern)+2)*rcs.dipole_spacing)
        nc_max_nc = int(rcs.tot_arc_length-rcs.dipole_length_tot)/2/(L_qp_min_nc*(1+r_sum_diff/alpha)+(len(rcs.pattern)+2)*rcs.dipole_spacing)
        print(f'Limit nc on QP SC length: {nc_max_sc}')
        print(f'Limit nc on QP NC length: {nc_max_nc}')

        nc_max_dip = []
        type_dip, counts = np.unique(np.array(rcs.pattern), return_counts=True)
        for dip, count in zip(type_dip, counts):
            nc_max_item = int(rcs.dipole_families[dip]['tot_length'] / (2*count*L_dip_min[dip]))
            nc_max_dip.append(nc_max_item)
            print(f'Limit nc on dip {dip}: {nc_max_item}')
        nc_max = min(nc_max_sc, nc_max_nc, min(nc_max_dip))

        if nc_max < N_START:
            raise ValueError('Need to increase N_START. Max number of cells < N_START')
        
        # Number of cells possible and associated cell length
        n_cell = np.arange(N_START, int(nc_max) + 1)

        rcs.nb_arc = 1
        rcs.nb_cell_arc = int(nc_max)

    elif isinstance(n_cell, (int, float, np.integer, np.floating)):
        # Convert single float or int to an array
        n_cell = np.array([n_cell])
    
    cell_length = rcs.tot_arc_length/n_cell
    # Compute QP length, gradient, strength and aperture
    l_qp_hyb = (rcs.tot_arc_length-rcs.dipole_length_tot)/2/n_cell - (len(rcs.pattern)+2)*rcs.dipole_spacing - 1e-5

    l_nc = l_qp_hyb / ( 1 + r_sum_diff/alpha)
    l_sc = l_qp_hyb * r_sum_diff/alpha / (1 + r_sum_diff/alpha)
    gradient_sc = 2 * np.sin(mu/2) * (rcs.ext_Brho + rcs.inj_Brho)/ cell_length / l_sc
    gradient_nc_ext = 2 * np.sin(mu/2) * (rcs.ext_Brho - rcs.inj_Brho)/ cell_length / l_nc
    k_sc_const = gradient_sc / rcs.ext_Brho
    k_nc_ext = gradient_nc_ext / rcs.ext_Brho

    qp_para = {
        'qp_nc' : { 
            'length' : l_nc, 
            'k' : k_nc_ext, 
            'gradient' : gradient_nc_ext},
        'qp_sc' : {
            'length' : l_sc, 
            'k' : k_sc_const, 
            'gradient' : gradient_sc},
        'dx' : {}
               }

    r_qp, disp_max = radius_qp_optics(
        rcs = rcs, 
        n_cell = n_cell, 
        type_qp = 'hybrid', 
        qp_para = qp_para, 
        sigma_delta = sigma_delta, 
        slice=False)
    
    qp_para['dx']['dx'] = disp_max
    
    # r_qp = radius_qp(rcs= rcs, cell_length = cell_length, mu=mu, sigma_delta=sigma_delta)
    
    if n_cell.size == 1:
        return n_cell[0], cell_length[0], {key: {subkey: value[0] for subkey, value in subdict.items()} for key, subdict in qp_para.items()}, r_qp[0]
    else:
        return n_cell, cell_length, qp_para, r_qp


def length_qp_normal(rcs : RingGeometry, sigma_delta: float, slice, n_cell = None, 
                     L_qp_min : float = L_QP_MIN_NC, L_dip_min: float = MIN_DIP_LENGTH, 
                     mu : float = MU):
    """ Computes the quadrupole parameter for a normal quadrupole cell. 

    Args:
        rcs (RingGeometry): RCS object defining the geometry. 
        n_cell (_type_, optional): Total number of cells in a RCS. Defaults to None.
        L_qp_min (float, optional): Minimal length for quadrupoles [m]. Defaults to L_QP_MIN_NC.
        L_dip_min (float, optional): Minimal lengths for SC and NC dipoles [m]. Defaults to MIN_DIP_LENGTH.
        mu (float, optional): Cell phase advance [rad]. Defaults to MU.

    Returns:
        n_cell : number of cell of study
        cell_length : cell length of study [m]
        qp_para : qp parameters (length [m], strength [1/m²], gradient[T/m])
        r_qp : radial aperture of QP [m]
    """
    rcs = copy.deepcopy(rcs)
    rcs.length_short_insertion = L_QP_MIN_NC

    if n_cell is None:
        nc_max_qp = int(rcs.tot_arc_length-rcs.dipole_length_tot)/2/(L_qp_min+(len(rcs.pattern)+1)*rcs.dipole_spacing)
        print(f'Limit nc on QP length: {nc_max_qp}')
        nc_max_dip = []
        type_dip, counts = np.unique(np.array(rcs.pattern), return_counts=True)
        for dip, count in zip(type_dip, counts):
            nc_max_item = int(rcs.dipole_families[dip]['tot_length'] / (2*count*L_dip_min[dip]))
            nc_max_dip.append(nc_max_item)
            print(f'Limit nc on dip {dip}: {nc_max_item}')
        nc_max = min(nc_max_qp, min(nc_max_dip))

        if nc_max < N_START:
            raise ValueError('Need to increase N_START. Max number of cells < N_START')
        
        n_cell = np.arange(N_START, int(nc_max) + 1)

        rcs.nb_arc = 1
        rcs.nb_cell_arc = int(nc_max)

    elif isinstance(n_cell, (int, float, np.integer, np.floating)):
        # Convert single float or int to an array
        n_cell = np.array([n_cell])

    cell_length = rcs.tot_arc_length/n_cell
    print('dip tot length', rcs.dipole_length_tot)
    l_qp = (rcs.tot_arc_length-rcs.dipole_length_tot)/2/n_cell - (len(rcs.pattern)+1)*rcs.dipole_spacing - 1e-5
    gradient_nc_ext = 4 * np.sin(mu/2) * rcs.ext_Brho / cell_length / l_qp
    k_nc_ext = gradient_nc_ext / rcs.ext_Brho
    qp_para = {
        'qp_nc' : {
            'length' : l_qp, 
            'k' : k_nc_ext, 
            'gradient' : gradient_nc_ext
            },
        'dx' : {}
        }
    r_qp, disp_max = radius_qp_optics(rcs = rcs, n_cell = n_cell, type_qp = 'normal', qp_para = qp_para, sigma_delta = sigma_delta, slice=slice)
    qp_para['dx']['dx'] = disp_max

    # r_qp = radius_qp(rcs = rcs, cell_length = cell_length, mu = mu, sigma_delta = sigma_delta)

    if n_cell.size == 1:
        print (n_cell, cell_length, qp_para, r_qp)
        return n_cell[0], cell_length[0], {key: {subkey: value[0] for subkey, value in subdict.items()} for key, subdict in qp_para.items()}, r_qp[0]
    else:
        return n_cell, cell_length, qp_para, r_qp


def radius_qp (rcs : RingGeometry, cell_length : float, mu : float, sigma_delta : float, 
               n_sigma : int = N_SIGMA, tr_emit_n : float = TR_EMIT_N, 
              dx : float = DX ) -> float :
    """Computes radial aperture of QP. 

    Args:
        rcs (RingGeometry): RCS object defining the geometry.
        cell_length (float): Cell length [m]
        mu (float): Phase advance [rad]
        sigma_delta (float): Max. energy dispersion of the beam
        n_sigma (int, optional): Number of sigma to compute beam size. Defaults to N_SIGMA.
        tr_emit_n (float, optional): Transverse normalized emittance at injection [m.rad]. Defaults to TR_EMIT_N.
        dx (float, optional): Radial aperture margin [m]. Defaults to DX.

    Returns:
        float: Radial aperture of QP : beam size + margin [m]. 
    """
    
    beta_max = cell_length * (1 + np.sin(mu/2)) / np.sin(mu)
    disp_max = 2*np.pi*cell_length**2 / rcs.tot_arc_length*(
        1 + np.sin(mu/2)) / 2 / (4*(np.sin(mu/2))**2)
    beam_size_radius = n_sigma * np.sqrt(beta_max*tr_emit_n/rcs.inj_gamma
                                )+ sigma_delta*disp_max
    
    return beam_size_radius + dx


def radius_qp_optics(
        rcs : RingGeometry, 
        n_cell : int, 
        type_qp : str, 
        qp_para : dict, 
        sigma_delta : float, 
        slice : int,
        n_sigma : int = N_SIGMA, 
        tr_emit_n : float = TR_EMIT_N, 
        dx : float = DX) -> np.array :
    """Computes the radial aperture of QP with the optics of the cell. 

    Args:
        rcs (RingGeometry): RCS object defining the geometry.
        n_cell (int): Total number of cells per RCS. 
        type_qp (str): Either 'hybrid' or 'normal' quadrupole. 
        qp_para (dict): qp parameters (length [m], strength [1/m²], gradient[T/m]).
        sigma_delta (float): Max. energy dispersion of the beam.
        n_sigma (int, optional): Number of sigma to compute beam size. Defaults to N_SIGMA.
        tr_emit_n (float, optional): Transverse normalized emittance at injection [m.rad]. Defaults to TR_EMIT_N.
        dx (float, optional): Radial aperture margin [m]. Defaults to DX.

    Returns:
         np.array : Radial aperture of QP : beam size + margin [m]. 
    """

    beam_radius = []
    beta = []
    disp = []

    rcs.nb_arc = 1
    for nc_idx, nc in enumerate(n_cell):
        rcs.nb_cell_arc = nc

        l_nc = qp_para['qp_nc']['length'][nc_idx]
        k_nc = qp_para['qp_nc']['k'][nc_idx]

        if type_qp == 'hybrid' :
            l_sc = qp_para['qp_sc']['length'][nc_idx]
            k_sc = qp_para['qp_sc']['k'][nc_idx]
            line_fodo = cell_qp_hybrid(rcs, l_sc=l_sc, k_sc=k_sc, l_nc=l_nc, k_nc=k_nc)

        elif type_qp == 'normal':
            line_fodo = cell_qp_normal(rcs, l_nc=l_nc, k_nc=k_nc)

        else:
            raise ValueError(f'Unsupported type_qp "{type_qp}". Expected "hybrid" or "normal".')
        if slice:
            line_fodo.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(5))])
        tw = line_fodo.twiss4d()
        beta_max = np.max(tw.betx)
        disp_max = np.max(tw.dx)

        beta.append(beta_max)
        disp.append(disp_max)
        beam_radius.append(n_sigma * np.sqrt(beta_max * tr_emit_n / rcs.inj_gamma) +
                           sigma_delta * disp_max + dx)
        
    return np.array(beam_radius), np.array(disp)

def apert_optics(
        tw : xt.TwissTable, 
        rcs: RingGeometry, 
        sigma_delta: float
        ) -> list[float, float, float, float]:
    """Computes the dipole apertures for a geometry and a cell Twiss table. 

    Args:
        tw (_type_): Twiss table of the cell. 
        rcs (RingGeometry): RCS object defining the geometry.

    Returns:
        list[float, float, float, float]: Traj excursion NC dip, Aperture NC dip, 
                                            Traj excursion SC dip, Aperture SC dip,
    """

    beta_x = np.max(tw.betx)
    disp_x = np.max(tw.dx)
    beam_size_x = sigma_delta*(disp_x) + N_SIGMA*np.sqrt(TR_EMIT_N / rcs.inj_gamma*beta_x)
    
    beta_y = np.max(tw.bety)
    disp_y = np.max(tw.dy)
    beam_size_y = sigma_delta*(disp_y) + N_SIGMA*np.sqrt(TR_EMIT_N / rcs.inj_gamma*beta_y)

    mask_nc = [item == 'BNC' for item in rcs.pattern]
    mask_sc = [item == 'BSC' for item in rcs.pattern]

    if rcs.type == 'hybrid':
        max_width_dip_1 = np.abs(rcs.extrema(rcs.t_ext)[1]-rcs.extrema(rcs.t_inj)[0])
        max_width_dip_2 = np.abs(rcs.extrema(rcs.t_ext)[0]-rcs.extrema(rcs.t_inj)[1])

        width_sc = max(np.max(max_width_dip_1[mask_sc]), np.max(max_width_dip_2[mask_sc]))
        width_nc = max(np.max(max_width_dip_1[mask_nc]), np.max(max_width_dip_2[mask_nc]))

        aperture_sc_hor = width_sc + (beam_size_x + DX)*2
        aperture_sc_vert = (beam_size_y + DX)*2

    elif rcs.type == 'normal':
        width_sc = 0
        aperture_sc_hor = 0
        aperture_sc_vert = 0
        width_nc = 0

    else :
        raise ValueError('Wrong attribute type in RingGeometry object. Must be "normal" or "hybrid"')
    
    aperture_nc_hor = width_nc + (beam_size_x + DX)*2
    aperture_nc_vert = (beam_size_y + DX)*2
    aperture_qp_hor = (beam_size_x + DX)*2

    results = {
        'aperture_qp_hor' : aperture_qp_hor,
        'width_nc' : width_nc,
        'aperture_nc_hor' : aperture_nc_hor,
        'aperture_nc_vert' : aperture_nc_vert,
        'width_sc' : width_sc,
        'aperture_sc_hor' : aperture_sc_hor,
        'aperture_sc_vert' : aperture_sc_vert
    }
    
    return results

def cell_qp_hybrid(
        rcs : RingGeometry, 
        l_sc : float, 
        k_sc : float, 
        l_nc : float, 
        k_nc : float, 
        option : str = 'var_ref', 
        time : float = 1, 
        t_ref : float = T_REF
        ):
    """_summary_

    Args:
        rcs (RingGeometry): RCS object defining the geometry.
        l_sc (float): Length SC qp [m].
        k_sc (float): Strength SC qp [1/m²].
        l_nc (float): Length NC qp [m].
        k_nc (float): Strength NC qp [1/m²].
        option (str, optional): Trajectory option, either 'var_ref' or 'var_k'. Defaults to 'var_ref'.
        time (float, optional): Accelerating time. Defaults to 0.
        t_ref (float, optional): Reference time (useful if 'var_k' is used). Defaults to T_REF.

    Raises:
        ValueError: Invalid option. 

    Returns:
        _type_: Returns cell line. Xsuite format xt.line. 
    """
    env = xt.Environment()
    env.particle_ref = xp.Particles(p0c = rcs.injection_energy + (rcs.extraction_energy-rcs.injection_energy)*time,
                                    q0 = 1, mass0 = xp.MUON_MASS_EV)
    env.vars({  'ksc': k_sc,
                'knc' : k_nc,
                'lsc' : l_sc,
                'lnc' : l_nc,
                'ldrift': rcs.dipole_spacing})
    env.new('qsc', xt.Quadrupole, length=env['lsc']) 
    env.new('qnc', xt.Quadrupole, length=env['lnc']) 
    env.new('drift', xt.Drift, length=env['ldrift'])

    qp_foc = env.new_line(components = [
            env.new('qsc.f', 'qsc', k1='ksc'),
            env.new('drift_qpf', xt.Drift, length=env['ldrift']),
            env.new('qnc.f', 'qnc', k1='knc'),
            ])
    
    qp_defoc = env.new_line(components = [
            env.new('qnc.d', 'qnc', k1='-knc'),
            env.new('drift_qpd', xt.Drift, length=env['ldrift']),
            env.new('qsc.d', 'qsc', k1='-ksc'),
            ])
    
    cell_components = []
    L_dd_path = rcs.L_dd_path(time)
    list_hn_ref = rcs.dip_bending(t_ref) 
    list_hn = rcs.dip_bending(time)
    epsilon = rcs.epsilon(time)
    eps_ref = rcs.epsilon(t_ref)
    L_dip_path = rcs.L_dip_path(time)
    L_dip_path_ref = rcs.L_dip_path(t_ref)
    L_qd_path = rcs.L_qp_dip_path(time)

    for id, item in enumerate(rcs.pattern):
        cell_components.append( env.new(f'drift_{id}', xt.Drift, length=L_dd_path[id]) )
        
        if option == 'var_ref':
            cell_components.append( 
                env.new(item+f'{id}', 
                xt.Bend, 
                k0=list_hn[id], 
                # h=list_hn[id], 
                length=L_dip_path[id],
                angle = list_hn[id] * L_dip_path[id],
                edge_entry_angle=epsilon[id], 
                edge_exit_angle=-epsilon[id+1]
                ))
        elif option == 'var_k':
            cell_components.append(
                env.new(item+f'{id}', 
                xt.Bend, 
                k0=list_hn[id], 
                # h=list_hn_ref[id], 
                length=L_dip_path_ref[id],
                angle = list_hn_ref[id] * L_dip_path_ref[id],
                edge_entry_angle=eps_ref[id], 
                edge_exit_angle=-eps_ref[id+1]
                ))
        else:
            raise ValueError('Invalid option: {}'.format(option))

    cell_components.append(env.new('drift_qd', xt.Drift, length=L_qd_path))

    cell = env.new_line(components = cell_components)
    # cell_1 = env.new(cell, mode='clone')
    # cell_2 = env.new(cell, mode='clone')

    fodo = qp_foc + cell + qp_defoc + cell

    return fodo

def cell_qp_normal(
    rcs : RingGeometry, 
    l_nc : float, 
    k_nc : float, 
    option : str = 'var_ref', 
    time : float = 1, 
    t_ref : float = T_REF
    ) -> xt.Line:
    """_summary_

    Args:
        rcs (RingGeometry): RCS object defining the geometry.
        l_nc (float): Length NC qp [m].
        k_nc (float): Strength NC qp [1/m²].
        option (str, optional): Trajectory option, either 'var_ref' or 'var_k'. Defaults to 'var_ref'.
        time (float, optional): Accelerating time. Defaults to 0.
        t_ref (float, optional): Reference time (useful if 'var_k' is used). Defaults to T_REF.

    Raises:
        ValueError: Invalid option. 

    Returns:
        xt.Line: Returns cell line.
    """

    env = xt.Environment()
    env.particle_ref = xp.Particles(
        p0c = rcs.injection_energy + (rcs.extraction_energy-rcs.injection_energy)*time,
        q0 = 1, 
        mass0 = xp.MUON_MASS_EV)
    
    env.vars({  'knc' : k_nc,
                'lnc' : l_nc,
                'ldrift': rcs.dipole_spacing})
    
    env.new('qnc', xt.Quadrupole, length=env['lnc']/2) 
    env.new('drift', xt.Drift, length=env['ldrift'])

    qp_foc = env.new_line(components = [
            env.new('qnc.f', 'qnc', k1='knc'),
            env.new('qnc.f2', 'qnc', k1='knc')
            ])
    
    qp_defoc = env.new_line(components = [
            env.new('qnc.d', 'qnc', k1='-knc'),
            env.new('qnc.d2', 'qnc', k1='-knc')
            ])
    
    cell_components = []
    L_dd_path = rcs.L_dd_path(time)
    list_hn_ref = rcs.dip_bending(t_ref)
    list_hn = rcs.dip_bending(time)
    epsilon = rcs.epsilon(time)
    eps_ref = rcs.epsilon(t_ref)
    L_dip_path = rcs.L_dip_path(time)
    L_dip_path_ref = rcs.L_dip_path(t_ref)
    L_qd_path = rcs.L_qp_dip_path(time)

    for id, item in enumerate(rcs.pattern):
        cell_components.append( env.new(f'drift_{id}', xt.Drift, length=L_dd_path[id]) )
        
        if option == 'var_ref':
            cell_components.append(env.new(
                item+f'{id}', 
                xt.Bend,
                k0=list_hn[id], 
                # h=list_hn[id],
                angle = list_hn[id]*L_dip_path[id],
                length=L_dip_path[id],
                edge_entry_angle=epsilon[id],
                edge_exit_angle=-epsilon[id+1]
                ))
        elif option == 'var_k':
            cell_components.append(env.new(
                item+f'{id}', 
                xt.Bend,
                k0=list_hn[id], 
                # h=list_hn_ref[id],
                angle = list_hn_ref[id]*L_dip_path_ref[id],
                length=L_dip_path_ref[id],
                edge_entry_angle=eps_ref[id],
                edge_exit_angle=-eps_ref[id+1]))
        else:
            raise ValueError('Invalid option: {}'.format(option))

    cell_components.append(env.new('drift_qd', xt.Drift, length=L_qd_path))

    cell = env.new_line(components = cell_components)
    # cell_1 = env.new(cell, mode='clone')
    # cell_2 = env.new(cell, mode='clone')

    fodo = qp_foc + cell + qp_defoc + cell
    
    return fodo

def y1_plot(
    cell_length: np.ndarray,
    y_values: np.ndarray,
    color: str,
    primary_label: str,
    secondary_label: str,
    n_cell: np.ndarray,
    title: str = None,
    figsize=(6.4, 4.8),
    n_cells_plot: int = 50,
    xticks_nc=None,
):
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(cell_length, y_values, color=color, linewidth=2)
    ax.set_ylim(0)
    ax.set_xlabel("Cell length [m]")
    ax.set_ylabel(primary_label, color=color)
    ax.tick_params(axis="y", labelcolor=color)
    ax.grid(True)

    # Secondary x-axis
    secax = ax.secondary_xaxis("bottom")
    secax.spines["bottom"].set_position(("outward", 45))

    if xticks_nc is None:
        secax.set_xticks(cell_length[::n_cells_plot])
        secax.set_xticklabels([str(int(n)) for n in n_cell[::n_cells_plot]])
    else:
        tick_pos = []
        tick_labels = []

        for nc in xticks_nc:
            idx = np.where(n_cell == nc)[0]
            if len(idx):
                tick_pos.append(cell_length[idx[0]])
                tick_labels.append(str(nc))

        secax.set_xticks(tick_pos)
        secax.set_xticklabels(tick_labels)

    secax.set_xlabel(secondary_label, color="black")
    secax.tick_params(axis="x", colors="black")
    secax.spines["bottom"].set_color("black")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

    return fig


def y2_plot(
    cell_length: np.ndarray,
    y1_values: np.ndarray,
    y2_values: np.ndarray,
    color1: str,
    color2: str,
    primary_label1: str,
    primary_label2: str,
    secondary_label: str,
    n_cell: np.ndarray,
    title: str = None,
    figsize=(6.4, 4.8),
    n_cells_plot: int = 50,
    xticks_nc=None,
):
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(cell_length, y1_values, color=color1, linewidth=2)
    ax1.set_xlabel(r"$L_c$ [m]")
    ax1.set_ylabel(primary_label1, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True)
    ax1.set_ylim(0)

    ax2 = ax1.twinx()
    ax2.plot(cell_length, y2_values, color=color2, linewidth=2)
    ax2.set_ylabel(primary_label2, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0)

    # Secondary x-axis
    secax = ax1.secondary_xaxis("bottom")
    secax.spines["bottom"].set_position(("outward", 45))

    if xticks_nc is None:
        secax.set_xticks(cell_length[::n_cells_plot])
        secax.set_xticklabels([str(int(n)) for n in n_cell[::n_cells_plot]])
    else:
        tick_pos = []
        tick_labels = []

        for nc in xticks_nc:
            idx = np.where(n_cell == nc)[0]
            if len(idx):
                tick_pos.append(cell_length[idx[0]])
                tick_labels.append(str(nc))

        secax.set_xticks(tick_pos)
        secax.set_xticklabels(tick_labels)

    secax.set_xlabel(secondary_label, color="black")
    secax.tick_params(axis="x", colors="black")
    secax.spines["bottom"].set_color("black")

    if title:
        ax1.set_title(title)

    plt.tight_layout()
    plt.show()

    return fig

def find_largest_n_with_dividers(n_cell : np.array):
    def find_largest_divider_in_range(n : int, start : int, end : int):

        # Finds the largest divider of n in the range [start, end]
        for d in range(end, start - 1, -1):
            if n % d == 0:
                return d
        return None

    for n in sorted(n_cell, reverse=True):
        largest_divider = find_largest_divider_in_range(n, 16, 32)
        if largest_divider is not None:
            return int(n), int(largest_divider)

    return None, None

def plot_path_diff(rcs):
    t_geo_path=np.linspace(0,1,20)
    path_length_geo=[]
    for t in t_geo_path:
        path_length_geo.append(rcs.path_length_tot(t))
    path_length_min = np.min(path_length_geo)
    path_length_geo -= path_length_min
    # plt.figure()
    # plt.plot(t_geo_path*rcs.circumference*n_turn_rcs/sc.c*1e3, np.array(path_length_geo)*1e3)
    # plt.xlabel('Time [ms]')
    # plt.ylabel('Path length variation [mm]')
    # # plt.legend()
    # plt.show()
    return (np.max(path_length_geo))

if __name__ == '__main__':

    path = Path(__file__).resolve().parent

    n_size = 3
    rcs_name = 'RCS1_gf'
    folder_fig = '/mnt/c/muco/Results/figures/chap5/FODO/'
    save_fig = False

    input = load_file(path / 'FODO_cells_energy_spread.json')
    
    file_input = input[rcs_name]['file']
    rcs_def = RingGeometry(file_input = file_input)
    SIGMA_DELTA_H = input[rcs_name]['sig_delta_H_'+f'{n_size}']
    SIGMA_DELTA_N = input[rcs_name]['sig_delta_N_'+f'{n_size}']

    N_SIGMA = n_size
    n_turn_rcs = input[rcs_name]['n_turn_rcs']
    # alpha = input[rcs_name]['alpha']
    alpha = 6

    n_cell_qpH, cell_length_qpH, qp_para_qpH, r_qp_qpH = length_qp_hybrid(
        rcs=rcs_def, 
        mu=MU, 
        sigma_delta = SIGMA_DELTA_H, 
        alpha = alpha
        )
    l_sc_qpH, k_sc_const_qpH, gradient_sc_qpH = qp_para_qpH['qp_sc']['length'], qp_para_qpH['qp_sc']['k'], qp_para_qpH['qp_sc']['gradient']
    l_nc_qpH, k_nc_ext_qpH, gradient_nc_ext_qpH = qp_para_qpH['qp_nc']['length'], qp_para_qpH['qp_nc']['k'], qp_para_qpH['qp_nc']['gradient']
    print('QP HYBRID')

    title_2nd_axis = r'$n_c$'

    # LENGTH OF SC/NC QP
    print('SC and NC quadrupole length')
    h_fig_qp_length, ax1 = plt.subplots(figsize=(6.4, 4.8))
    ax1.plot(cell_length_qpH, l_sc_qpH, label=r'SC',color='tab:purple', linewidth = 2) 
    ax1.plot(cell_length_qpH, l_nc_qpH, label=r'NC', color='tab:blue', linewidth = 2)
    ax1.set_xlabel(r'$L_c$ [m]')
    ax1.set_ylabel('Quadrupole Length [m]')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim(0)
    secax = ax1.secondary_xaxis('bottom')
    secax.spines['bottom'].set_position(('outward', 45))  # Move secondary axis outward
    secax.set_xticks(cell_length_qpH[::N_CELLS_PLOT])
    secax.set_xticklabels([f'{int(n)}' for n in n_cell_qpH[::N_CELLS_PLOT]])  # Format as integers
    secax.set_xlabel(title_2nd_axis, color='black')
    secax.tick_params(axis='x', colors='black',)
    secax.spines['bottom'].set_color('black')
    # plt.title('Hybrid QP : SC and NC Quadrupole Length')
    plt.tight_layout()
    plt.show()

    # SC : GRADIENT & APERTURE
    # y2_plot(cell_length_qpH, gradient_sc_qpH, 2 * r_qp_qpH * 1e3, 
    #         'tab:purple', 'red', 
    #         'SC Gradient [T/m]', 'QP Aperture $\\phi$ [mm]', 
    #         title_2nd_axis, 
    #         'Hybrid Quadrupole: SC QP', 
    #         n_cell_qpH )
    
    # SC : GRADIENT & FIELD ON THE POLE
    print('SC : Gradient and pole field')
    h_fig_sc_g = y2_plot(
        cell_length = cell_length_qpH, 
        y1_values = gradient_sc_qpH, 
        y2_values = gradient_sc_qpH * r_qp_qpH, 
        color1 = 'tab:purple', 
        color2 = 'tab:orange', 
        primary_label1 = 'SC Gradient [T/m]', 
        primary_label2 ='SC Pole field [T]', 
        secondary_label = title_2nd_axis, 
        # title = 'Hybrid Quadrupole: SC QP', 
        n_cell = n_cell_qpH 
        )
    
    # # SC : FIELD ON THE POLE & INTEGRATED FIELD
    # y2_plot(cell_length_qpH, gradient_sc_qpH * r_qp_qpH, gradient_sc_qpH * r_qp_qpH * l_sc_qpH,
    #         'blue', 'purple',
    #         'Pole field [T]', 'Integrated field [T.m]',
    #         title_2nd_axis, 'Hybrid Quadrupole: SC QP', n_cell_qpH)

    # NC : GRADIENT & APERTURE
    # y2_plot(
    #     cell_length = cell_length_qpH, 
    #     y1_values = gradient_nc_ext_qpH, 
    #     y2_values = 2 * r_qp_qpH * 1e3, 
    #     color1 = 'green', 
    #     color2 = 'red', 
    #     primary_label1 = 'NC Gradient [T/m]', 
    #     primary_label2 = 'QP Aperture $\\phi$ [mm]', 
    #      secondary_label = title_2nd_axis, 
    #     # title = 'Hybrid Quadrupole: NC QP', 
    #     n_cell = n_cell_qpH 
    #     )

    print('NC : Gradient and pole field')
    h_fig_nc_g = y2_plot(
        cell_length = cell_length_qpH, 
        y1_values = gradient_nc_ext_qpH, 
        y2_values = gradient_nc_ext_qpH*r_qp_qpH,
        color1 = 'tab:blue', 
        color2 = 'tab:red', 
        primary_label1 = 'NC Gradient [T/m]', 
        primary_label2 = 'NC Pole field [T]', 
        secondary_label = title_2nd_axis, 
        # title = 'Hybrid Quadrupole: NC QP', 
        n_cell = n_cell_qpH 
        )
    
    # NC : FIELD ON THE POLE & INTEGRATED FIELD
    # y2_plot( cell_length_qpH, gradient_nc_ext_qpH*r_qp_qpH, gradient_nc_ext_qpH*r_qp_qpH*l_nc_qpH,
    #         'green', 'tab:orange',
    #         'NC QP Pole field [T]', 'NC QP Integrated field [T.m]',
    #         title_2nd_axis, 'Hybrid Quadrupole: NC QP', n_cell_qpH )
    
    print('Quadrupole aperture')
    h_fig_qp_apert = y2_plot( 
        cell_length = cell_length_qpH, 
        y1_values = 2 * r_qp_qpH * 1e3, 
        y2_values = qp_para_qpH['dx']['dx'],
        color1 = 'royalblue',
        color2 = 'brown',
        primary_label1 = 'QP Aperture $\\phi$ [mm]', 
        primary_label2 = r'$D_x$ [m]',
        secondary_label = title_2nd_axis, 
        # title = 'Normal Quadrupole', 
        n_cell = n_cell_qpH 
        )
        
    if save_fig:
        h_fig_qp_length.savefig(folder_fig + 'hybrid_qp_length')
        h_fig_sc_g.savefig(folder_fig + 'hybrid_sc_gradient')
        h_fig_nc_g.savefig(folder_fig + 'hybrid_nc_gradient')
        h_fig_qp_apert.savefig(folder_fig + 'hybrid_qp_apert')
    
    maskH = (gradient_sc_qpH  < 400) & (gradient_nc_ext_qpH * r_qp_qpH < 1)
    
    n_cell_tot_H, nb_arc_H = find_largest_n_with_dividers(n_cell_qpH[maskH])
    indexH = np.where(n_cell_qpH[maskH] == n_cell_tot_H)[0][0]

    nb_cell_arc_H = int(n_cell_tot_H / nb_arc_H)
    rcs_def.nb_arc = 1
    rcs_def.nb_cell_arc = n_cell_tot_H

    n_cell_H, cell_length_H, qp_para_H, r_qp_H = length_qp_hybrid(rcs=rcs_def, n_cell = n_cell_tot_H, mu = MU, sigma_delta = SIGMA_DELTA_H, alpha=alpha)
    l_sc_H, l_nc_H = qp_para_H['qp_sc']['length'], qp_para_H['qp_nc']['length']
    k_sc_H, k_nc_H = qp_para_H['qp_sc']['k'], qp_para_H['qp_nc']['k']
    gradient_sc_H, gradient_nc_H = qp_para_H['qp_sc']['gradient'], qp_para_H['qp_nc']['gradient']

    rcsH = RingGeometry(
        file_input = file_input, 
        length_short_insertion = l_sc_H + l_nc_H + rcs_def.dipole_spacing, 
        nb_cell_arc = nb_cell_arc_H,
        nb_arc=nb_arc_H)
    line_fodo_H = cell_qp_hybrid(rcs = rcsH, l_sc = l_sc_H, l_nc = l_nc_H, k_sc = k_sc_H, k_nc = k_nc_H, option = 'var_ref')
    line_fodo_H.survey().plot()
    line_H = line_fodo_H.copy()

    line_fodo_H.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(3))])
    twH = line_fodo_H.twiss4d()
    twH.plot()
    rcsH.plot_traj()

    dico_apert_H  = apert_optics(twH, rcsH, sigma_delta = SIGMA_DELTA_H)
    aperture_qp_hor_H = dico_apert_H['aperture_qp_hor']
    width_nc_H = dico_apert_H['width_nc']
    aperture_nc_hor_H = dico_apert_H['aperture_nc_hor']
    aperture_nc_vert_H = dico_apert_H['aperture_nc_vert']
    width_sc_H = dico_apert_H['width_sc']
    aperture_sc_hor_H = dico_apert_H['aperture_sc_hor']
    aperture_sc_vert_H = dico_apert_H['aperture_sc_vert']

    if rcsH.type == 'hybrid':
        dip_l_sc_H = rcsH.dipole_families['BSC']['length']
    elif rcsH.type == 'normal':
        dip_l_sc_H = 0
    else :
        raise ValueError('Wrong attribute type in RingGeometry object. Must be "normal" or "hybrid"')
    
    dip_l_nc_H = rcsH.dipole_families['BNC']['length']
    max_path_diff_H = plot_path_diff(rcsH)
    rel_path_diff_H = max_path_diff_H / rcsH.circumference
    mcf_H = twH.momentum_compaction_factor * rcsH.filling_ratio

    parameter_name = ['Circumference [m]', 'Injection energy [GeV]', 'Extraction energy [GeV]', 'Normalized transverse emittance [mm.mrad]', 
                      r'Beam size [$\sigma$]', 'Relative energy dispersion', 'Aperture margin (diam) [mm]', 
                      'Nb arc', 'Nb cells per arc', 'Cell length [m]', 'Interconnection length [m]', 'Length SC Quad. [m]', 'Length NC Quad. [m]', 
                      'SC Quad. Gradient [T/m]', 'SC Quad. Pole Field [T]', 'NC Quad. Gradient [T/m]', 
                      'NC Quad. Pole Field [T]', 'Quad. Aperture [mm]', 
                      'Length SC Dip. [m]', 'Trajectory excursion in SC Dip. [mm]', 'Hor. Aperture SC Dip. [mm]', 'Vert. Aperture SC Dip. [mm]', 
                      'Length NC Dip. [m]', 'Trajectory excursion in NC Dip. [mm]', 'Hor. Aperture NC Dip. [mm]', 'Vert. Aperture NC Dip. [mm]',
                      'Path length diff. [mm]', 'Relative path length diff. [1e-6]', 'Momentum Compaction Factor [1e-4]']
    
    results_hybrid = [
        rcsH.circumference,
        f'{rcsH.injection_energy*1e-9:.0f}',
        f'{rcsH.extraction_energy*1e-9:.0f}',
        f'{TR_EMIT_N*1e6}',
        f'{N_SIGMA}',
        f'{SIGMA_DELTA_H:.2}',
        f'{DX*2*1e3}',
        int(rcsH.nb_arc),
        rcsH.nb_cell_arc, 
        f'{rcsH.cell_length:.1f}', 
        f'{rcsH.dipole_spacing:.1f}',
        f'{l_sc_H:.2f}', 
        f'{l_nc_H:.2f}', 
        f'{gradient_sc_H:.1f}',
        f'{gradient_sc_H*aperture_qp_hor_H/2:.1f}',
        f'{gradient_nc_H:.1f}',
        f'{gradient_nc_H*aperture_qp_hor_H/2:.1f}', 
        f'{aperture_qp_hor_H * 1e3:.1f}', 
        f'{dip_l_sc_H:.1f}', 
        f'{width_sc_H * 1e3:.1f}', 
        f'{aperture_sc_hor_H * 1e3:.1f}', 
        f'{aperture_sc_vert_H * 1e3:.1f}', 
        f'{dip_l_nc_H:.1f}', 
        f'{width_nc_H * 1e3:.1f}', 
        f'{aperture_nc_hor_H * 1e3:.1f}',
        f'{aperture_nc_vert_H * 1e3:.1f}',
        f'{max_path_diff_H * 1e3:.1f}',
        f'{rel_path_diff_H * 1e6:.1f}',
        f'{mcf_H * 1e4:.2f}',
    ]

    print(f'Arc: na={rcsH.nb_arc}')
    print(f'Cell per arc: nc={nb_cell_arc_H}, nc_best={rcsH.nc_best}')
    print(f'Quadrupole length: L_QP_SC={l_sc_H:.1f}m, L_QP_NC={l_nc_H:.1f}m')
    print(f'Dipole length: L_DIP_SC={dip_l_sc_H:.1f}m, L_DIP_NC={dip_l_nc_H:.1f}m')
    print(f'Quadrupole aperture: r_qp={aperture_qp_hor_H*1e3:.1f}m')
    print(f'Gradient: G_SC={gradient_sc_H:.1f}T/m, G_NC={gradient_nc_H:.1f}T/m')
    print(f'Dipole excursion: width_NC={width_nc_H*1e3:.1f}mm, width_sc={width_sc_H*1e3:.1f}mm')
    print(f'Hor. Dipole aperture: aperture_NC={aperture_nc_hor_H*1e3:.1f}mm, aperture_SC={aperture_sc_hor_H*1e3:.1f}mm')
    print(f'Vert. Dipole aperture: aperture_NC={aperture_nc_vert_H*1e3:.1f}mm, aperture_SC={aperture_sc_vert_H*1e3:.1f}mm')
    print(f'Momentum compaction factor: mcf={mcf_H * 1e4:.2f} 1e-4')
    print(f'Path length diff: max_path_diff={max_path_diff_H * 1e3:.1f}mm, rel_path_diff={rel_path_diff_H * 1e6:.1f} 1e-6')
    print('QP NORMAL')

    # sys.exit(0)
           
    rcs_def = RingGeometry(file_input = file_input)
    n_cell_qpN, cell_length_qpN, qp_para_qpN, r_qp_qpN = length_qp_normal(rcs_def, mu=MU, sigma_delta = SIGMA_DELTA_N, slice=False)
    l_qp_nc_qpN, k_nc_ext_qpN, gradient_nc_ext_qpN = qp_para_qpN['qp_nc']['length'], qp_para_qpN['qp_nc']['k'], qp_para_qpN['qp_nc']['gradient']
    
    xticks_nc = [450, 250, 150, 100, 50]
    # LENGTH
    n_fig_qp_length = y1_plot(
        cell_length = cell_length_qpN, 
        y_values = l_qp_nc_qpN, 
        color = 'tab:blue', 
        primary_label = 'Quadrupole Length [m]', 
        secondary_label= title_2nd_axis, 
        # title = 'Normal Quadrupole', 
        n_cell = n_cell_qpN,
        # n_cells_plot = 100,
        xticks_nc = xticks_nc,
        )

    # GRADIENT & APERTURE
    # y2_plot(
    #     cell_length_qpN, 
    #     gradient_nc_ext_qpN, 
    #     2 * r_qp_qpN * 1e3, 
    #     'green', 'red', 
    #     'NC Gradient [T/m]', 'QP Aperture $\\phi$ [mm]', 
    #     title_2nd_axis, 'Normal Quadrupole', n_cell_qpN )

    # GRADIENT & POLE FIELD
    n_fig_nc_gradient = y2_plot(
        cell_length = cell_length_qpN, 
        y1_values = gradient_nc_ext_qpN, 
        y2_values = gradient_nc_ext_qpN * r_qp_qpN, 
        color1 = 'tab:blue', 
        color2 = 'tab:red', 
        primary_label1 = 'NC Gradient [T/m]', 
        primary_label2 = 'Pole field [T]', 
        secondary_label = title_2nd_axis, 
        # title = 'Normal Quadrupole',
        n_cell = n_cell_qpN,
        # n_cells_plot = 100,
        xticks_nc = xticks_nc,
        )

    # APERTURE & DISPERSION
    n_fig_aperture = y2_plot( 
        cell_length = cell_length_qpN, 
        y1_values = 2 * r_qp_qpN * 1e3, 
        y2_values = qp_para_qpN['dx']['dx'],
        color1 = 'royalblue', 
        color2 = 'brown',
        primary_label1 = 'QP Aperture $\\phi$ [mm]', 
        primary_label2 = r'$D_x$ [m]', 
        secondary_label = title_2nd_axis, 
        # title = 'Normal Quadrupole', 
        n_cell = n_cell_qpN,
        # n_cells_plot = 100,
        xticks_nc = xticks_nc
        )

    if save_fig:
        n_fig_qp_length.savefig(folder_fig + 'normal_qp_length')
        n_fig_nc_gradient.savefig(folder_fig + 'normal_nc_gradient')
        n_fig_aperture.savefig(folder_fig + 'normal_qp_apert')

    maskN = gradient_nc_ext_qpN * r_qp_qpN < 1

    n_cell_tot_N, nb_arc_N = find_largest_n_with_dividers(n_cell_qpN[maskN]) 
    indexN = np.where(n_cell_qpN[maskN] == n_cell_tot_N)[0][0]

    nb_cell_arc_N = int(n_cell_tot_N / nb_arc_N)
    rcs_def.nb_arc = 1
    rcs_def.nb_cell_arc = n_cell_tot_N
    n_cell_N, cell_length_N, qp_para_N, r_qp_N = length_qp_normal(rcs_def, n_cell = n_cell_tot_N, mu = MU, sigma_delta = SIGMA_DELTA_N, slice=False)
    l_nc_N, k_nc_N = qp_para_N['qp_nc']['length'], qp_para_N['qp_nc']['k']

    gradient_nc_N = qp_para_N['qp_nc']['gradient']
    rcsN = RingGeometry(file_input = file_input, length_short_insertion =l_nc_N, nb_cell_arc = nb_cell_arc_N, nb_arc = nb_arc_N)
    line_fodo_N = cell_qp_normal(rcs=rcsN, l_nc=l_nc_N, k_nc=k_nc_N, option='var_ref')

    line_N = line_fodo_N.copy()
    line_fodo_N.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(3))])

    twN = line_fodo_N.twiss4d()
    twN.plot()
    dico_apert_N = apert_optics(twN, rcsN, sigma_delta = SIGMA_DELTA_N)
    aperture_qp_hor_N = dico_apert_N['aperture_qp_hor']
    width_nc_N = dico_apert_N['width_nc']
    aperture_nc_hor_N =     dico_apert_N['aperture_nc_hor']
    aperture_nc_vert_N = dico_apert_N['aperture_nc_vert']
    width_sc_N = dico_apert_N['width_sc']
    aperture_sc_hor_N = dico_apert_N['aperture_sc_hor']
    aperture_sc_vert_N = dico_apert_N['aperture_sc_vert']
    
    if rcsN.type == 'hybrid':
        dip_l_sc_N = rcsN.dipole_families['BSC']['length']
    elif rcsN.type == 'normal':
        dip_l_sc_N = 0
    else :
        raise ValueError('Wrong attribute type in RingGeometry object. Must be "normal" or "hybrid"')
    dip_l_nc_N = rcsN.dipole_families['BNC']['length']
    max_path_diff_N = plot_path_diff(rcsN)
    rel_path_diff_N = max_path_diff_N/rcsN.circumference
    mcf_N = twN.momentum_compaction_factor*rcsN.filling_ratio

    print(f'Arc: na={rcsN.nb_arc}')
    print(f'Cell per arc: nc={nb_cell_arc_N}, nc_best={rcsN.nc_best}')
    print(f'Cell length: Lc={rcsN.cell_length}')
    print(f'Quadrupole length: L_QP_NC={l_nc_N:.1f}m')
    print(f'Dipole length: L_DIP_SC={dip_l_sc_N:.1f}m, L_DIP_NC={dip_l_nc_N:.1f}m')
    print(f'Quadrupole aperture: apert_qp={aperture_qp_hor_N*1e3:.1f}mm')
    print(f'Gradient: G_NC={gradient_nc_N:.1f}T/m')
    print(f'Dipole excursion: width_NC={width_nc_N*1e3:.1f}mm, width_sc={width_sc_N*1e3:.1f}mm')
    print(f'Horizontal dipole aperture: aperture_NC={aperture_nc_hor_N*1e3:.1f}mm, aperture_SC={aperture_sc_hor_N*1e3:.1f}mm')
    print(f'Vertical dipole aperture: aperture_NC={aperture_nc_vert_N*1e3:.1f}mm, aperture_SC={aperture_sc_vert_N*1e3:.1f}mm')
    print(f'Path length diff: max_path_diff={max_path_diff_N * 1e3:.1f}mm, rel_path_diff={rel_path_diff_N * 1e6:.1f} 1e-6')
    print(f'Momentum compaction factor: mcf={mcf_N * 1e4:.2f} 1e-4')

    results_normal = [
        rcsN.circumference,
        f'{rcsN.injection_energy*1e-9:.0f}',
        f'{rcsN.extraction_energy*1e-9:.0f}',
        f'{TR_EMIT_N*1e6}',
        f'{N_SIGMA}',
        f'{SIGMA_DELTA_N:.2}',
        f'{DX*2*1e3}',
        int(rcsN.nb_arc),
        rcsN.nb_cell_arc,
        f'{rcsN.cell_length:.1f}',
        f'{rcsN.dipole_spacing:.1f}',
        '-',
        f'{l_nc_N:.2f}',
        '-',
        '-',
        f'{gradient_nc_N:.1f}',
        f'{gradient_nc_N*aperture_qp_hor_N/2:.1f}',
        f'{aperture_qp_hor_N * 1e3:.1f}',
        f'{dip_l_sc_N:.1f}',
        f'{width_sc_N * 1e3:.1f}',
        f'{aperture_sc_hor_N * 1e3:.1f}',
        f'{aperture_sc_vert_N * 1e3:.1f}',
        f'{dip_l_nc_N:.1f}',
        f'{width_nc_N * 1e3:.1f}',
        f'{aperture_nc_hor_N * 1e3:.1f}',
        f'{aperture_nc_vert_N * 1e3:.1f}',
        f'{max_path_diff_N * 1e3:.1f}',
        f'{rel_path_diff_N * 1e6:.1f}',
        f'{mcf_N * 1e4:.2f}',
    ]

    l_qp_min = 4*np.sin(MU/2)*rcsN.ext_Brho/rcsN.cell_length/1*r_qp_N

    def save_para(rcs, folder, qp_type):
        para_rcs = {
        "type": rcs.type,
        "circumference": rcs.circumference,
        "injection_energy": rcs.injection_energy,
        "extraction_energy": rcs.extraction_energy,
        "length_short_insertion": rcs.length_short_insertion,
        "dipole_spacing": rcs.dipole_spacing,
        "nb_RF_section": rcs.nb_RF_section,
        "nb_arc": rcs.nb_arc,
        "nb_cell_arc": rcs.nb_cell_arc,
        "dipoles": rcs.dipoles,
        "filling_ratio": rcs.filling_ratio,
        "filling_ratio_RF": rcs.filling_ratio_RF,
        "pattern": rcs.pattern
        }
    
        with open(folder + f"para_{qp_type}_{rcs_name}.txt", "w") as outfile: 
            json.dump(para_rcs, outfile, indent=4)

    # folder = f'/mnt/c/muco/code/lattices/latex_table_hyb_qd_v2/{rcs_name}/'
    # save_para(rcs=rcsH, folder=folder, qp_type='hybrid')
    # save_para(rcs=rcsN, folder=folder, qp_type='normal')
    # line_fodo_H.to_json(folder+f'hybrid_{rcs_name}.json')
    # line_fodo_N.to_json(folder+f'normal_{rcs_name}.json')
    # name_table = f'tab_hyb_normal_{rcs_name}.tex'
    # df = pd.DataFrame({rcs_name+' Parameters': parameter_name, 'Hybrid QP' : results_hybrid, 'Normal QP' : results_normal})
    # with open(folder+name_table, 'w') as f:
    #     f.write(df.to_latex(index=False, column_format='lcc'))

    from plot_twiss_strength import plot_cell_k_twiss, plot_enveloppe, beam_optics
    
    fig_cell_N = plot_cell_k_twiss(line_N, twN, figsize=(6.4,5))
    beamX_N, beamY_N = beam_optics(
        tw = twN, 
        sigma_delta = SIGMA_DELTA_N, 
        n_sigma = N_SIGMA, 
        tr_emit_n = TR_EMIT_N, 
        inj_gamma = rcsN.inj_gamma)
    fig_env_N = plot_enveloppe(twN, beamX_N, beamY_N, figsize=(6.4, 5))

    fig_cell_H = plot_cell_k_twiss(line_H, twH, figsize=(6.4,5))
    beamX_H, beamY_H = beam_optics(
        tw = twH, 
        sigma_delta = SIGMA_DELTA_H, 
        n_sigma = N_SIGMA, 
        tr_emit_n = TR_EMIT_N, 
        inj_gamma = rcsH.inj_gamma)
    fig_env_H = plot_enveloppe(twH, beamX_H, beamY_H, figsize=(6.4, 5))

    save_fig = False
    if save_fig:
        fig_cell_N.savefig(folder_fig+f'normal_fodo_{rcs_name}_cell_layout')
        fig_env_N.savefig(folder_fig+f'normal_fodo_{rcs_name}_beam_size')
        fig_cell_H.savefig(folder_fig+f'hybrid_fodo_{rcs_name}_cell_layout')
        fig_env_H.savefig(folder_fig+f'hybrid_fodo_{rcs_name}_beam_size')
