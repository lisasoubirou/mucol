"""
Old functions to generate and arc structure. 
"""

import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import sys
import xplt
import scipy.constants as sc

plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

step_quad = 1e-8

def fodo(RCS, n_turn_rcs, length_sxt, time_frac, option, n_slice):
    time=time_frac
    t_ref=0.5
    energy=RCS.injection_energy+(RCS.extraction_energy-RCS.injection_energy)*time_frac

    #Get data from class_geo
    nb_arc = RCS.nb_arc
    nb_cell_arc = RCS.nb_cell_arc
    
    geo_impossible = True
    while geo_impossible:
        L_ins_add = RCS.L_extra_arc-2*RCS.nb_cell_arc*length_sxt-2*RCS.length_short_insertion
        if L_ins_add > 0: 
            L_ext_qp = L_ins_add/(2*nb_cell_arc+2)
            geo_impossible = False
        else:
            print (f'Geometry not possible, decreases nc from {RCS.nb_cell_arc} to {RCS.nb_cell_arc-1}')
            RCS.nb_cell_arc -= 1
    print('Extra per quad:', L_ext_qp)
    
    nb_cell_arc = RCS.nb_cell_arc
    pattern = RCS.pattern
    Ldd = RCS.dipole_spacing 
    L_ins = RCS.insertion_length #+RCS.L_extra_arc
    list_hn = RCS.dip_bending(time) #List of bendings
    list_hn_ref = RCS.dip_bending(t_ref) #List of reference bendings
    epsilon = RCS.epsilon(time)
    eps_ref = RCS.epsilon(t_ref)
    L_dip_path = RCS.L_dip_path(time)
    L_dip_path_ref = RCS.L_dip_path(t_ref)
    L_dd_path = RCS.L_dd_path(time)
    L_dd_path_ref = RCS.L_dd_path(t_ref)
    L_qd_path = RCS.L_qp_dip_path(time)
    L_sxt = length_sxt
    Ls = RCS.length_short_insertion #Length of short straight section

    # L_ext_qp=1/2/(1+nb_cell_arc)*(RCS.L_extra_arc*(1+1/nb_cell_arc)+L_ins-2*(nb_cell_arc+1)*L_sxt-RCS.arc_length/nb_cell_arc)
    # L_ins_add=RCS.L_extra_arc-2*nb_cell_arc*L_sxt-(2*nb_cell_arc+1)*L_ext_qp-2*Ls
    # if L_ext_qp < 0 : 
    #     L_ins_add=L_ins_add+(2*nb_cell_arc+1)*L_ext_qp
    #     L_ext_qp=0
    #     print('No extra next to the quad, L_ins_add=', L_ins_add)
    #     print('Total insertion length', L_ins_add+L_ins)
    #     if L_ins_add < 0:
    #         raise ValueError('Geometry not possible!')
    # if L_ins_add < 0 :
    #     L_ins_add=L_ins_add+(2*nb_cell_arc+1)*L_ext_qp
    # else:
    #     print('Extra per quad:', L_ext_qp)
    #     print(L_ins_add, L_ins, L_ext_qp, RCS.length_short_insertion)
    #     print('Total insertion length', L_ins_add+L_ins+L_ext_qp+2*RCS.length_short_insertion )
    
    #RF data
    n_turns = n_turn_rcs*nb_arc
    energy_increment_per_arc = (RCS.extraction_energy-RCS.injection_energy)/n_turns
    sync_phase=45
    phase=180-sync_phase
    volt_cell=energy_increment_per_arc/np.sin(phase*np.pi/180)

    #Define elements: drifts, quads, sextupoles
    mu = RCS.mu
    f = RCS.cell_length/(4*np.sin(mu/2)) #Focusing strength of quadrupole 
    drift_ext = xt.Drift(length=L_ext_qp/2)
    # drift_ins=xt.Drift(length=(L_ins+L_ins_add)/2)
    drift_ins = xt.Drift(length=L_ins/2)

    def quad_f():
        return xt.Quadrupole(k1=1/f/Ls, length=Ls)

    def quad_d():
        return xt.Quadrupole(k1=-1/f/Ls, length=Ls)

    def quad_d_half():
        return xt.Quadrupole(k1=-1/f/Ls, length=Ls/2)

    def sxt_d():
        return xt.Sextupole(k2=0, length=L_sxt)
    
    def sxt_f():
        return xt.Sextupole(k2=0, length=L_sxt)

    #Cavity settings
    cavity = xt.Cavity(voltage=volt_cell/2, frequency=1.3e9, lag=phase)
    RF_acc = xt.ReferenceEnergyIncrease(Delta_p0c=energy_increment_per_arc/2)

    beg = xt.Marker()
    end = xt.Marker()

    hcell_el=[]
    hcell_name=[]
    for id, item in enumerate(pattern):
        hcell_el.append(xt.Drift(length=L_dd_path[id]))
        hcell_name.append('drift_dd')   
        if option == 'var_ref':
            hcell_el.append(xt.Bend(k0=list_hn[id], h=list_hn[id], length=L_dip_path[id], 
                                    edge_entry_angle=epsilon[id], edge_exit_angle=-epsilon[id+1]))
        elif option == 'var_k':
            hcell_el.append(xt.Bend(k0=list_hn[id], h=list_hn_ref[id], length=L_dip_path_ref[id],
                                    edge_entry_angle=eps_ref[id], edge_exit_angle=-eps_ref[id+1]))
        else:
            raise ValueError("Invalid option: {}".format(option))
        # if item == 'BSC':
        hcell_name.append(item+f'{id}')
        # elif item == 'BNC':
        #     hcell_name.append('BNC')
        # elif item == 'BNC1':
        #     hcell_name.append('BNC1')
    hcell_el.append(xt.Drift(length=L_qd_path))
    hcell_name.append('drift_dd')

    def gen_FODO_el():
        return [drift_ext, quad_f(), sxt_f(), drift_ext]+hcell_el+[drift_ext, quad_d(), sxt_d(), drift_ext]+hcell_el
    
    FODO_el = gen_FODO_el()
    FODO_names = ['drift_ext','quad_f','sxt_f','drift_ext']+hcell_name+['drift_ext','quad_d','sxt_d','drift_ext']+hcell_name 
    
    def gen_FODO_arc():
        list_el = [beg, drift_ext, quad_d_half(), drift_ext, RF_acc, cavity, drift_ins]
        for i in range(int(nb_cell_arc)):
            list_el = list_el + gen_FODO_el()
        list_el = list_el + [drift_ext, quad_f(), drift_ext, drift_ins, cavity, RF_acc, quad_d_half(), end]
        return list_el

    FODO_arc = gen_FODO_arc()
    FODO_arc_names = ['marker_beg','drift_ext','quad_d_half','drift_ext', 'RF_acc', 'cavity', 'drift_ins'] + FODO_names*int(nb_cell_arc) + ['drift_ext','quad_f','drift_ext', 'drift_ins','cavity','RF_acc','quad_d_half', 'marker_end']


    tr_el=FODO_arc[len(FODO_arc)//2:]
    tr_names=FODO_arc_names[len(FODO_arc)//2:]

    # name_counter = {}
    # unique_names = []

    # for name in tr_names:
    #     if name in name_counter:
    #         name_counter[name] += 1
    #     else:
    #         name_counter[name] = 1
    
    #     unique_names.append(f"{name}_{name_counter[name]}")
    #     print(f"{name}_{name_counter[name]}")

    # FODO_names = ['drift_ext','quad_f_1','sxt_f','drift_ext']+hcell_name+['drift_ext','quad_d_1','sxt_d','drift_ext']+hcell_name 


    # return (FODO_arc, FODO_arc_names)
    return(FODO_el, FODO_names, tr_el, tr_names)


def plot_layout(line):
    sv = line.survey()
    plot = xplt.FloorPlot(
        sv,
        line,
        projection="ZX",
        boxes={  # adjust box style for element names matching regex
            "BSC": dict(color="maroon",width=5),
            "BNC": dict(color="blue",width=5),
            "quad_d" : dict(color="green",width=10),
            "quad_f" : dict(color="green",width=10),
            "sxt_d" : dict(color="purple",width=15),
            "sxt_f" : dict(color="purple",width=15),
        }
)
    plot.legend(loc="lower left")

def plot_twiss(tw):
    fig1 = plt.figure(1, figsize=(6.4, 4.8*1.5))
    bet = plt.subplot(2,1,1)
    disp = plt.subplot(2,1,2, sharex=bet)

    bet.plot(tw.s, tw.betx, label=r'$\beta_x$')
    bet.plot(tw.s, tw.bety,label=r'$\beta_y$')
    bet.set_ylabel(r'$\beta_{x,y}$ [m]')
    # bet.axvspan(tw['s','marker_beg'], tw['s','quad_f'],
    #                 color='b', alpha=0.1, linewidth=0)
    # bet.axvspan(tw['s','drift_ins_1'], tw['s','marker_end'],
    #                 color='b', alpha=0.1, linewidth=0)
    bet.legend()

    # spbet.legend()
    disp.plot(tw.s, tw.dx,label=r'$D_x$')
    disp.plot(tw.s, tw.dy,label=r'$D_y$')
    disp.set_ylabel(r'$D_{x,y}$ [m]')
    disp.set_xlabel(r"s [m]")
    disp.legend(loc='lower center')
    # disp.axvspan(tw['s','marker_beg'], tw['s','quad_f'],
    #                 color='b', alpha=0.1, linewidth=0)
    # disp.axvspan(tw['s','drift_ins_1'], tw['s','marker_end'],
    #                 color='b', alpha=0.1, linewidth=0)
    # fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
    plt.tight_layout()
    plt.show()
    return fig1

def plot_chroma(tw):
    fig1 = plt.figure(1, figsize=(6.4, 4.8*1.5))
    bet = plt.subplot(2,1,1)
    disp = plt.subplot(2,1,2, sharex=bet)

    bet.plot(tw.s, tw.wx_chrom, label=r'$W_x$')
    bet.plot(tw.s, tw.wy_chrom,label=r'$W_y$')
    bet.set_ylabel(r'$W_{x,y}$')
    # bet.axvspan(tw['s','marker_beg'], tw['s','quad_f'],
    #                 color='b', alpha=0.1, linewidth=0)
    # bet.axvspan(tw['s','drift_ins_1'], tw['s','marker_end'],
    #                 color='b', alpha=0.1, linewidth=0)
    bet.legend()

    # spbet.legend()
    disp.plot(tw.s, tw.ddx,label=r"$D'_x$")
    disp.plot(tw.s, tw.ddy,label=r"$D'_y$")
    disp.set_ylabel(r"$D'_{x,y}$ [m]")
    disp.set_xlabel(r"s [m]")
    disp.legend()
    # disp.axvspan(tw['s','marker_beg'], tw['s','quad_f'],
    #                 color='b', alpha=0.1, linewidth=0)
    # disp.axvspan(tw['s','drift_ins_1'], tw['s','marker_end'],
    #                 color='b', alpha=0.1, linewidth=0)
    # fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
    plt.tight_layout()
    plt.show()
    return fig1

def apert_1b_optics(tw, rcs):
    beta_max = np.max(tw.betx)
    disp_max = np.max(tw.dx)
    beam_size = rcs.sigma_delta*(disp_max) + rcs.n_sigma*np.sqrt(rcs.tr_emittance_inj*beta_max)
    mask_nc = [item == 'BNC' for item in rcs.pattern]
    mask_sc = [item == 'BSC' for item in rcs.pattern]
    max_width_dip = rcs.extrema(rcs.t_ext)[1]-rcs.extrema(rcs.t_inj)[0]
    width_sc = np.max(max_width_dip[mask_sc])
    width_nc = np.max(max_width_dip[mask_nc])

    aperture_sc= width_sc + (beam_size + rcs.dx_qp)*2
    aperture_nc= width_nc + (beam_size + rcs.dx_qp)*2
    return (aperture_nc, aperture_sc)

def apert_2b_optics(tw, rcs):
    beta_max = np.max(tw.betx)
    disp_max = np.max(tw.dx)
    beam_size = rcs.sigma_delta*(disp_max) + rcs.n_sigma*np.sqrt(rcs.tr_emittance_inj*beta_max)
    nd=len(rcs.pattern)
    n_bloc=len(rcs.pattern)//2
    y_rot_inj=rcs.rot_coord(rcs,n_bloc,0)[1]
    y_rot_ext=rcs.rot_coord(rcs,n_bloc,1)[1]
    y_max=np.array([np.max(y_rot_ext[id*11:(id+1)*11]) for id in range(nd//2)])
    y_min=np.array([np.min(y_rot_inj[id*11:(id+1)*11]) for id in range(nd//2)])
    max_width_dip=y_max-y_min
    mask_nc=[item == 'BNC' for item in rcs.pattern[0:n_bloc]]
    mask_sc=[item == 'BSC' for item in rcs.pattern[0:n_bloc]]
    width_sc=np.max(max_width_dip[mask_sc])
    width_nc=np.max(max_width_dip[mask_nc])
    aperture_sc= width_sc + (beam_size + rcs.dx_qp)*2
    aperture_nc= width_nc + (beam_size + rcs.dx_qp)*2
    return (aperture_nc, aperture_sc)

def size_beam_diam_hor(tw, rcs, delta=None):
    if delta is None:
        delta = rcs.sigma_delta
    beta_max = np.max(tw.betx)
    disp_max = np.max(tw.dx)
    beam_size = 2*(delta*(disp_max) + rcs.n_sigma*np.sqrt(rcs.tr_emittance_inj*beta_max))
    return beam_size

def size_beam_diam_vert(tw, rcs,delta=None):
    if delta is None:
        delta = rcs.sigma_delta
    beta_max = np.max(tw.bety)
    disp_max = np.max(tw.dy)
    beam_size = 2*(delta*(disp_max) + rcs.n_sigma*np.sqrt(rcs.tr_emittance_inj*beta_max))
    return beam_size

def vert_apert(tw, rcs, delta=None):
    if delta is None:
        delta = rcs.sigma_delta
    vert_beam = size_beam_diam_vert(tw, rcs, delta)
    apert = vert_beam + 2*rcs.dx_qp
    return apert 

def plot_path_diff(rcs, n_turn_rcs):
    t_geo_path=np.linspace(0,1,20)
    path_length_geo=[]
    for t in t_geo_path:
        path_length_geo.append(rcs.path_length_tot(t)-rcs.path_length_min)
    plt.figure()
    plt.plot(t_geo_path*rcs.circumference*n_turn_rcs/sc.c*1e3, np.array(path_length_geo)*1e3)
    plt.xlabel('Time [ms]')
    plt.ylabel('Path length variation [mm]')
    # plt.legend()
    plt.show()
    return (np.max(path_length_geo))
