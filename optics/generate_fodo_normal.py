"""
    Methods to generate an arc geometry and match the 
    dispersion suppressor for a conventional FODO cell. 
"""

import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import xpart as xp
from hybrid_qp import length_qp_normal
from class_geometry.ring_geometry.ring_geometry import RingGeometry, load_file
from generate_fodo import plot_twiss
from hybrid_qp import apert_optics, plot_path_diff
import json 

plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

STEP_QUAD = 1e-8
STEP_SXT = 1e-4

RF_FREQUENCY = 1.3e9

class DS_normal:
    def __init__(
            self, 
            rcs, 
            n_turn_rcs, 
            time_frac, 
            option, 
            mu, 
            t_ref=0.5, 
            add_quad=0, 
            plot=True, 
            plot_match=False,
            matrix_stability_tol = 0.1
            ):

        self.rcs = rcs
        self.n_turn_rcs = n_turn_rcs
        self.time_frac = time_frac
        self.t_ref = t_ref
        self.option = option
        self.mu = mu
        self.add_quad = add_quad
        self.plot = plot
        self.plot_match = plot_match
        self.matrix_stability_tol = matrix_stability_tol

        self.energy = self.rcs.injection_energy+(self.rcs.extraction_energy-self.rcs.injection_energy)*self.time_frac
        
        self.n_qd_match = self.rcs.nb_cell_arc + 1
        if self.rcs.nb_cell_arc < 3 :
            raise ValueError(f'DS matching not possible for {rcs.nb_cell_arc} cells per arc')

    def fodo_normal_qp (self):

        def _quad(k1, length):
            return xt.Quadrupole(k1 = k1, length = length)

        # Get geometry
        nb_arc = self.rcs.nb_arc   
        nb_cell_arc = int(self.rcs.nb_cell_arc)
        pattern = self.rcs.pattern
        L_ins = self.rcs.insertion_length #+self.rcs.L_extra_arc
        list_hn = self.rcs.dip_bending(self.time_frac) #List of bendings
        list_hn_ref = self.rcs.dip_bending(self.t_ref) #List of reference bendings
        epsilon = self.rcs.epsilon(self.time_frac)
        eps_ref = self.rcs.epsilon(self.t_ref)
        L_dip_path = self.rcs.L_dip_path(self.time_frac)
        L_dip_path_ref = self.rcs.L_dip_path(self.t_ref)
        L_dd_path = self.rcs.L_dd_path(self.time_frac)
        L_qd_path = self.rcs.L_qp_dip_path(self.time_frac)
        Ls = self.rcs.length_short_insertion #Length of short straight section
        
        # RF data
        n_turns = self.n_turn_rcs*nb_arc
        energy_increment_per_arc = (self.rcs.extraction_energy-self.rcs.injection_energy)/n_turns
        sync_phase = 45
        phase = 180 - sync_phase
        volt_cell = energy_increment_per_arc/np.sin( phase*np.pi/180 )

        # Defining quads 
        f = self.rcs.cell_length/(4*np.sin(self.mu/2)) # Focusing strength of quadrupole 
        kf = 1/f/Ls
        kd = -1/f/Ls
        
        # Defining drifts
        L_ins = L_ins - 2*Ls  # !! We put quad for the rf insertion on the place for the rf, to me modified later by according margin from filling factor
        drift_ins = xt.Drift(length = L_ins/2)
    
        # Cavity settings
        cavity = xt.Cavity(
            voltage = volt_cell/2, 
            frequency = RF_FREQUENCY, 
            lag = phase)
        RF_acc = xt.ReferenceEnergyIncrease(Delta_p0c = energy_increment_per_arc/2)

        beg = xt.Marker()
        end = xt.Marker()

        #Dipole cell
        hcell_el=[]
        hcell_name=[]
        for id, item in enumerate(pattern):
            hcell_el.append(xt.Drift(length = L_dd_path[id]))
            hcell_name.append(f'drift_dd_{id}')   

            if self.option == 'var_ref':
                hcell_el.append(xt.Bend(
                    k0 = list_hn[id], 
                    angle = L_dip_path[id]*list_hn[id],
                    length = L_dip_path[id], 
                    edge_entry_angle = epsilon[id], 
                    edge_exit_angle = -epsilon[id+1]
                    ))
                
            elif self.option == 'var_k':
                hcell_el.append(xt.Bend(
                    k0 = list_hn[id], 
                    angle = L_dip_path_ref[id]*list_hn_ref[id],
                    length = L_dip_path_ref[id],
                    edge_entry_angle = eps_ref[id], 
                    edge_exit_angle = -eps_ref[id+1]
                    ))
                
            else:
                raise ValueError("Invalid option: {}".format(self.option))
            
            hcell_name.append(item + f'{id}')

        hcell_el.append(xt.Drift(length = L_qd_path))
        hcell_name.append(f'drift_dd_{id+1}')
        
        # FODO cell
        FODO_el = [_quad(kf, Ls)] + hcell_el + [_quad(kd, Ls)] + hcell_el 
        FODO_names = ['qp_foc'] + hcell_name + ['qp_defoc'] + hcell_name 

        line_FODO = xt.Line(
            elements=FODO_el, 
            element_names=FODO_names
            )
        line_FODO.particle_ref = xp.Particles(
            p0c=self.energy, 
            q0=1, 
            mass0=xp.MUON_MASS_EV
            )
        
        line_FODO.vars['kf'] = kf
        line_FODO.element_refs['qp_foc'].k1 = line_FODO.vars['kf']
        line_FODO.vars['kd'] = kd
        line_FODO.element_refs['qp_defoc'].k1 = line_FODO.vars['kd']

        qx =  self.mu / (2*np.pi)
        qy = self.mu / (2* np.pi)
        line_FODO.match(
                vary = [xt.VaryList(['kf', 'kd'], step=STEP_QUAD, tag='quad')],
                targets = [xt.TargetSet(qx = qx, qy = qy, tol=1e-9)], 
                method = '4d',
                solve = True
                )
        
        # Update strength after FODO matching
        kf = line_FODO['qp_foc'].k1
        kd = line_FODO['qp_defoc'].k1
        
        if self.add_quad == 0:
            FODO_arc = [beg, _quad(kd, Ls/2), RF_acc, cavity, drift_ins]
            FODO_arc_names = ['marker_beg', 'qp0', 'RF_acc_1', 'cavity_1', 'drift_RF_ins_1']

        elif self.add_quad in [1, 3, 5, 7, 9]:
            num_quads = self.add_quad
            num_drifts = 2 * num_quads
            L_ins -= 2 * num_quads * Ls
            drift_ins = xt.Drift(length= L_ins / (num_drifts + 2))

            FODO_arc = [beg, _quad(kf, Ls/2), RF_acc, cavity]
            FODO_arc_names = ['marker_beg', 'qp0', 'RF_acc_1', 'cavity_1']

            for i in range(1, num_quads + 1):
                quad = _quad(kd, Ls) if i % 2 == 1 else _quad(kf, Ls)
                FODO_arc += [drift_ins, quad]
                FODO_arc_names += [f'drift_RF_ins_{i}', f'qp{i}']

            FODO_arc.append(drift_ins)
            FODO_arc_names.append(f'drift_RF_ins_{num_quads+1}')

        else:
            raise ValueError(f'Add quad {self.add_quad}')

        for i in range(nb_cell_arc):
            FODO_arc += [_quad(kf, Ls)] + hcell_el + [_quad(kd, Ls)] + hcell_el
            FODO_arc_names += [f'qp{self.add_quad+2*i+1}'] + [el + f'_h0_{i}' for el in hcell_name] + [f'qp{self.add_quad+2*i+2}'] + [el + f'_h1_{i}' for el in hcell_name]
        
        end_index = self.add_quad + 2 * nb_cell_arc + 1

        if self.add_quad == 0:
            FODO_arc += [_quad(k1=kf, length=Ls), drift_ins] #, cavity, RF_acc, _quad(kd, Ls/2), end]
            FODO_arc_names += [f'qp{end_index}', 'drift_RF_ins_2'] #, 'cavity_2', 'RF_acc_2', f'qp{end_index + 1}', 'marker_end']
        
        elif self.add_quad in [1, 3, 5, 7, 9]:
            for i in range(self.add_quad+1):
                quad = _quad(kf, Ls) if i % 2 == 0 else _quad(kd, Ls)
                FODO_arc += [quad, drift_ins]
                FODO_arc_names += [f'qp{end_index + i}', f'drift_RF_ins_{self.add_quad + 2 + i}']

        else:
            raise ValueError(f'Add quad {self.add_quad}')

        FODO_arc += [cavity, RF_acc, _quad(kf, Ls/2), end]
        FODO_arc_names += ['cavity_2', 'RF_acc_2', f'qp{end_index + self.add_quad}', 'marker_end']

        return(FODO_el, FODO_names, FODO_arc, FODO_arc_names)
    
    def attach_knobs(self, line, i_a = None):
        if not i_a :
            i_a = 5 + self.add_quad
    
        if self.rcs.nb_cell_arc >= 4:
            for i in np.arange(0,i_a+1):
                line.vars[f'k{i+1}'] = line[f'qp{i+1}'].k1
                line.element_refs[f'qp{i+1}'].k1 = line.vars[f'k{i+1}']
                print(f'qp{i+1}', f'k{i+1}')
            # for i in np.arange(1,i_a):
            #     line.vars[f'k{i}'] = line[f'qp{i}'].k1
            #     line.element_refs[f'qp{i}'].k1 = line.vars[f'k{i}']
            #     print(f'qp{i}', f'k{i}')
        # if self.rcs.nb_cell_arc >= i_a:
        #     line.vars['ka'] = line[f'qp{i_a+2}'].k1
        #     for i in np.arange(i_a+1, self.n_qd_match+3, 2):
        #         line.element_refs[f'qp{i+1}'].k1 = line.vars['ka']
        #         print(f'qp{i+1}', 'ka')
        # if self.rcs.nb_cell_arc >= i_a+1:
        #     line.vars['kb'] = line[f'qp{i_a+3}'].k1
        #     for i in np.arange(i_a+2, self.n_qd_match+3, 2):
        #         line.element_refs[f'qp{i+1}'].k1 = line.vars['kb']
        #         print(f'qp{i+1}', 'kb')

    def attach_knobs_arc(self, line, i_a = None):
        if not i_a :
            i_a = 5 + self.add_quad
        if self.rcs.nb_cell_arc >= 4:
            for i in np.arange(0,i_a):
                line.vars[f'k{i+1}'] = line[f'qp{i+1}_r'].k1
                line.element_refs[f'qp{i+1}_l'].k1 = line.vars[f'k{i+1}']
                line.element_refs[f'qp{i+1}_r'].k1 = line.vars[f'k{i+1}']
                print(f'qp{i+1}', f'k{i+1}')
            # for i in np.arange(1,i_a):
            #     line.vars[f'k{i}'] = line[f'qp{i}_r'].k1
            #     line.element_refs[f'qp{i}_l'].k1 = line.vars[f'k{i}']
            #     line.element_refs[f'qp{i}_r'].k1 = line.vars[f'k{i}']
            #     print(f'qp{i}', f'k{i}')
        # if self.rcs.nb_cell_arc >= i_a:
        #     line.vars['ka'] = line[f'qp{i_a+2}_r'].k1
        #     for i in np.arange(i_a+1, self.n_qd_match+1, 2):
        #         line.element_refs[f'qp{i+1}_l'].k1 = line.vars['ka']
        #         line.element_refs[f'qp{i+1}_r'].k1 = line.vars['ka']
        #         print(f'qp{i+1}', 'ka')
        # if self.rcs.nb_cell_arc >= i_a+1:
        #     line.vars['kb'] = line[f'qp{i_a+3}_r'].k1
        #     for i in np.arange(i_a+2, self.n_qd_match+1, 2):
        #         line.element_refs[f'qp{i+1}_l'].k1 = line.vars['kb']
        #         line.element_refs[f'qp{i+1}_r'].k1 = line.vars['kb']
        #         print(f'qp{i+1}', 'kb')
    
    def match_tr(self, line, knobs = None):
        if not knobs:
            knobs = ['k1',
                    'k2','k3',
                    'k4', 'k5',
                    # 'k6',
                    # 'k7'
                    # 'k1', 'k2', 'k3', 'k4', 'k5'
                    # 'k3', 'k4', 'k5', 'k6','k7'
                    # 'k5','k6','k7','k8','k9', 
                    # 'k6','k7', 'k8','k9', 'k10',
            ]
        matcher = line.match(
                start = line.element_names[0],
                end = line.element_names[-1], 
                init = self.tw_init,
                vary = [xt.VaryList(
                    knobs,
                    step=STEP_QUAD, 
                    tag='quad'
                    )],
                targets = [xt.TargetSet(dx=0, dpx=0, alfx=0, alfy=0, at='marker_beg', tol=1e-9)],
                method = '4d',
                solve = True
                )
        if self.plot_match:
            print('RESULTS FROM TR MATCHING')
            matcher.target_status()
            matcher.vary_status()
        return matcher

    def match_ds(self, line, knobs = None, qx = None, qy = None):
        target_list = [xt.TargetSet(dx=0, at=0, tol=1e-9)]

        if qx is not None:
            target_list.append(xt.TargetSet(qx=qx, tol=1e-9))
        if qy is not None:
            target_list.append(xt.TargetSet(qy=qy, tol=1e-9))
        

        if not knobs:
            knobs = [
                'k1', 'k2', 'k3',
                'k4', 'k5', 
                # 'k6',
                # 'k7'
                # 'k1', 'k2', 'k3', 'k4', 'k5'
                # 'k3', 'k4', 'k5', 'k6','k7'
                # 'k5','k6','k7','k8','k9', 
                # 'k6','k7', 'k8','k9', 'k10', 
                ]
        matcher = line.match(
                vary = [xt.VaryList(
                    knobs,
                    step=STEP_QUAD, 
                    tag='quad'
                    )],
                targets= target_list,
                method='6d',
                solve=True,
                matrix_stability_tol=self.matrix_stability_tol
                )
        if self.plot_match:
            print('RESULTS FROM DS MATCHING')
            matcher.target_status()
            matcher.vary_status()
        return matcher

    def match_chroma(self, line, dqx_rcs, dqy_rcs):
        line.vars['k_sxt1']=0
        line.vars['k_sxt2']=0
        for el in line.element_names:
            if 'qp' in el:
                if line[el].k1 > 0 :
                    line.element_refs[el].knl[2] = line.vars['k_sxt1']
                else : 
                    line.element_refs[el].knl[2] = line.vars['k_sxt2']

        matcher = line.match(
                    vary = [xt.VaryList(['k_sxt1', 'k_sxt2'], step=STEP_SXT, tag='quad')],
                    targets = [xt.TargetSet(dqx = dqx_rcs / self.rcs.nb_arc, 
                                            dqy = dqy_rcs / self.rcs.nb_arc, tol=1e-6)], 
                    method = '6d',
                    solve = True,
                    matrix_stability_tol = self.matrix_stability_tol
                    )
        if self.plot_match:
            matcher.target_status()
            matcher.vary_status()
        return matcher
    
    def build_arc_ds(self, ia = None, knobs = None, qx = None, qy = None):

        FODO_el, FODO_names, tr_el, tr_names = self.fodo_normal_qp()

        # FODO cell
        self.line_FODO = xt.Line(
            elements=FODO_el, 
            element_names=FODO_names
            )
        self.line_FODO.particle_ref = xp.Particles(
            p0c=self.energy, 
            q0=1, 
            mass0=xp.MUON_MASS_EV
            )
        self.tw_FODO = self.line_FODO.twiss4d()

        if self.plot:
            plot_twiss(self.tw_FODO)

        if self.rcs.nb_cell_arc%2 == 0:
            self.tw_init = self.tw_FODO.get_twiss_init('qp_foc')
        else:
            self.tw_init = self.tw_FODO.get_twiss_init('qp_defoc')

        # TRANSFER LINE: half-arc
        self.line_tr = xt.Line(
            element_names=tr_names[:len(tr_names)//2+1][::-1], 
            elements=tr_el[:len(tr_el)//2+1][::-1])
        self.line_tr.particle_ref = xp.Particles(
            p0c=self.energy, 
            q0=1,
            mass0=xp.MUON_MASS_EV
            )
        
        self.tw_init.element_name = self.line_tr.element_names[0]
        self.attach_knobs(self.line_tr, ia)
        self.match_tr(self.line_tr, knobs)

        self.tw_tr = self.line_tr.twiss(
            start = self.line_tr.element_names[0],
            end = self.line_tr.element_names[-1], 
            init = self.tw_init
            )
        if self.plot:
            plot_twiss(self.tw_tr)

        line1 = xt.Line(
            elements=self.line_tr.elements[::-1][:-1], 
            element_names=self.line_tr.element_names[::-1][:-1]
            )

        # Inverse dipole names to correct entry and exit edges. 
        nd = len(self.rcs.pattern) - 1
        for id, name in enumerate(line1.element_names):
            if ('BSC' in name) or ('BNC' in name):
                line1.element_names[id] = name[:3] + str(nd-int(name[3])) + name[4:]
        
        self.line_arc = xt.Line(
            elements = line1.elements + self.line_tr.elements,
            # element_names = [el + '_l' for el in line1.element_names] + [
                        #el + '_r' for el in self.line_tr.element_names])
            element_names = [el for el in line1.element_names] + [el for el in self.line_tr.element_names])

        self.line_arc.particle_ref = xp.Particles(
            p0c=self.energy, 
            q0=1, 
            mass0=xp.MUON_MASS_EV
            )

        # self.attach_knobs_arc(self.line_arc)
        self.attach_knobs(self.line_arc, ia)
        self.match_ds(self.line_arc, knobs, qx, qy)

        self.line_arc = xt.Line(
            elements = line1.elements + self.line_tr.elements,
            element_names = [el + '_l' for el in line1.element_names] + [
                                        el + '_r' for el in self.line_tr.element_names])
        self.line_arc.particle_ref = xp.Particles(
            p0c=self.energy, 
            q0=1, 
            mass0=xp.MUON_MASS_EV)

        #Matching chromaticity
        #match_chroma(self.line_arc)

        self.tw_arc = self.line_arc.twiss(matrix_stability_tol=self.matrix_stability_tol)

        if self.plot:
            plot_twiss(self.tw_arc)

def save_para(rcs, arc, rcs_name, folder):
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

    with open(f"/mnt/c/muco/code/{folder}/para_{rcs_name}_normal_qp_{arc}.json", "w") as outfile: 
        json.dump(para_rcs, outfile, indent=4)


def save_tunes(tw, rcs, arc, folder, rcs_name):
    para_rcs = {
    "nb_arc": rcs.nb_arc,
    "qx_arc": tw.qx,
    "qy_arc": tw.qy,
    "qx_ring": tw.qx*rcs.nb_arc,
    "qy_ring": tw.qy*rcs.nb_arc,
    "betx_mean": np.mean(tw.betx),
    "bety_mean": np.mean(tw.bety),
    'dqx': tw.dqx,
    'dqy': tw.dqy,
    'mcf' : tw.momentum_compaction_factor,
    }

    with open(f"/mnt/c/muco/code/{folder}/tunes_{rcs_name}_normal_qp_{arc}.json", "w") as outfile: 
        json.dump(para_rcs, outfile, indent=4)

def save_table(rcs, tw_arc, qp_para, n_sigma, sigma_delta, folder, rcs_name, add_quad):
    dico_apert  = apert_optics(tw_arc, rcs, sigma_delta = sigma_delta)
    path_length_diff_rel = plot_path_diff(rcs) / rcs.circumference

    if rcs.type == 'hybrid':
        dip_sc_length = rcs.dipole_families['BSC']['length']
    elif rcs.type == 'normal':
        dip_sc_length = 0
    
    table = {
        'filling_ratio_dipole' : rcs.dipole_length_tot/rcs.circumference,
        'nb_cell_arc' : rcs.nb_cell_arc,
        'nb_arc' : rcs.nb_arc,
        'cell_length' : rcs.cell_length,
        'NC_apert_hor' : dico_apert['aperture_nc_hor'],
        'SC_apert_hor' : dico_apert['aperture_sc_hor'],
        'quad_apert_hor' : dico_apert['aperture_qp_hor'],
        'apert_vert' : dico_apert['aperture_nc_vert'],
        'path_length_diff_rel' : path_length_diff_rel,
        'mcf' : tw_arc.momentum_compaction_factor,
        'pattern' : rcs.pattern,
        'qp_length' : rcs.length_short_insertion,
        'dip_sc_length' : dip_sc_length,
        'dip_nc_length' : rcs.dipole_families['BNC']['length'],
        'gradient_qp_nc' : qp_para['qp_nc']['gradient'],
        'n_sigma' : n_sigma,
        'sigma_delta' : sigma_delta,
        'add_quad': add_quad
    }

    with open(f"/mnt/c/muco/code/{folder}/table_{rcs_name}_normal_qp_{rcs.nb_arc}.json", "w") as outfile: 
        json.dump(table, outfile, indent=4)

if __name__ == '__main__':

    MU = np.pi/2
    n_size = 3
    rcs_name = 'RCS1_gf'

    input = load_file('/mnt/c/muco/code/FODO_cells_energy_spread.json')

    sigma_delta = input[rcs_name]['sig_delta_N_'+f'{n_size}']
    file_input = input[rcs_name]['file']
    n_turn_rcs = input[rcs_name]['n_turn_rcs']

    rcs = RingGeometry(file_input = file_input)
    n_cell_qpN, cell_length_qpN, qp_para_qpN, r_qp_qpN = length_qp_normal(
        rcs, 
        mu=MU, 
        sigma_delta=sigma_delta, 
        slice=False
        )
    l_qp_nc_qpN, k_nc_ext_qpN, gradient_nc_ext_qpN = qp_para_qpN['qp_nc']['length'], qp_para_qpN['qp_nc']['k'], qp_para_qpN['qp_nc']['gradient']
    mask = gradient_nc_ext_qpN * r_qp_qpN < 1

    # If we want to work with maximum number of cells
    # n_cell_tot_N, nb_arc_N = find_largest_n_with_dividers(n_cell_qpN[mask])

    # If we want to work with a given number of arcs
    nb_arc_N = 28
    n_cell_tot_N = max([nc for nc in n_cell_qpN[mask] if nc % nb_arc_N == 0], default=None)

    # Resolution for given geometry
    nb_cell_arc_N = int(n_cell_tot_N / nb_arc_N)
    n_cell_N, cell_length_N, qp_para_N, r_qp_N = length_qp_normal(
        rcs, 
        n_cell = n_cell_tot_N, 
        mu = MU, 
        sigma_delta=sigma_delta, 
        slice = False
        )
    l_nc_N, k_nc_N = qp_para_N['qp_nc']['length'], qp_para_N['qp_nc']['k']
    rcs = RingGeometry(
        file_input = file_input, 
        length_short_insertion = l_nc_N, 
        nb_cell_arc = nb_cell_arc_N, 
        nb_arc = nb_arc_N
        )

    time_frac = 0
    option = 'var_ref'
    mu = np.pi/2

    add_quad = 1  #If add_quad is 7, maybe uncommenting k7 in matching routine can help
    DS = DS_normal(
        rcs = rcs, 
        n_turn_rcs=n_turn_rcs, 
        time_frac=time_frac, 
        option=option, mu=mu, 
        add_quad=add_quad, 
        plot=True
        )
    
    DS.build_arc_ds()
    line_FODO = DS.line_FODO
    line_arc = DS.line_arc
    tw_arc = DS.tw_arc

    fig = plot_twiss(tw_arc)

    apert_qp, width_nc_N, nc_apert_hor_N, nc_apert_vert_N, width_sc_N, sc_apert_hor_N, sc_apert_vert_N = apert_optics(tw_arc, rcs, sigma_delta = sigma_delta)
    if rcs.type == 'hybrid':
        dip_l_sc_N = rcs.dipole_families['BSC']['length']
    elif rcs.type == 'normal':
        dip_l_sc_N = 0
    else :
        raise ValueError('Wrong attribute type in Geometry object. Must be "normal" or "hybrid"')
    
    dip_l_nc_N = rcs.dipole_families['BNC']['length']
    max_path_diff_N = plot_path_diff(rcs)
    rel_path_diff_N = max_path_diff_N/rcs.circumference
    mcf_N = tw_arc.momentum_compaction_factor


    # folder = 'lattice_consolidated_ESPPU'
    # save_para(rcs=rcs, arc=f'_{nb_arc_N}_arc', folder=folder)
    # save_tunes(tw=tw_arc, rcs=rcs, arc=f'_{nb_arc_N}_arc', folder=folder)
    # save_table(rcs=rcs, tw_arc=tw_arc, qp_para=qp_para_N, 
    #            n_sigma=n_size, sigma_delta=sigma_delta, folder=folder)
    # line_arc.to_json(f'/mnt/c/muco/code/{folder}/line_{rcs_name}_normal_qp_{nb_arc_N}_arc.json')
    # fig.savefig(f'/mnt/c/muco/code/{folder}/twiss_{rcs_name}_normal_qp_{nb_arc_N}_arc.png')

    # COMPUTES SIGMA DELTA FROM CELL + CHECK WITH ARC BEAM DISTRIBUTION
    # delta3, delta6 =  compute_sigma_delta (line_FODO, rcs, n_turn_rcs)
    # print(delta3, delta6)
    # n_part = 20000
    # nemitt_x_0 = 25e-6
    # nemitt_y_0 = 25e-6
    # emit_s = 0.025
    # adapt_ho_delta=False
    # particles_optics = generate_particles(rcs = rcs, line = line_arc, tw = tw_arc, 
    #                                     n_part = n_part, n_turn_rcs = n_turn_rcs, 
    #                                     e_x = nemitt_x_0, e_y = nemitt_y_0, 
    #                                     e_s = emit_s, adapt=adapt_ho_delta)
    # delta3_bis = np.percentile(particles_optics.delta, 93.32)
    # delta6_bis = np.percentile(particles_optics.delta, 99.99966)
    # print(delta3_bis, delta6_bis)
