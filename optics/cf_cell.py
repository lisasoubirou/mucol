"""
Optimisation code for combined-function cells.
"""

import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import xpart as xp
import sympy as sp
import copy
from class_geometry.ring_geometry.ring_geometry import RingGeometry, load_file
from generate_particles import generate_particles
from pathlib import Path

plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

T_REF = 0.5

class CombinedFunctionCell:
    def __init__(
        self, 
        rcs, 
        Bmax, 
        mux, 
        muy, 
        sigma_delta, 
        n_sigma, 
        tr_emit_n, 
        dx
        ):

        self.rcs_scanB = copy.deepcopy(rcs)
        self.rcs_test = copy.deepcopy(rcs)
        self.Bmax = Bmax
        self.mux = mux
        self.muy = muy
        self.sigma_delta = sigma_delta
        self.n_sigma = n_sigma
        self.tr_emit_n = tr_emit_n
        self.inj_gamma = rcs.inj_gamma
        self.tot_arc_length = rcs.tot_arc_length
        self.Brho=rcs.ext_Brho
        self.dx = dx
        self.Bbend = rcs.dipoles['BNC'][1]
        self.res_Bscan = {}
        self.compute_u_cf()

    def compute_u_cf(self):
        uf, ud = sp.symbols('uf ud')
        eq1 = sp.Eq(1/2 * ( 2*sp.cos(uf/2)*sp.cosh(ud/2) + (ud**2 - uf**2)/uf/ud*sp.sin(uf/2)*sp.sinh(ud/2)), sp.cos(self.mux) )
        eq2 = sp.Eq(1/2 * ( 2*sp.cos(ud/2)*sp.cosh(uf/2) - (ud**2 - uf**2)/uf/ud*sp.sin(ud/2)*sp.sinh(uf/2)), sp.cos(self.muy) )
        sol = sp.nsolve((eq1, eq2), (uf, ud), (0.1, 0.1))
        self.uf = float(sol[0])
        self.ud = float(sol[1])

    @property
    def B_ext_min(self):
        return 2 * np.pi * rcs.ext_Brho / rcs.tot_arc_length / 0.95

    def scan_nc(self, rcs, n_cell = None):
        sol = {}
        if n_cell is not None:
            sol['n_cell'] = n_cell
        else:
            sol['n_cell'] = np.arange(100, rcs.nc_best, 1)
        sol['cell_length'] = rcs.tot_arc_length / sol['n_cell']
        sol['cell_angle'] = 2 * np.pi / sol['n_cell']

        sol['kf'] = self.uf**2 / (sol['cell_length'])**2
        sol['kd'] = self.ud**2 / (sol['cell_length'])**2
        sol['betx'] = self.compute_betaX_F(
            uf=self.uf, 
            ud=self.ud, 
            cell_length=sol['cell_length']
            )
        sol['dx'] = self.compute_dispX_F(
            uf=self.uf, 
            ud=self.ud, 
            cell_length=sol['cell_length'], 
            cell_angle=sol['cell_angle']
            )
        sol['beam_size'] = self.sigma_delta * sol['dx'] + self.n_sigma * np.sqrt(self.tr_emit_n / rcs.inj_gamma * sol['betx'])
        sol['B_dip'] = rcs.dipole_families['BNC']['B'](1) + sol['kf'] * rcs.ext_Brho * (sol['beam_size']+self.dx)

        return sol
    
    def make_cell(
            self, 
            rcs, 
            option: str = 'var_ref', 
            time: float = 0, 
            t_ref: float = T_REF, 
            hto0=False, 
            slice=True
            ):
        
        kf = self.uf**2 / (rcs.cell_length)**2
        kd = -self.ud**2 / (rcs.cell_length)**2
        cell = self.build_cell(
            rcs=rcs, 
            kf=kf, 
            kd=kd, 
            hto0=hto0, 
            option=option, 
            time=time, 
            t_ref=t_ref)
        if slice : 
            cell.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(11))])
        tw_sliced = cell.twiss4d()
        tw_sliced.plot()
        return cell

    def plot_scan_nc(self, rcs, n_cell = None):
        scan_nc = self.scan_nc(rcs, n_cell)

        plt.figure()
        plt.scatter(scan_nc['cell_length'], scan_nc['betx'], label='betx')
        plt.xlabel('Cell length [m]')
        plt.ylabel(r'$\beta_x$ [m]')
        # plt.legend()
        plt.show()

        plt.figure()
        plt.scatter(scan_nc['cell_length'], scan_nc['dx'], label='dx')
        plt.xlabel('Cell length [m]')
        plt.ylabel(r'$D_x$ [m]')
        # plt.legend()
        plt.show()

        plt.figure()
        plt.scatter(scan_nc['cell_length'], 2*np.array(scan_nc['beam_size'])*1e3, label=r'Total beam size')
        plt.scatter(scan_nc['cell_length'], 2*np.array(scan_nc['dx'])*SIGMA_DELTA*1e3, label=r'$D_x$ contribution')
        plt.legend()
        plt.xlabel('Cell length [m]')
        plt.ylabel(r'Beam size $\phi$ [mm]')
        plt.show()

        plt.figure()
        plt.scatter(scan_nc['cell_length'], np.array(scan_nc['B_dip']), label=r'$B_{pole}$')    
        # plt.legend()
        plt.xlabel('Cell length [m]')
        plt.ylabel('B Pole [T]')
        plt.show()

    def compare_th_optics(self, nb_point=10):
        scan_nc = self.scan_nc(n_cell=np.arange(100, self.rcs_test.nc_best, nb_point))
        comp_model = {'betx_optics':[], 'dx_optics':[], 'betx_th':[], 'dx_th':[], 'kf_match':[], 'kd_match':[]}
        comp_model.update(scan_nc)

        for id, nc in enumerate(comp_model['n_cell']):
            self.rcs_test.nb_cell_arc = nc
            k1 = comp_model['kf'][id]
            k2 = comp_model['kd'][id]
            cell_cf = self.build_cell(self.rcs_test, kf = k1, kd = -k2)

            cell_cf.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(11))])
            tw_sliced = cell_cf.twiss4d()

            match_tune(cell_cf, self.mux/2/np.pi, self.muy/2/np.pi, print_results=False)

            tw_sliced = cell_cf.twiss4d()
            # tw_sliced.plot()

            comp_model['betx_optics'].append(np.max(tw_sliced.betx))
            comp_model['dx_optics'].append(np.max(tw_sliced.dx))

            comp_model['kf_match'].append(cell_cf.vars['k1']._get_value())
            comp_model['kd_match'].append(-cell_cf.vars['k2']._get_value())
        
        plt.figure()
        plt.scatter(comp_model['cell_length'], comp_model['kf_match'], label=r'$k_f$ match')    
        plt.scatter(comp_model['cell_length'], comp_model['kd_match'], label=r'$k_d$ match')    
        plt.scatter(comp_model['cell_length'], comp_model['kd'], label=r'$k_d$ th') 
        plt.scatter(comp_model['cell_length'], comp_model['kf'], label=r'$k_f$ th')    
        plt.legend()
        plt.xlabel('Cell length [m]')
        plt.ylabel(r'Strength $k_1$ [$1/m^2$]')
        plt.show()

        plt.figure()
        plt.scatter(comp_model['cell_length'], (np.array(comp_model['betx'])/np.array(comp_model['betx_optics'])-1), label=r'$\frac{\beta_{model}-\beta_{optics}}{\beta_{optics}}$')
        plt.scatter(comp_model['cell_length'], (np.array(comp_model['dx'])/np.array(comp_model['dx_optics'])-1), label=r'$\frac{D_{model}-D_{optics}}{D_{optics}}$')
        plt.xlabel('Cell length [m]')
        plt.legend(loc='center right')
        plt.show()

        plt.figure()
        plt.scatter(comp_model['cell_length'], (np.array(comp_model['kf_match'])/np.array(comp_model['kf'])-1), label=r'$\frac{kf_{model}-kf_{optics}}{kf_{optics}}$')
        plt.scatter(comp_model['cell_length'], (np.array(comp_model['kd_match'])/np.array(comp_model['kd'])-1), label=r'$\frac{kd_{model}-kd_{optics}}{kd_{optics}}$')
        plt.xlabel('Cell length [m]')
        plt.legend(loc='lower left')
        plt.show()
        
        return comp_model

    def scan_Bext(self):
        B_ext_scan = np.linspace(self.B_ext_min, self.Bmax, 150)
        length_dip = 2 * np.pi * self.rcs_scanB.ext_Brho / B_ext_scan
        B_inj_scan = 2 * np.pi * self.rcs_scanB.inj_Brho / length_dip

        self.res_Bscan = {
            'nc_scan': [], 'cell_length_scan': [], 'betx_scan': [], 'dx_scan': [],
            'beam_size_scan': [], 'kinit_scan': [], 'B_pole_scan': [], 'B1_dip_poss': [], 'B2_dip_poss': []
        }

        for B1, B2 in zip(B_inj_scan, B_ext_scan):
            self.rcs_scanB.dipoles = {"BNC": [B1, B2, "linear"]}
            scan_nc = self.scan_nc(self.rcs_scanB)
            if np.min(scan_nc['B_dip']) < self.Bmax:
                mask = np.array(scan_nc['B_dip']) < self.Bmax
                nc_max = np.max(scan_nc['n_cell'][mask])
                index = np.where(scan_nc['n_cell'] == nc_max)[0][0]

                self.res_Bscan['B1_dip_poss'].append(B1)
                self.res_Bscan['B2_dip_poss'].append(B2)
                self.res_Bscan['nc_scan'].append(scan_nc['n_cell'][index])
                self.res_Bscan['cell_length_scan'].append(scan_nc['cell_length'][index])
                self.res_Bscan['betx_scan'].append(scan_nc['betx'][index])
                self.res_Bscan['dx_scan'].append(scan_nc['dx'][index])
                self.res_Bscan['beam_size_scan'].append(scan_nc['beam_size'][index])
                self.res_Bscan['kinit_scan'].append(scan_nc['kf'][index])
                self.res_Bscan['B_pole_scan'].append(scan_nc['B_dip'][index])

    def find_opt_cf(self, n_slice=11):
        lc_min = min(self.res_Bscan['cell_length_scan'])
        index_min = self.res_Bscan['cell_length_scan'].index(lc_min)
        nc_opt = self.res_Bscan['nc_scan'][index_min]
        self.rcs_opt = copy.deepcopy(self.rcs_scanB)
        self.rcs_opt.dipoles = {"BNC": [self.res_Bscan['B1_dip_poss'][index_min], self.res_Bscan['B2_dip_poss'][index_min], "linear"]}
        self.rcs_opt.nb_cell_arc = nc_opt

        # Assuming cell_combined_functions is defined elsewhere
        cell_cf = self.build_cell(self.rcs_opt, kf=self.uf**2 / (self.rcs_opt.cell_length)**2, kd=-self.ud**2 / (self.rcs_opt.cell_length)**2)
        cell_cf.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(n_slice))])
        tw_sliced = cell_cf.twiss4d()
        tw_sliced.plot()
        dico_opt = {'n_cell': nc_opt, 'B_inj': self.res_Bscan['B1_dip_poss'][index_min], 'B_ext:':self.res_Bscan['B2_dip_poss'][index_min],
                    'dx': self.res_Bscan['dx_scan'][index_min], 'betx': self.res_Bscan['betx_scan'][index_min], 
                    'beam_size': self.res_Bscan['beam_size_scan'][index_min], 'Bpole': self.res_Bscan['B_pole_scan'][index_min]}
        return dico_opt

    def compute_matrix_X_F(self, uf, ud, cell_length):
        """Cell transfer matrix in horizontal plane considering a focusing quadrupole component in the first dipole"""
        M1 = (ud**2 - uf**2) / (2 * uf * ud) * np.sinh(ud / 2) * np.sin(uf / 2) + np.cosh(ud / 2) * np.cos(uf/ 2)
        M2 = - cell_length * (ud**2 - uf**2) / (2 * ud * uf**2) * np.sinh(ud / 2) * np.cos(uf / 2) + (ud**2 + uf**2)*cell_length/ (2 * ud * uf**2) * np.sinh(ud / 2) + cell_length/uf * np.cosh(ud / 2) * np.sin(uf / 2)
        M3 = (ud**2 - uf**2) / (2 * cell_length * ud) * np.sinh(ud / 2) * np.cos(uf / 2) + (ud**2 + uf**2)/(2 * cell_length * ud) * np.sinh(ud / 2)  - uf/cell_length * np.cosh(ud / 2) * np.sin(uf / 2)
        M4 = M1
        return M1, M2, M3, M4
    
    def compute_tr_X(self, uf, ud):
        """Trace of the cell transfer matrix in horizontal plane"""
        return 2*((ud**2 - uf**2) / (2 * uf * ud) * np.sinh(ud / 2) * np.sin(uf / 2) + np.cosh(ud / 2) * np.cos(uf/ 2))

    def compute_matrix_Y_F(self, uf, ud, cell_length):
        """Cell transfer matrix in vertical plane considering a focusing quadrupole component in the first dipole"""
        M1 = -(ud**2 - uf**2) / (2 * uf * ud) * np.sin(ud / 2) * np.sinh(uf / 2) + np.cos(ud / 2) * np.cosh(uf/ 2)
        M2 = - cell_length * (ud**2 - uf**2) / (2 * ud * uf**2) * np.sin(ud / 2) * np.cosh(uf / 2) + (ud**2 + uf**2)*cell_length/ (2 * ud * uf**2) * np.sin(ud / 2)  + cell_length/uf * np.cos(ud / 2) * np.sinh(uf / 2)
        M3 = ud/cell_length*np.sin(ud/2)*( (uf/ud*np.sinh(uf/4))**2 - (np.cosh((uf/4)**2))) + uf/cell_length * np.cos(ud / 2) * np.sinh(uf / 2)
        M4 = M1
        return M1, M2, M3, M4
    
    def compute_tr_Y(self, uf, ud):
        return 2*(-(ud**2 - uf**2) / (2 * uf * ud) * np.sin(ud / 2) * np.sinh(uf / 2) + np.cos(ud / 2) * np.cosh(uf/ 2))
    
    def compute_betaX_F(self, uf, ud, cell_length):
        M2 = self.compute_matrix_X_F(uf=uf, ud=ud, cell_length=cell_length)[1]
        return M2/np.sin(self.compute_mu_x(uf, ud))

    def compute_betaY_F(self, uf, ud, cell_length):
        M2 = self.compute_matrix_Y_F(uf=uf, ud=ud, cell_length=cell_length)[1]
        return M2/np.sin(self.compute_mu_y(uf, ud))
    
    def compute_betaX_D(self, uf, ud, cell_length):
        return self.compute_betaY_F(uf=ud, ud=uf, cell_length=cell_length)
    
    def compute_betaY_D(self, uf, ud, cell_length):
        return self.compute_betaX_F(uf=ud, ud=uf, cell_length=cell_length)

    def compute_dispX_F(self, uf, ud, cell_length, cell_angle):
        d = cell_length*cell_angle/ud/uf**2*(ud**2*np.sinh(ud/4)*np.cos(uf/4) - uf**2*np.sinh(ud/4) 
            - uf*ud*np.cosh(ud/4)*np.sin(uf/4) - ud**2*np.sinh(ud/4))/(ud*np.sinh(ud/4)*np.cos(uf/4)-uf*np.cosh(ud/4)*np.sin(uf/4))
        return d

    def compute_dispX_D(self, uf, ud, cell_length, cell_angle):
        d = cell_length*cell_angle/ud**2/uf*( uf**2*np.cosh(ud/4) - uf*ud*np.sinh(ud/4)/np.tan(uf/4)
            - uf**2 - ud**2) / (ud*np.sinh(ud/4)/np.tan(uf/4) - uf*np.cosh(ud/4))
        return d
    
    def compute_mu_x(self, uf, ud):
        return np.arccos(1/2*self.compute_tr_X(uf, ud))

    def compute_mu_y(self, uf, ud):
        return np.arccos(1/2*self.compute_tr_Y(uf, ud))
    
    def compute_beam_sizeX(self, uf, ud, cell_length):
        disp = self.compute_dispX_F(uf, ud, cell_length, 2*np.pi*cell_length/self.tot_arc_length)
        beta = self.compute_betaX_F(uf, ud, cell_length)
        return self.sigma_delta*disp + self.n_sigma*np.sqrt(self.tr_emit_n / self.inj_gamma*beta) + self.dx
    
    def compute_beam_sizeY(self, uf, ud, cell_length):
        disp = 0
        beta = self.compute_betaY_D(uf, ud, cell_length)
        return self.sigma_delta*disp + self.n_sigma*np.sqrt(self.tr_emit_n / self.inj_gamma*beta) + self.dx
       
    def BpoleF(self, uf, ud, cell_length):
        beam_size = self.compute_beam_sizeX(uf, ud, cell_length)
        return self.Bbend + uf**2/cell_length**2 * self.Brho * beam_size
    
    def BpoleD(self, uf, ud, cell_length):
        beam_size = self.compute_beam_sizeY(uf, ud, cell_length)
        return self.Bbend + ud**2/cell_length**2 * self.Brho * beam_size
   
    def plot_scan_Bext(self):
        if self.res_Bscan == {}:
            self.scan_Bext()

        plt.figure()
        plt.scatter(self.res_Bscan['B2_dip_poss'], self.res_Bscan['cell_length_scan'])
        plt.ylabel('Cell length [m]')
        plt.xlabel(r'$B_{bending}$ [T]')
        plt.show()

        # plt.figure()
        # plt.scatter(self.res_Bscan['B2_dip_poss'], self.res_Bscan['dx_scan'], label='dx')
        # plt.xlabel(r'$B_{bending}$ [T]')
        # plt.ylabel(r'$D_x$ [m]')
        # # plt.legend()
        # plt.show()

        # plt.figure()
        # plt.scatter(self.res_Bscan['B2_dip_poss'], self.res_Bscan['betx_scan'], label='betx')
        # plt.xlabel(r'$B_{bending}$ [T]')
        # plt.ylabel(r'$\beta_x$ [m]')
        # # plt.legend()
        # plt.show()

        plt.figure()
        plt.scatter(self.res_Bscan['B2_dip_poss'], 2 * np.array(self.res_Bscan['beam_size_scan']) * 1e3, label=r'Total beam size')
        plt.scatter(self.res_Bscan['B2_dip_poss'], 2 * np.array(self.res_Bscan['dx_scan']) * self.sigma_delta * 1e3, label=r'$D_x$ contribution')
        plt.legend()
        plt.xlabel(r'$B_{bending}$ [T]')
        plt.ylabel(r'Beam size $\phi$ [mm]')
        plt.show()

        plt.figure()
        plt.scatter(self.res_Bscan['B2_dip_poss'], np.array(self.res_Bscan['B_pole_scan']), label=r'$B_{pole}$')
        plt.ylabel(r'$B_{pole}$ [T]')
        plt.xlabel(r'$B_{bending}$ [T]')
        plt.show()

    def build_cell(self, rcs, kf, kd, hto0=False, option: str = 'var_ref', time: float = 0, t_ref: float = T_REF):
        if len(rcs.pattern) > 1:
            print('CAREFUL, build_cell FUNCTION WORKS FOR PATTERN OF 1DIP ONLY')

        if hto0:
            fh = 0
        else:
            fh = 1

        env = xt.Environment()
        env.particle_ref = xp.Particles(
            p0c=rcs.injection_energy + (rcs.extraction_energy - rcs.injection_energy) * time,
            q0=1, mass0=xp.MUON_MASS_EV
        )

        lsss = rcs.length_short_insertion + rcs.L_extra_arc / rcs.nb_cell_arc / 2
        env.new('drift_sss', xt.Drift, length=lsss)
        # env.new('rot', xt.Bend, length=1e-5, k0=0, h=rcs.cell_angle/4e-5)

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
            if option == 'var_ref':
                cell_components.append(env.new(
                    item + f'{id}_1', 
                    xt.Bend, 
                    k0 = list_hn[id], 
                    angle = list_hn[id] * fh * L_dip_path[id] / 2,
                    length = L_dip_path[id] / 2, 
                    edge_entry_angle = epsilon[id] * 0,
                    edge_exit_angle = -epsilon[id + 1] * 0, 
                    k1 = kf 
                ))

            elif option == 'var_k':
                cell_components.append(env.new(
                    item + f'{id}_1', 
                    xt.Bend, 
                    k0 = list_hn[id], 
                    angle = list_hn_ref[id] * fh * L_dip_path_ref[id] / 2,
                    length = L_dip_path_ref[id] / 2, 
                    edge_entry_angle = eps_ref[id],
                    edge_exit_angle = -eps_ref[id + 1], 
                    k1 = kf
                ))
            else:
                raise ValueError(f'Invalid option: {option}')

        if hto0:    
            cell_components.append(env.new('rot1', 'rot'))

        cell_components.append(env.new(f'drift_1', xt.Drift, length=L_qd_path))
        cell_components.append(env.new('drift_sss.1', 'drift_sss'))
        cell_components.append(env.new(f'drift_2', xt.Drift, length=L_dd_path[0]))

        if hto0:
            cell_components.append(env.new('rot2', 'rot'))

        for id, item in enumerate(rcs.pattern):
            if option == 'var_ref':
                cell_components.append(env.new(
                    item + f'{id}_2', 
                    xt.Bend, 
                    k0 = list_hn[id], 
                    angle = list_hn[id] * fh * L_dip_path[id],
                    length = L_dip_path[id], 
                    edge_entry_angle = epsilon[id] * 0,
                    edge_exit_angle = -epsilon[id + 1] * 0, 
                    k1 = kd
                ))
            elif option == 'var_k':
                cell_components.append(env.new(
                    item + f'{id}_2', 
                    xt.Bend, 
                    k0 = list_hn[id], 
                    angle = list_hn_ref[id] * fh * L_dip_path_ref[id],
                    length = L_dip_path_ref[id], 
                    edge_entry_angle = eps_ref[id],
                    edge_exit_angle = -eps_ref[id + 1], 
                    k1 = kd
                ))
            else:
                raise ValueError(f'Invalid option: {option}')
        
        if hto0:
            cell_components.append(env.new('rot3', 'rot'))

        cell_components.append(env.new(f'drift_3', xt.Drift, length=L_qd_path))
        cell_components.append(env.new('drift_sss.2', 'drift_sss'))
        cell_components.append(env.new(f'drift_4', xt.Drift, length=L_dd_path[0]))

        if hto0:
            cell_components.append(env.new('rot4', 'rot'))

        for id, item in enumerate(rcs.pattern):
            if option == 'var_ref':
                cell_components.append(env.new(
                    item + f'{id}_3', 
                    xt.Bend, 
                    k0 = list_hn[id], 
                    angle = list_hn[id] * fh * L_dip_path[id] / 2,
                    length = L_dip_path[id] / 2, 
                    edge_entry_angle = epsilon[id] * 0,
                    edge_exit_angle = -epsilon[id + 1] * 0, 
                    k1 = kf
                ))
            elif option == 'var_k':
                cell_components.append(env.new(
                    item + f'{id}_3', 
                    xt.Bend, 
                    k0 = list_hn[id], 
                    angle = list_hn_ref[id] * fh * L_dip_path_ref[id] / 2,
                    length = L_dip_path_ref[id] / 2, 
                    edge_entry_angle = eps_ref[id],
                    edge_exit_angle = -eps_ref[id + 1], 
                    k1 = kf
                ))
            else:
                raise ValueError(f'Invalid option: {option}')

        cell = env.new_line(components=cell_components)
        return cell
    
def match_tune(line, qx, qy, print_results=False):
        line.vars['k1'] = line['BNC0_1'].k1
        line.vars['k2'] = line['BNC0_2'].k1
        line.element_refs['BNC0_1'].k1 = line.vars['k1']
        line.element_refs['BNC0_3'].k1 = line.vars['k1']
        line.element_refs['BNC0_2'].k1 = line.vars['k2']
        matcher = line.match(
            vary = [
                xt.VaryList(['k1', 'k2'], step=1e-8, tag='quad')
                ],
            targets=[xt.TargetSet(qx=qx, qy=qy, tol=1e-4)], 
            method='4d',
            solve=True
            )

        if print_results:
            print('RESULTS FROM TUNE MATCHING')
            matcher.target_status()
            matcher.vary_status()

class ModelCF:
    def __init__(
        self, 
        sigma_delta : float, 
        n_sigma : float, 
        tr_emit_n : float, 
        inj_gamma : float, 
        dx : float, 
        Brho : float, 
        Bext : float
        ):
        
        self.sigma_delta = sigma_delta  # Relative dispersion in energy
        self.n_sigma = n_sigma  # Number of sigma to compute beam size
        self.tr_emit_n = tr_emit_n  # Normalized trasnverse emittance
        self.inj_gamma = inj_gamma  # Gamma factor at injection
        self.dx = dx    # Radial margin for aperture
        self.Brho = Brho    # Magnetic rigidity at extraction energy
        self.Bext = Bext    # Bfield at extraction

    def compute_matrix_X_F(self, uf, ud, cell_length):
        """Cell transfer matrix in horizontal plane considering a focusing quadrupole component in the first dipole"""
        M1 = (ud**2 - uf**2) / (2 * uf * ud) * np.sinh(ud / 2) * np.sin(uf / 2) + np.cosh(ud / 2) * np.cos(uf/ 2)
        M2 = - cell_length * (ud**2 - uf**2) / (2 * ud * uf**2) * np.sinh(ud / 2) * np.cos(uf / 2) + (ud**2 + uf**2)*cell_length/ (2 * ud * uf**2) * np.sinh(ud / 2) + cell_length/uf * np.cosh(ud / 2) * np.sin(uf / 2)
        M3 = (ud**2 - uf**2) / (2 * cell_length * ud) * np.sinh(ud / 2) * np.cos(uf / 2) + (ud**2 + uf**2)/(2 * cell_length * ud) * np.sinh(ud / 2)  - uf/cell_length * np.cosh(ud / 2) * np.sin(uf / 2)
        M4 = M1
        return M1, M2, M3, M4
    
    def compute_tr_X(self, uf, ud):
        """Trace of the cell transfer matrix in horizontal plane"""
        return 2*((ud**2 - uf**2) / (2 * uf * ud) * np.sinh(ud / 2) * np.sin(uf / 2) + np.cosh(ud / 2) * np.cos(uf/ 2))

    def compute_matrix_Y_F(self, uf, ud, cell_length):
        """Cell transfer matrix in vertical plane considering a focusing quadrupole component in the first dipole"""
        M1 = -(ud**2 - uf**2) / (2 * uf * ud) * np.sin(ud / 2) * np.sinh(uf / 2) + np.cos(ud / 2) * np.cosh(uf/ 2)
        M2 = - cell_length * (ud**2 - uf**2) / (2 * ud * uf**2) * np.sin(ud / 2) * np.cosh(uf / 2) + (ud**2 + uf**2)*cell_length/ (2 * ud * uf**2) * np.sin(ud / 2)  + cell_length/uf * np.cos(ud / 2) * np.sinh(uf / 2)
        M3 = ud/cell_length*np.sin(ud/2)*( (uf/ud*np.sinh(uf/4))**2 - (np.cosh((uf/4)**2))) + uf/cell_length * np.cos(ud / 2) * np.sinh(uf / 2)
        M4 = M1
        return M1, M2, M3, M4

    def compute_tr_Y(self, uf, ud):
        """Trace of the cell transfer matrix in vertical plane"""
        return 2*(-(ud**2 - uf**2) / (2 * uf * ud) * np.sin(ud / 2) * np.sinh(uf / 2) + np.cos(ud / 2) * np.cosh(uf/ 2))
    
    def compute_betaX_F(self, uf, ud, cell_length):
        """Beta X function in the middle of the dipole with focusing gradient"""
        M2 = self.compute_matrix_X_F(uf=uf, ud=ud, cell_length=cell_length)[1]
        return M2/np.sin(self.compute_mu_x(uf, ud))

    def compute_betaY_F(self, uf, ud, cell_length):
        """Beta Y function in the middle of the dipole with focusing gradient"""
        M2 = self.compute_matrix_Y_F(uf=uf, ud=ud, cell_length=cell_length)[1]
        return M2/np.sin(self.compute_mu_y(uf, ud))
    
    def compute_betaX_D(self, uf, ud, cell_length):
        """Beta X function in the middle of the dipole with defocusing gradient"""
        return self.compute_betaY_F(uf=ud, ud=uf, cell_length=cell_length)
    
    def compute_betaY_D(self, uf, ud, cell_length):
        """Beta Y function in the middle of the dipole with defocusing gradient"""
        return self.compute_betaX_F(uf=ud, ud=uf, cell_length=cell_length)

    def compute_dispX_F(self, uf, ud, cell_length, cell_angle):
        """Dispersion X in the middle of the dipole with focusing gradient"""
        d = cell_length*cell_angle/ud/uf**2*(ud**2*np.sinh(ud/4)*np.cos(uf/4) - uf**2*np.sinh(ud/4) 
            - uf*ud*np.cosh(ud/4)*np.sin(uf/4) - ud**2*np.sinh(ud/4))/(ud*np.sinh(ud/4)*np.cos(uf/4)-uf*np.cosh(ud/4)*np.sin(uf/4))
        return d

    def compute_dispX_D(self, uf, ud, cell_length, cell_angle):
        """Dispersion X in the middle of the dipole with defocusing gradient"""
        d = cell_length*cell_angle/ud**2/uf*( uf**2*np.cosh(ud/4) - uf*ud*np.sinh(ud/4)/np.tan(uf/4)
            - uf**2 - ud**2) / (ud*np.sinh(ud/4)/np.tan(uf/4) - uf*np.cosh(ud/4))
        return d
    
    def compute_mu_x(self, uf, ud):
        """Horizontal phase advance"""
        return np.arccos(1/2*self.compute_tr_X(uf, ud))

    def compute_mu_y(self, uf, ud):
        """Vertical phase advance"""
        return np.arccos(1/2*self.compute_tr_Y(uf, ud))
    
    def compute_beam_sizeX(self, uf, ud, cell_length, cell_angle):
        """Horizontal radial beam size + margin"""
        disp = self.compute_dispX_F(uf, ud, cell_length, cell_angle)
        beta = self.compute_betaX_F(uf, ud, cell_length)
        return self.sigma_delta*disp + self.n_sigma*np.sqrt(self.tr_emit_n / self.inj_gamma*beta) + self.dx
    
    def compute_beam_sizeY(self, uf, ud, cell_length):
        """Vertical radial beam size + margin"""
        disp = 0
        beta = self.compute_betaY_D(uf, ud, cell_length)
        return self.sigma_delta*disp + self.n_sigma*np.sqrt(self.tr_emit_n / self.inj_gamma*beta) + self.dx
       
    def BpoleF(self, uf, ud, cell_length, cell_angle):
        """Field on the pole of the dipole with focusing gradient"""
        beam_size = self.compute_beam_sizeX(uf, ud, cell_length, cell_angle)
        return self.Bext + uf**2/cell_length**2 * self.Brho * beam_size
    
    def BpoleD(self, uf, ud, cell_length):
        """Field on the pole of the dipole with defocusing gradient"""
        beam_size = self.compute_beam_sizeY(uf, ud, cell_length)
        return self.Bext + ud**2/cell_length**2 * self.Brho * beam_size
    
class OptmizeMUCF:
    def __init__(
        self, 
        rcs : RingGeometry, 
        Bmax : float, 
        sigma_delta : float, 
        n_sigma : float, 
        tr_emit_n : float, 
        dx : float, 
        n_point : int = 1000, 
        disp_max : float= 1, 
        limit_mux : float = 0.33, 
        limit_muy : float = 0.1, 
        plot_stab : bool = False, 
        plot : bool = False, 
        print_min : bool = False
        ):
        
        self.B1 = rcs.dipoles['BNC'][0]
        self.B2 = rcs.dipoles['BNC'][1]
        self.n_point = n_point  # Number of point of the phase advance grid to scan
        self.disp_max = disp_max    # Maximum dispersion function to filter
        self.limit_mux = limit_mux # Maximum horizontal phase advance to consider
        self.limit_muy = limit_muy  # Maximum vertical phase advance to consider
        self.Bmax = Bmax    # Maximum field on the dipole pole
        self.sigma_delta = sigma_delta  # Relative dispersion in energy
        self.n_sigma = n_sigma  # Number of sigma to compute beam size
        self.tr_emit_n = tr_emit_n  # Normalized transverse emittance
        self.dx = dx    # Radial aperture margin
        self.inj_gamma = rcs.inj_gamma  # Gamma factor at injection
        self.tot_arc_length = rcs.tot_arc_length    #T otal arc length
        self.ext_Brho = rcs.ext_Brho    # Magnetic rigidity at extraction
        self.inj_Brho = rcs.inj_Brho    # Magnetic rigidity at injection
        self.B_ext = rcs.dipoles['BNC'][1]  # Dipole magnetic field at extraction

        self.mod = ModelCF(
            sigma_delta = self.sigma_delta, 
            n_sigma = self.n_sigma, 
            tr_emit_n = self.tr_emit_n, 
            inj_gamma = self.inj_gamma, 
            dx = self.dx, 
            Brho = self.ext_Brho, 
            Bext = self.B_ext)

        self.plot_stab = plot_stab # Plot stable uf/ud and mux/muy
        self.plot = plot    # Plot scan results
        self.print_min = print_min  # Print minimum Dx results

        self.l_cell = rcs.tot_arc_length/np.arange(100, rcs.nc_best, 10) # Cell length to scan 
        
        self.opt_mu()

    def opt_mu(self):
        self.grid_u_stable()
        self.scan_uf_ud_lc()
        self.filter_res()

    def grid_u_stable(self):
        """Computes (uf, ud) and (mux, muy) for which the lattice is stable"""
        u_max=4
        uf_list = np.linspace(1e-3, u_max, self.n_point)
        ud_list = np.linspace(1e-3, u_max, self.n_point)

        uf_gr, ud_gr = np.meshgrid(uf_list, ud_list)
        uf = ud_gr.ravel()
        ud = uf_gr.ravel()

        MX_values = self.mod.compute_tr_X(uf, ud)
        MY_values = self.mod.compute_tr_Y(uf, ud)
        mask_stab = (np.abs(MX_values) <= 2) & (np.abs(MY_values) <= 2)
        uf_stable = uf[mask_stab]
        ud_stable = ud[mask_stab]

        if self.plot_stab:
            print('Plot stability region')
            mux_stable = self.mod.compute_mu_x(uf_stable, ud_stable)
            muy_stable = self.mod.compute_mu_y(uf_stable, ud_stable)

            # Plot
            self.fig_stab_u = plt.figure(figsize=(6.4, 4.8))
            plt.scatter(uf_stable, ud_stable, color="red", s=2, alpha=0.6)  
            plt.xlabel(r"$u_f$")
            plt.ylabel(r"$u_d$")
            plt.title(r"Stable combination of ($u_f$, $u_d$)")
            plt.show()

            self.fig_stab_k = plt.figure(figsize=(6.4, 4.8))
            plt.scatter(mux_stable/2/np.pi, muy_stable/2/np.pi, color="red", s=2, alpha=0.6)  
            plt.xlabel(r"$\mu_x/ \ 2\pi$")
            plt.ylabel(r"$\mu_y/ \ 2\pi$")
            plt.title(r"Stable combination of ($\mu_x/2\pi$, $\mu_y/2\pi$)")
            plt.show()

        self.uf_stable = uf_stable # List of stable coordinate uf
        self.ud_stable = ud_stable # List of stable coordinate ud
    
    def scan_uf_ud_lc(self):
        """Scan on uf, ud and cell lengths and computes parameters: dispersion, beta, beam sizes,
        phase advances, pole field. """
        l_cell_gr = np.tile(self.l_cell, (len(self.uf_stable), 1))

        self.uf_scan = np.repeat(self.uf_stable, (len(self.l_cell)))
        self.ud_scan = np.repeat(self.ud_stable, (len(self.l_cell)))
        self.l_cell_scan = l_cell_gr.ravel()
        cell_angle_scan = 2*np.pi*self.l_cell_scan/self.tot_arc_length
        self.dispx_scan = self.mod.compute_dispX_F(uf=self.uf_scan, ud=self.ud_scan, cell_length=self.l_cell_scan, cell_angle=cell_angle_scan)
        self.betax_scan = self.mod.compute_betaX_F(uf=self.uf_scan, ud=self.ud_scan, cell_length=self.l_cell_scan)
        self.betay_scan = self.mod.compute_betaY_D(uf=self.uf_scan, ud=self.ud_scan, cell_length=self.l_cell_scan)
        self.beamsizex_scan = self.mod.compute_beam_sizeX(uf=self.uf_scan, ud=self.ud_scan, cell_length=self.l_cell_scan, cell_angle=cell_angle_scan)
        self.beamsizey_scan = self.mod.compute_beam_sizeY(uf=self.uf_scan, ud=self.ud_scan, cell_length=self.l_cell_scan)
        self.mux_scan = self.mod.compute_mu_x(uf=self.uf_scan, ud=self.ud_scan)
        self.muy_scan = self.mod.compute_mu_y(uf=self.uf_scan, ud=self.ud_scan)
        self.bpF_scan = self.mod.BpoleF(uf=self.uf_scan, ud=self.ud_scan, cell_length=self.l_cell_scan, cell_angle=cell_angle_scan)
        self.bpD_scan = self.mod.BpoleD(uf=self.uf_scan, ud=self.ud_scan, cell_length=self.l_cell_scan)

    def apply_mask(self, mask):
        """Filter the data for a given mask"""
        self.uf_scan = self.uf_scan[mask]
        self.ud_scan = self.ud_scan[mask]
        self.l_cell_scan = self.l_cell_scan[mask]
        self.dispx_scan = self.dispx_scan[mask]
        self.betax_scan = self.betax_scan[mask]
        self.betay_scan = self.betay_scan[mask]
        self.beamsizex_scan = self.beamsizex_scan[mask]
        self.beamsizey_scan = self.beamsizey_scan[mask]
        self.mux_scan = self.mux_scan[mask]
        self.muy_scan = self.muy_scan[mask]
        self.bpF_scan = self.bpF_scan[mask]
        self.bpD_scan = self.bpD_scan[mask]  

    def filter_res(self):
        """ Filter the results on criteria: Field on the pole (focusing and defocusing),
        Dispersion inferior to a maximum value, Phase advances to be limited to a given range.
        Plot the result of the scan if self.plot = True.
        """
        
        mask_Bp = (self.bpF_scan < self.Bmax) & (self.bpD_scan < self.Bmax) & (self.dispx_scan < self.disp_max) 
        self.apply_mask(mask_Bp)

        if self.plot:
            print('Plot results with constraints applied')
            self.scan_k_L_D = plt.figure(figsize=(6.4,4.8))
            plt.scatter(
                self.mux_scan / 2 / np.pi, 
                self.muy_scan / 2 / np.pi, 
                c=self.dispx_scan, 
                cmap="viridis", 
                alpha=0.7,
                edgecolors='none'
                )
            plt.colorbar(label=r"$D_x$ [m]")
            plt.xlabel(r"$\mu_x/ \ 2\pi$")
            plt.ylabel(r"$\mu_y/ \ 2\pi$")
            # plt.title(f"Scan $u_f$, $u_d$ and $L_c$ \n $B_{{inj}}$ = {self.B1:.2f}T, $B_{{ext}}$ = {self.B2:.2f}T")
            plt.tight_layout()
            plt.show()

        mask_mu = (self.mux_scan / 2 / np.pi < self.limit_mux) & (self.muy_scan / 2 / np.pi > self.limit_muy)
        self.apply_mask(mask_mu)

        if np.any(self.dispx_scan):
            index_min_limitMU = np.where(self.dispx_scan == np.min(self.dispx_scan))[0][0]

            lc_min = self.l_cell_scan[index_min_limitMU]
            mask_lc_min = (self.l_cell_scan == lc_min)
            
            self.uf_opt = self.uf_scan[index_min_limitMU]
            self.ud_opt = self.ud_scan[index_min_limitMU]
            self.l_cell_opt = self.l_cell_scan[index_min_limitMU]
            self.dispx_opt = self.dispx_scan[index_min_limitMU]
            self.betax_opt = self.betax_scan[index_min_limitMU]
            self.betay_opt = self.betay_scan[index_min_limitMU]
            self.beamsizex_opt = self.beamsizex_scan[index_min_limitMU]
            self.beamsizey_opt = self.beamsizey_scan[index_min_limitMU]
            self.mux_opt = self.mux_scan[index_min_limitMU]
            self.muy_opt = self.muy_scan[index_min_limitMU]
            self.bpF_opt = self.bpF_scan[index_min_limitMU]
            self.bpD_opt = self.bpD_scan[index_min_limitMU]
            self.nc_opt = int(self.tot_arc_length/self.l_cell_opt)

            if self.plot:
                print(f'Plot for L={lc_min}, Bext={self.B2}, Binj={self.B1}')
                print('D_x')
                self.scan_k_optL_D = plt.figure(figsize=(6.4,4.8))
                plt.scatter(
                    self.mux_scan[mask_lc_min]/2/np.pi, 
                    self.muy_scan[mask_lc_min]/2/np.pi, 
                    c=self.dispx_scan[mask_lc_min], 
                    cmap="viridis", 
                    edgecolors='none')
                plt.colorbar(label=r"$D_x$ [m]")
                plt.xlabel(r"$\mu_x/ \ 2\pi$")
                plt.ylabel(r"$\mu_y/ \ 2\pi$")
                # plt.title(f"Scan $u_f$ and $u_d$ for $L_c$ ={lc_min:.1f}m \n $B_{{inj}}$ = {self.B1:.2f}T, $B_{{ext}}$ = {self.B2:.2f}T")
                plt.tight_layout()
                plt.show()

                print('Pole field')
                self.scan_k_optL_Bpole = plt.figure(figsize=(6.4,4.8))
                plt.scatter(
                    self.mux_scan[mask_lc_min]/2/np.pi, 
                    self.muy_scan[mask_lc_min]/2/np.pi, 
                    c=self.bpF_scan[mask_lc_min], 
                    cmap="viridis", 
                    edgecolors='none'
                    )
                plt.colorbar(label="Pole field [T]")
                plt.xlabel(r"$\mu_x/ \ 2\pi$")
                plt.ylabel(r"$\mu_y/ \ 2\pi$")
                # plt.title(f"Scan $u_f$ and $u_d$ for $L_c$ ={lc_min:.1f}m \n $B_{{inj}}$ = {self.B1:.2f}T, $B_{{ext}}$ = {self.B2:.2f}T")
                plt.tight_layout()
                plt.show()

                print('Beam size')
                self.scan_k_optL_beamsize = plt.figure(figsize=(6.4,4.8))
                plt.scatter(
                    self.mux_scan[mask_lc_min]/2/np.pi, 
                    self.muy_scan[mask_lc_min]/2/np.pi, 
                    c=self.beamsizex_scan[mask_lc_min]*1e3, 
                    cmap="viridis", 
                    edgecolors='none'
                    )
                plt.colorbar(label="Beam size [mm]")
                plt.xlabel(r"$\mu_x/ \ 2\pi$")
                plt.ylabel(r"$\mu_y/ \ 2\pi$")
                # plt.title(f"Scan $u_f$ and $u_d$ for $L_c$ ={lc_min:.1f}m \n $B_{{inj}}$ = {self.B1:.2f}T, $B_{{ext}}$ = {self.B2:.2f}T")
                plt.tight_layout()
                plt.show()  

            if self.print_min:
                self.print_opt()
        
        else:
            print(f'Solution does not exist for Binj={self.B1}T, Bext={self.B2}T')
            self.uf_opt = np.nan
            self.ud_opt = np.nan
            self.l_cell_opt = np.nan
            self.dispx_opt = np.nan
            self.betax_opt = np.nan
            self.betay_opt = np.nan
            self.beamsizex_opt = np.nan
            self.beamsizey_opt = np.nan
            self.mux_opt = np.nan
            self.muy_opt = np.nan
            self.bpF_opt = np.nan
            self.bpD_opt = np.nan
            self.nc_opt = np.nan

    def print_opt(self):
        print('Disp x', self.dispx_opt, '[m]')
        print('Beam size radius + margin', self.beamsizex_opt, '[m]')
        print('Beta x ', self.betax_opt, '[m]')
        print('Beta y', self.betay_opt, '[m]')
        print('Mu x', self.mux_opt/2/np.pi)
        print('Mu y', self.muy_opt/2/np.pi)
        print('uf', self.uf_opt)
        print('ud', self.ud_opt)
        print('Cell length', self.l_cell_opt, '[m]')
        print('Bpole F', self.bpF_opt, 'T')
        print('Bpole D', self.bpD_opt, 'T')

def aperture_optics(tw, sigma_delta, n_sigma, tr_emit_n, inj_gamma, dx):
    """Computes the aperture based on a Twiss table. 

    Args:
        tw (_type_): Twiss Table.
        sigma_delta (_type_): Relative energy dispersion.
        n_sigma (_type_): Number of sigma to compute the beam size.
        tr_emit_n (_type_): Normalized transver emittance.
        inj_gamma (_type_): Gamma factor at injection.
        dx (_type_): Radial aperture margin. 

    Returns:
        aperture_x, aperture_y: apertures in horizontal and vertical plane 
    """
    beta_x = np.max(tw.betx)
    disp_x = np.max(tw.dx)
    beam_size_x = sigma_delta * disp_x + n_sigma * np.sqrt(tr_emit_n / inj_gamma * beta_x)
    aperture_x = 2 * beam_size_x + 2 * dx

    beta_y = np.max(tw.bety)
    disp_y = np.max(tw.dy)
    beam_size_y = sigma_delta * disp_y + n_sigma * np.sqrt(tr_emit_n / inj_gamma * beta_y)
    aperture_y = 2 * beam_size_y + 2 * dx

    return aperture_x, aperture_y

def compute_sigma_delta (line, rcs, n_turn_rcs):
    """Computes relative energy dispersion for a given CELL. DOES NOT WORK FOR AN ARC.

    Args:
        line (_type_): Line object
        rcs (_type_): Geometry object
        n_turn_rcs (_type_): number of turns in the rcs

    Returns:
        delta3, delta6: delta at 3sigma, delta at 6sigma
    """
    tw1 = line.twiss4d()
    mcf_cell = tw1.momentum_compaction_factor*rcs.filling_ratio

    n_turns = n_turn_rcs * rcs.nb_arc * rcs.nb_cell_arc
    energy_increment_per_arc = (rcs.extraction_energy - rcs.injection_energy) / n_turns
    sync_phase = 45
    phase = 180 - sync_phase
    volt_arc = energy_increment_per_arc / np.sin(phase * np.pi / 180)
    frequency = 1.3e9

    segment1 = xt.LineSegmentMap(
                            length = rcs.arc_length,
                            qx = tw1.qx, qy = tw1.qy, 
                            betx = tw1.betx[0], bety = tw1.bety[0],
                            alfx = tw1.alfx[0], alfy = tw1.alfy[0], 
                            dx = tw1.dx[0], dpx = tw1.dpx[0], 
                            dy = tw1.dy[0], dpy = tw1.dpy[0],
                            x_ref = tw1.x[0], px_ref = tw1.px[0],
                            y_ref = tw1.y[0], py_ref = tw1.py[0],
                            longitudinal_mode = 'nonlinear',
                            momentum_compaction_factor = mcf_cell,
                            voltage_rf = volt_arc,
                            frequency_rf = frequency,
                            lag_rf = phase,
                            energy_ref_increment = energy_increment_per_arc,
                            energy_increment = energy_increment_per_arc
                            )

    seg_line1 = xt.Line(elements = [segment1],
                    element_names = ['seg'])
    seg_line1.particle_ref = xt.Particles(p0c=rcs.injection_energy, mass0=xp.MUON_MASS_EV)
    tw_seg1 = seg_line1.twiss(method='6d', matrix_stability_tol=0.07, compute_chromatic_properties=True)

    n_part = 20000
    nemitt_x_0 = 25e-6
    nemitt_y_0 = 25e-6
    emit_s = 0.025
    adapt_ho_delta = False

    particles = generate_particles(
        rcs = rcs, 
        line = seg_line1, 
        tw = tw_seg1, 
        n_part = n_part, 
        n_turn_rcs = n_turn_rcs, 
        e_x = nemitt_x_0,
        e_y = nemitt_y_0, 
        e_s = emit_s,
        adapt=adapt_ho_delta
        )

    delta1 = np.std(particles.delta)
    delta3 = delta1*3
    delta6 = delta1*6
     
    return delta3, delta6

if __name__ == '__main__':
    n_size = 3
    rcs_name = 'RCS1_gf'

    path = Path(__file__).resolve().parent.parent

    input = {
        'RCS1_gf': {
            'file': path / 'class_geometry/parameter_files/para_RCS_LE_cf.txt',
            'sig_delta_3': 0.04032127,
            'sig_delta_6': 0.07846999,
            'n_turn_rcs': 17
        },
        'RCS1_CERN': {
            'file': path / 'class_geometry/parameter_files/para_RCS_CERN_RCS1.txt',
            'sig_delta_3': 0.04256914,
            'sig_delta_6': 0.08513828,
            'n_turn_rcs': 19
        },
        'RCS2_CERN': {
            'file': path / 'class_geometry/parameter_files/para_RCS_CERN_RCS2.txt',
            'sig_delta_3': 0.01539568,
            'sig_delta_6': 0.03079135,
            'n_turn_rcs': 29
        }
    }
        
    file_input = input[rcs_name]['file']
    SIGMA_DELTA = input[rcs_name]['sig_delta_'+f'{n_size}']
    N_TURN_RCS = input[rcs_name]['n_turn_rcs']
    TR_EMIT_N = 25e-6
    N_SIGMA = 3
    DX = 10e-3

    para = load_file(file_input)
    para['nb_arc'] = 1
    para['length_short_insertion'] = 0.5
    para['nb_cell_arc'] = 100
    # para['pattern'] = ['BNC', 'BNC']

    rcs = RingGeometry(file_input = para)

    mux = np.pi / 2
    muy = np.pi / 2
    qx = mux / 2 / np.pi
    qy = muy / 2 / np.pi
    
    Bmax = 1.8

    n_point = 1000
    disp_max = 1
    limit_mux = 0.333
    limit_muy = 0.11

    dx_min_list = []
    B_min = np.round(2 * np.pi * rcs.ext_Brho / rcs.tot_arc_length / 0.95, 2) 
    nB = 11
    B_ext_list = np.linspace(1.4, 1.5, nB)
    length_dip = 2 * np.pi * rcs.ext_Brho / B_ext_list
    B_inj_list = 2 * np.pi * rcs.inj_Brho / length_dip

    for B_ext, B_inj in zip(B_ext_list, B_inj_list):
        print(B_ext)
        rcs.dipoles =  {"BNC": [B_inj, B_ext, "linear"]}
        opt_cf = OptmizeMUCF(
            rcs = rcs, 
            Bmax = Bmax, 
            sigma_delta = SIGMA_DELTA, 
            n_sigma = N_SIGMA, 
            tr_emit_n = TR_EMIT_N, 
            dx = DX, 
            n_point = 1000, 
            disp_max = disp_max, 
            limit_mux = limit_mux, 
            limit_muy = limit_muy, 
            plot = False, 
            print_min = False
            )
        dx_min_list.append(opt_cf.dispx_opt)

    fig_dx_Bext = plt.figure(figsize=(6.4,4.8))
    plt.scatter(B_ext_list, dx_min_list)
    plt.xlabel(r'$B_{ext}$ [T]')
    plt.ylabel(r'Min $D_x$ [m]')
    plt.tight_layout()
    plt.show()

    index_min = np.where(dx_min_list == np.min(dx_min_list))
    B_ext_opt = B_ext_list[np.where(dx_min_list == np.nanmin(dx_min_list))[0][0]]
    B_inj_opt = B_inj_list[np.where(dx_min_list == np.nanmin(dx_min_list))[0][0]]
    Dx_opt = dx_min_list[np.where(dx_min_list == np.nanmin(dx_min_list))[0][0]]

    rcs.dipoles =  {"BNC": [B_inj_opt, B_ext_opt, "linear"]}
    opt_cf = OptmizeMUCF(
        rcs = rcs, 
        Bmax = Bmax, 
        sigma_delta = SIGMA_DELTA, 
        n_sigma = N_SIGMA, 
        tr_emit_n = TR_EMIT_N, 
        dx = DX, 
        n_point = 1000*2, 
        disp_max = disp_max, 
        limit_mux = limit_mux, 
        limit_muy = limit_muy, 
        plot = True, 
        print_min = True,
        plot_stab = True
        )

    rcs.nb_cell_arc=opt_cf.nc_opt
    rcs.nb_arc=1
    CF = CombinedFunctionCell(
        rcs = rcs, 
        Bmax = Bmax, 
        mux = opt_cf.mux_opt, 
        muy = opt_cf.muy_opt, 
        sigma_delta = SIGMA_DELTA, 
        n_sigma = N_SIGMA, 
        tr_emit_n = TR_EMIT_N, 
        dx = DX
        )
    CF.plot_scan_nc(rcs)
    
    line=CF.make_cell(
        rcs = rcs, 
        hto0 = False, 
        slice = False
        )
    tw=line.twiss4d()

    mcf_cf = tw.momentum_compaction_factor*rcs.filling_ratio
    l_dip = rcs.dipole_families['BNC']['length']
    foc_gradient = CF.uf**2 / opt_cf.l_cell_opt**2 * rcs.ext_Brho
    defoc_gradient = CF.ud**2 / opt_cf.l_cell_opt**2 * rcs.ext_Brho
    hor_apert_diam=opt_cf.beamsizex_opt * 2 * 1e3
    vert_apert_diam=opt_cf.beamsizey_opt * 2 * 1e3

    sigma3, sigma6 = compute_sigma_delta(
        line=line, 
        rcs = rcs, 
        n_turn_rcs = N_TURN_RCS
        )

    hor_apert_twiss, vert_apert_twiss = aperture_optics(
        tw = tw, 
        sigma_delta = sigma3, 
        n_sigma = N_SIGMA, 
        tr_emit_n = TR_EMIT_N, 
        inj_gamma = rcs.inj_gamma, 
        dx = DX
        )

    mu_x, mu_y = tw.mux[-1]*360, tw.muy[-1]*360
    print(f'Arc: na = {rcs.nb_arc}')
    print(f'Cell per arc: nc = {rcs.nb_cell_arc}')
    print(f'Cell length: lc = {opt_cf.l_cell_opt:.1f}m')
    print(f'Dipole bending field: B_inj = {B_inj_opt:.2f}T, B_ext={B_ext_opt:.2f}T')
    print(f'Dipole length: L_DIP_NC = {l_dip:.1f}m')
    print(f'Focusing gradient: G_f = {foc_gradient:.1f}T/m')
    print(f'Focusing Bpole: B_pole = {opt_cf.bpF_opt:.1f}T')
    print(f'Defocusing gradient: G_d = {defoc_gradient:.1f}T/m')
    print(f'Defocusing Bpole: B_pole = {opt_cf.bpD_opt:.1f}T')
    print(f'Horizontal dipole aperture (diam): aperture_hor = {hor_apert_diam:.1f}mm, apert_hor_twiss = {hor_apert_twiss*1e3:.1f}mm')
    print(f'Vertical dipole aperture (diam): aperture_vert = {vert_apert_diam:.1f}mm, apert_vert_twiss = {vert_apert_twiss*1e3:.1f}mm')
    print(f'Momentum compaction factor: mcf = {mcf_cf * 1e4:.2f} 1e-4')
    print(f'Phase advance: mux = {mu_x:.1f}°, muy = {mu_y:.1f}°')

    parameter_name = [
        'Circumference [m]', 
        'Injection energy [GeV]', 
        'Extraction energy [GeV]', 
        'Injection Bending Field [T]',
        'Extraction Bending Field [T]', 
        'Interconnection length [m]',
        'Short Insertion Length  [m]',
        'Normalized transverse emittance [mm.mrad]',   
        r'Beam size [$\sigma$]', 
        'Relative energy dispersion', 
        'Aperture margin (diam) [mm]', 
        'Number of arc', 
        'Number of cells per arc', 
        'Cell length [m]', 
        'Length NC Dip. [m]', 
        'Focusing Gradient [T/m]', 
        'Focusing Pole Field [T]', 
        'Defocusing Gradient [T/m]', 
        'Defocusing Pole Field [T]', 
        'Hor. Dipole Aperture (diam) [mm]', 
        'Vert. Dipole Aperture (diam) [mm]',
        'Momentum Compaction Factor [1e-4]', 
        f'Phase advance $\mu_x$ [°]', 
        f'Phase advance $\mu_y$ [°]'
        ]

    results_CF = [
            rcs.circumference,
            f'{rcs.injection_energy*1e-9:.0f}',
            f'{rcs.extraction_energy*1e-9:.0f}',
            f'{B_inj_opt:.2f}',
            f'{B_ext_opt:.2f}',
            f'{rcs.dipole_spacing:.1f}',
            f'{rcs.length_short_insertion:.1f}',
            f'{TR_EMIT_N*1e6}',
            f'{N_SIGMA}',
            f'{sigma3:.2}',
            f'{DX*2*1e3}',
            int(rcs.nb_arc),
            rcs.nb_cell_arc, 
            f'{opt_cf.l_cell_opt:.1f}', 
            f'{l_dip:.2f}', 
            f'{foc_gradient:.2f}', 
            f'{opt_cf.bpF_opt:.1f}',
            f'{defoc_gradient:.1f}',
            f'{opt_cf.bpD_opt:.1f}',
            f'{hor_apert_twiss*1e3:.1f}', 
            f'{vert_apert_twiss*1e3:.1f}', 
            f'{mcf_cf * 1e4:.2f}',
            f'{mu_x:.1f}',
            f'{mu_y:.1f}'
        ]

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
    
    from plot_twiss_strength import plot_cell_k_twiss, plot_enveloppe, beam_optics
    figsize = (6.4,5)
    line_sliced = line.copy()
    line_sliced.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(10))])
    myTwiss = line_sliced.twiss4d()
    fig_cell = plot_cell_k_twiss(line, myTwiss, figsize=figsize)
    # sys.exit(0)
    # Assuming beamX and beamY contain beam sizes at each s position
    beamX, beamY = beam_optics(
        tw = myTwiss, 
        sigma_delta = SIGMA_DELTA, 
        n_sigma = N_SIGMA, 
        tr_emit_n = TR_EMIT_N, 
        inj_gamma = rcs.inj_gamma)
    
    fig_env = plot_enveloppe(myTwiss, beamX, beamY, figsize=(6.4,5.2))

    folder = '/mnt/c/muco/Results/figures/chap5/CF/'
    save_fig = False
    if save_fig:
        opt_cf.scan_k_L_D.savefig(folder + f'cf_{rcs_name}_scan3d')
        opt_cf.scan_k_optL_D.savefig(folder+ f'cf_{rcs_name}_scan_k_dx')
        opt_cf.scan_k_optL_Bpole.savefig(folder+ f'cf_{rcs_name}_scan_k_bpole')
        opt_cf.scan_k_optL_beamsize.savefig(folder+ f'cf_{rcs_name}_scan_k_beamsize')
        fig_dx_Bext.savefig(folder+ f'cf_{rcs_name}_scan_bext_dx')
        fig_cell.savefig(folder+ f'cf_{rcs_name}_cell_layout')
        fig_env.savefig(folder+ f'cf_{rcs_name}_beam_size')

    
    # with open(f"/mnt/c/muco/code/latex_table_CF/CF_{rcs_name}.txt", "w") as outfile: 
    #     json.dump(para_rcs, outfile, indent=4)

# line.to_json(f'/mnt/c/muco/code/latex_table_CF/cellCF_{rcs_name}.json')

# name_table = f'CF_{rcs_name}.tex'
# df = pd.DataFrame({rcs_name+' Parameters': parameter_name, 'Combined Function Magnets' : results_CF})
# with open('/mnt/c/muco/code/latex_table_CF/'+name_table, 'w') as f:
#     f.write(df.to_latex(index=False, column_format='lcc'))