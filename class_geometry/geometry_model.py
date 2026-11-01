import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xtrack as xt
import xpart as xp
from class_geometry.ring_geometry.ring_geometry import RingGeometry
from matplotlib.ticker import ScalarFormatter


FIGSIZE = (6.4, 4.8)

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=11)    # legend fontsize


class GeoModel(RingGeometry):

    def __init__(
        self, 
        file_input : dict,
        n_slice : int = 10
        ):

        super().__init__(file_input=file_input)

        self.n_slice = n_slice

    def make_cell(self, time_frac : float, method : str = 'var_ref', t_ref = 0.5, slicing = True, n_slice = None):

        energy = self.injection_energy + (self.extraction_energy - self.injection_energy) * time_frac

        pattern = self.pattern  # Dipole pattern
        L_dd_n = self.L_dd_path(time_frac)  # Dipole spacing [m]
        L_dip_n = self.L_dip_path(time_frac)    # Dipole length [m]
        h_n = self.dip_bending(time_frac)   # Bending [1/m]
        e_n = self.epsilon(time_frac)   # Entry angle [rad]

        L_dip_n_ref = self.L_dip_path(t_ref)
        h_n_ref = self.dip_bending(t_ref)
        e_n_ref = self.epsilon(t_ref)

        L_qd_extra = self.L_extra_arc/self.nb_cell_arc/4/np.cos(e_n[0])

        hcell_el = []
        hcell_name = []

        hcell_el.append(xt.Drift(length = L_qd_extra))
        hcell_name.append(f'drift_qd_0')

        for i_dip, item in enumerate(pattern):
            hcell_el.append(xt.Drift(length = L_dd_n[i_dip]))
            hcell_name.append(f'drift_dd_{i_dip}')

            if method == 'var_ref':            
                hcell_el.append(xt.Bend(
                    k0 = h_n[i_dip], 
                    angle = L_dip_n[i_dip] * h_n[i_dip],
                    length = L_dip_n[i_dip],
                    edge_entry_angle = e_n[i_dip], 
                    edge_exit_angle = -e_n[i_dip+1]
                    ))
            elif method == 'var_k':
                hcell_el.append(xt.Bend(
                    k0 = h_n[i_dip], 
                    angle = L_dip_n_ref[i_dip] * h_n_ref[i_dip],
                    length = L_dip_n_ref[i_dip],
                    edge_entry_angle = e_n_ref[i_dip], 
                    edge_exit_angle = -e_n_ref[i_dip+1]
                    ))
            else:
                raise ValueError(f'Method {method} is not valid. Valid methods: var_ref or var_k.')

            hcell_name.append(item + f'{i_dip}')

        hcell_el.append(xt.Drift(length = L_dd_n[0]))
        hcell_name.append(f'drift_dd_{i_dip+1}')

        hcell_el.append(xt.Drift(length = L_qd_extra))
        hcell_name.append(f'drift_qd_1')
        
        line_dip =  xt.Line(
            elements = hcell_el, 
            element_names = hcell_name)

        line_dip.particle_ref = xp.Particles(
            p0c = energy, 
            q0 = 1,
            mass0 = xp.MUON_MASS_EV)

        if slicing:
            if not n_slice:
                n_slice = self.n_slice

            line_dip.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(n_slice))])

        return line_dip

    def _plot_magnet_limits(self, ax=None):
        if ax is None:
            ax = plt.gca()

        zloc = (
            self._dipole_spacing
            + self.L_extra_arc / self.nb_cell_arc / 4
            / np.cos(self.epsilon(0)[0])
        )

        for idx, (z1, z2) in enumerate(zip(self.z_begin(1), self.z_end(1))):
            x1, x2 = np.real(z1), np.real(z2)

            ax.axvline(zloc + x1, color="grey", alpha=0.3)
            ax.axvline(zloc + x2, color="grey", alpha=0.3)

            ax.text(
                zloc + 0.5 * (x1 + x2),
                1.02,
                self.pattern[idx][1:],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
            )

    def _time_legend(self, lines, labels):
        plt.legend(
            lines,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=3,
        )

    def _add_style_legend(self, label):
        legend = plt.legend(
            handles=[
                Line2D([0], [0], color='black', lw=1, label=label[0]),
                Line2D([0], [0], marker='+', color='black',
                    linestyle='None', label=label[1])
            ],
            loc='upper right'
        )
        plt.gca().add_artist(legend)
    
    def _finalize_plot(
        self,
        xlabel,
        ylabel,
        plot_magnet_limit=False,
        y_scientific = False
    ):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if y_scientific:
            plt.ticklabel_format(axis='y', style='sci', scilimits=(-6,-6))

        if plot_magnet_limit:
            plt.subplots_adjust(top=0.88, bottom=0.2)
        # else:
        #     plt.tight_layout()

        plt.show()

    def comp_method(self, t_traj: np.ndarray = np.linspace(0,1,6), figsize: tuple = (6.4, 5.2), t_ref : float = 0.5, slicing = True):
        res = {}

        for t in t_traj:
            res[str(t)] = {}
            line_t = self.make_cell(time_frac = t, method='var_ref', slicing=slicing)
            line_ref =self.make_cell(time_frac = t_ref, method='var_ref', slicing=slicing)
            line_t_var = self.make_cell(time_frac = t, method='var_k', t_ref=t_ref, slicing=slicing)

            sv_t = line_t.survey(theta0=2 * np.pi / self.nb_cell_rcs / 4)
            sv_ref = line_ref.survey(theta0=2 * np.pi / self.nb_cell_rcs / 4)
            part = line_t_var.build_particles()
            line_t_var.track(particles=part, turn_by_turn_monitor='ONE_TURN_EBE')
            rec = line_t_var.record_last_track

            res[str(t)]['ref_z'] = sv_t['Z']
            res[str(t)]['ref_x'] = sv_t['X']
            res[str(t)]['k_z'] = sv_ref['Z']
            res[str(t)]['k_x'] = sv_ref['X'] + rec.x[0]

        self.method_res = res

        fig1 = plt.figure(figsize=figsize, constrained_layout=True)
        lines, labels = [], []
        for t in t_traj:
            l, =  plt.plot(res[str(t)]['ref_z'], res[str(t)]['ref_x']*1e3)
            lines.append(l)
            labels.append(f"t={t:.1f}")
            plt.scatter(res[str(t)]['k_z'], res[str(t)]['k_x']*1e3, marker='+', color = l.get_color())
        self._add_style_legend([r'$h = k_0$', r'$h \ne k_0$'])
        self._time_legend(lines, labels)
        self._finalize_plot('s [m]', 'x [mm]')

        fig2 = plt.figure(figsize=figsize, constrained_layout=True)
        for t in t_traj:
            plt.plot(
                res[str(t)]['ref_z'], 
                res[str(t)]['ref_z']-res[str(t)]['k_z']
                )
        self._time_legend(lines, labels)
        self._finalize_plot('s [m]', '$\Delta z$ [m]')

        fig3 = plt.figure(figsize=figsize, constrained_layout=True)
        for t in t_traj:
            plt.plot(
                res[str(t)]['ref_z'], 
                (res[str(t)]['ref_x']-res[str(t)]['k_x']) * 1e3
                )
        self._time_legend(lines, labels)
        self._finalize_plot('s [m]', '$\Delta x$ [mm]', y_scientific=True)

        print('==============METHOD: ENDINGS POINTS==============')
        for key in self.method_res.keys():
            print(key)
            print('x', self.method_res[key]['ref_x'][-1] - self.method_res[key]['k_x'][-1])
            print('z', self.method_res[key]['ref_z'][-1] - self.method_res[key]['k_z'][-1])

        return fig1, fig2, fig3

    def comp_traj(self, t_traj : np.ndarray = np.linspace(0,1,6), figsize: tuple = (6.4,5.2), plot_magnet_limit = False):
        
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        lines, labels = [], []
        for i, time in enumerate(t_traj):
            # line = self.make_cell(time)
            # survey = line.survey(theta0=2 * np.pi / self.nb_cell_rcs / 4)
            # l, = plt.plot(survey['Z'], survey['X'] * 1e3) #, color=color)

            t_ref = 0.5
            line_ref =self.make_cell(time_frac = t_ref, method='var_ref', slicing=True)
            line_t_var = self.make_cell(time_frac = time, method='var_k', t_ref=t_ref, slicing=True)
            sv_ref = line_ref.survey(theta0=2 * np.pi / self.nb_cell_rcs / 4)
            part = line_t_var.build_particles()
            line_t_var.track(particles=part, turn_by_turn_monitor='ONE_TURN_EBE')
            rec = line_t_var.record_last_track
            l, = plt.plot(sv_ref['Z'], (sv_ref['X']+rec.x[0]) * 1e3) #, color=color)


            lines.append(l)
            labels.append(f"t={time:.1f}")

            plt.scatter(np.real(self.zn(time)),
                        np.imag(self.zn(time)) * 1e3,
                        marker='+',
                        color=l.get_color())

        if plot_magnet_limit:
            self._plot_magnet_limits()
        self._add_style_legend(['Xsuite', 'Model'])
        self._time_legend(lines, labels)
        self._finalize_plot('s [m]', 'x [mm]', plot_magnet_limit)

        return fig
    
    def compute_zn_model(self, x: float, s: np.ndarray) -> np.ndarray:
        
        eps = self.epsilon(x)
        z_b = self.z_begin(x)
        bb = self.list_b(x)
        
        npt = len(s)
        z = np.zeros(npt + 2, dtype=complex)
        pts_per_dipole = npt // self.nd

        z[1] = (self._dipole_spacing + self.L_extra_arc /self.nb_cell_arc / 4) / np.cos(eps[0]) * np.exp(1j * eps[0])

        for n in np.arange(self.nd):
            i0 = n * pts_per_dipole
            i1 = (n + 1) * pts_per_dipole

            s_loc = s[i0:i1]
            s_entry = s_loc[0]
            s_exit  = s_loc[-1]
            dc = (s_loc - s_entry)/(s_exit - s_entry)

            e0 = eps[n]
            e1 = eps[n + 1]
            e = e0 + dc * (e1 - e0)
            cc = (np.cos(e) - np.cos(e0)) / bb[n]

            z0 = z[1] + z_b[n]
            z[1+i0:1+i1]  = z0 + self.l_dipoles[n] * dc + 1j * self.rho * cc
        z[-1] = z[-2] + (self._dipole_spacing + self.L_extra_arc / self.nb_cell_arc / 4) / np.cos(eps[-1]) * np.exp(1j * eps[-1])

        return z

    def error_traj(self, t_traj : np.ndarray = np.linspace(0,1,6), figsize: tuple = (6.4,5.2), plot_magnet_limit = False, n_slice = None):
        
        res = {}
        for i, time in enumerate(t_traj):
            res[str(time)] = {}
            line = self.make_cell(time_frac=time, slicing=True, n_slice=n_slice)
            survey = line.survey(theta0=2 * np.pi / self.nb_cell_rcs / 4)

            t_ref = 0.5
            line_ref =self.make_cell(time_frac = t_ref, method='var_ref', slicing=True, n_slice=n_slice)
            line_t_var = self.make_cell(time_frac = time, method='var_k', t_ref=t_ref, slicing=True, n_slice=n_slice)
            survey = line_ref.survey(theta0=2 * np.pi / self.nb_cell_rcs / 4)
            part = line_t_var.build_particles()
            line_t_var.track(particles=part, turn_by_turn_monitor='ONE_TURN_EBE')
            rec = line_t_var.record_last_track
            
            el_names = survey['name']
            mask = [s.startswith('B') and 'map' not in s for s in el_names]

            zn_points = self.compute_zn_model(x=time, s = survey['Z'][mask])

            mask[0], mask[-1] = True, True

            res[str(time)]['x_survey'] = survey['X'][mask] + rec.x[0][mask]
            res[str(time)]['z_survey'] = survey['Z'][mask]
            res[str(time)]['z_model'] = np.real(zn_points) 
            res[str(time)]['x_model'] = np.imag(zn_points) 
        
        self.error_traj_res = res

        lines, labels = [], []
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        for time in t_traj:
            l, = plt.plot(
                res[str(time)]['z_survey'], 
                (res[str(time)]['x_survey'] - res[str(time)]['x_model']) * 1e3, 
                label=f"t={time:.1f}"
                )
            lines.append(l)
            labels.append(f"t={time:.1f}")

        if plot_magnet_limit:
            self._plot_magnet_limits()

        self._time_legend(lines, labels)
        self._finalize_plot('s [m]', 'x [mm]', plot_magnet_limit)

        plt.figure()
        for time in t_traj:
            l, = plt.plot(
                res[str(time)]['z_model'],
                res[str(time)]['z_model']-res[str(time)]['z_survey'],
                label=f"t={time:.1f}"
                )
        plt.ylabel('dz [m]')
        plt.xlabel('z [m]')
        plt.show()

        print('==============MODEL vs XSUITE: ENDINGS POINTS==============')
        for key in self.error_traj_res.keys():
            print(key)
            print('x', self.error_traj_res[key]['x_survey'][-1] - self.error_traj_res[key]['x_model'][-1])
            print('z', self.error_traj_res[key]['z_survey'][-1] - self.error_traj_res[key]['z_model'][-1])

        return fig

    def comp_path_length(
        self, 
        t_traj : np.ndarray = np.linspace(0,1,15), 
        t_ref : float = 0.5, 
        figsize : tuple = FIGSIZE
        ):

        # Model
        path_length_model = np.array([self.path_length(time) for time in t_traj])
        path_length_model -= np.min(path_length_model)

        # Xsuite
        
        # for i, time in enumerate(t_traj):
        #     line = self.make_cell(time_frac = time, slicing = False)
        #     survey = line.survey(theta0= 2 * np.pi / self.nb_cell_rcs / 4)
        #     path_length_xsuite[i] = survey['s','_end_point']
        # path_length_xsuite -= np.min(path_length_xsuite)

        path_length_xsuite = np.zeros(len(t_traj))
        for i, time in enumerate(t_traj):
            line_t_var = self.make_cell(time_frac = time, method='var_k', t_ref=t_ref, slicing=False)
            part = line_t_var.build_particles()
            line_t_var.track(particles=part, turn_by_turn_monitor='ONE_TURN_EBE')
            rec = line_t_var.record_last_track
            path_length_xsuite[i] = - rec.zeta[0][-1]
        path_length_xsuite -= np.min(path_length_xsuite)

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        plt.plot(
            t_traj, 
            path_length_xsuite * self.nb_cell_rcs * 2 * 1e3, 
            color='tab:blue', label='Xsuite'
            )
        plt.scatter(
            t_traj, 
            path_length_model * self.nb_cell_rcs * 2 * 1e3, 
            marker = '+', color='tab:orange', label='Model'
            )
        plt.legend(loc='upper right')
        self._finalize_plot(r'$\text{t} \ / \ \text{t}_\text{acc}$', 'Path length variation [mm]')


        fig1 = plt.figure(figsize=figsize, constrained_layout=True)
        plt.plot(
            t_traj, 
            (path_length_xsuite - path_length_model) * self.nb_cell_rcs * 2 * 1e3, 
            color='tab:blue'
            )
        self._finalize_plot(r'$\text{t} \ / \ \text{t}_\text{acc}$', 'Path length variation difference [mm]', y_scientific=True)

        return fig, fig1

def compute_mcf(rcs, mu : float, time : np.ndarray):
    n_c = rcs.nb_cell_rcs
    L_arc = rcs.tot_arc_length
    L_dipole_tot = rcs.dipole_length_tot
    n_d = (len(rcs.pattern) - 1) / 2
    L_dd = rcs.dipole_spacing

    B_SC = rcs.dipole_families['BSC']['B'](0)
    B_NC = rcs.dipole_families['BNC']['B']
    Brho = np.array([rcs.func_Brho(t) for t in time])

    if rcs.pattern[0] == 'BSC':
        a = (rcs.ext_Brho - rcs.inj_Brho) * B_SC / ( 
            (rcs.ext_Brho - rcs.inj_Brho) * B_SC + (rcs.ext_Brho + rcs.inj_Brho) * B_NC(1)
        )
        b = B_NC(time) * L_dipole_tot / (2 * np.pi * Brho)

    elif rcs.pattern[0] == 'BNC':
        a = (rcs.ext_Brho + rcs.inj_Brho) * B_NC(1) / ( 
            (rcs.ext_Brho - rcs.inj_Brho) * B_SC + (rcs.ext_Brho + rcs.inj_Brho) * B_NC(1)
        )
        b = B_SC * L_dipole_tot / (2 * np.pi * Brho)

    alpha_c = (
        (np.pi / n_c) ** 2 
        * (1 / (np.sin(mu / 2) ** 2) 
        - 1 / 4 
        + L_dipole_tot / (6 * L_arc) 
        + ((a * b + n_d) / (6 * L_arc * n_d * (1 + n_d)))
        * (
            a * (1 - b) * L_dipole_tot 
            + 4 * n_c * (n_d * (2 + n_d)- a * b * (1 + 2 * n_d))* L_dd
        )
        )
    )

    return alpha_c

class GeoLaw():

    def __init__(
        self,
        file_input : dict,
        patterns : dict,
        nc_list : np.ndarray = np.arange(50, 300, 10),
        mu: float = 90,
        color_map : dict= {'3': 'blue', '5': 'red', '7': 'green'},
        style_map : dict = {'SC': '-', 'NC': '--'},
        time : np.ndarray = np.linspace(0, 1, 11)
        ):

        self.input = file_input
        self.patterns = patterns
        self.nc_list = nc_list
        self.mu = mu * np.pi / 180
        self.color_map = color_map
        self.style_map = style_map
        self.time = time

        self.rcs_dict = {}
        self.qty = {}

        for key in self.patterns.keys():
            self.rcs_dict[key] = {}
            self.qty[key] = {}

            self.rcs_dict[key]['rcs'] = RingGeometry(file_input=self.input, pattern=self.patterns[key])
            self.rcs_dict[key]['nc_max'] = self.rcs_dict[key]['rcs'].nc_best * self.rcs_dict[key]['rcs'].nb_arc

        self._compute_qty()

    def _make_env_FODO(self, f : float):

        foc = xt.Line(
            elements= [
                xt.Multipole(knl = [0, -1/f], length= 0), 
                xt.Drift(length=rcs.length_short_insertion)
                ],
            element_names = ['qd_foc', 'drift_qd_f']
            )
        defoc = xt.Line(
            elements= [
                xt.Multipole(knl = [0, 1/f], length= 0), 
                xt.Drift(length=rcs.length_short_insertion)
                ],
            element_names = ['qd_defoc', 'drift_qd_d']
            )

        env = xt.Environment()
        env.import_line(line=foc, line_name='foc')
        env.import_line(line=defoc, line_name='defoc')

        return env

    def _compute_mcf_varE(self, time : np.ndarray):

        self.mcf_res = {}

        for key in self.patterns.keys():
            self.mcf_res[key] = {}
            rcs = RingGeometry(file_input=self.input, pattern=self.patterns[key])

            mcf_model = compute_mcf(rcs=rcs, mu=self.mu, time=time)
            self.mcf_res[key]['model'] = mcf_model

            geo = GeoModel(
                file_input = file_input,
                n_slice = 3,
                )

            f = rcs.cell_length / 4 / np.sin(self.mu/2)
            env = self._make_env_FODO(f=f)

            mcf_xsuite = []

            for i_t, t in enumerate(time):
                print(t)
                
                cell = self.make_cell(
                    time_frac = t, 
                    slicing=False
                    )
                cell_name = 'cell' + str(i_t)

                env.import_line(line=cell, line_name=cell_name)
                env.particle_ref = cell.particle_ref
                cell_line = getattr(env, cell_name)
                env.FODO_cell = env.foc + cell_line + env.defoc + cell_line

                tw4d = env.FODO_cell.twiss4d()
                mcf_xsuite.append(tw4d.momentum_compaction_factor)
            
            self.mcf_res[key]['xsuite'] = mcf_xsuite

    def plot_mcf_varE(self, time : np.ndarray = np.linspace(0,1,6), figsize: tuple = FIGSIZE):

        try:
            self.mcf_res
        except:
            self._compute_mcf_varE(time)

        fig = plt.figure(figsize=figsize)
        for key in self.patterns.keys():
            plt.plot(time, self.mcf_res[key]['model'], label=key)
            plt.scatter(time, self.mcf_res[key]['xsuite'], marker='+')

        ax = plt.gca()
        legend1 = ax.legend(
            loc='lower right',
            bbox_to_anchor=(1, 0.0)
        )
        ax.add_artist(legend1)

        handles = [
            Line2D([0], [0], color='black', linestyle='-', label='Model'),
            Line2D([0], [0], color='black', marker='+', linestyle='None',
                markersize=8, label='Xsuite')
        ]
        ax.legend(
            handles=handles,
            loc='lower right',
            bbox_to_anchor=(0.73, 0.0)
        )
                
        plt.xlabel(r'$t \ / \ t_{acc}$')
        plt.ylabel(r'$\alpha_c$')
        plt.show()

        return fig

    
    def _compute_qty(self):
        nc_list = self.nc_list
        
        for key in self.patterns.keys():

            rcs = self.rcs_dict[key]['rcs']

            # Filter out impossible geometries
            mask_nc = nc_list < self.rcs_dict[key]['nc_max']
            nc_real = nc_list[mask_nc]
            n_nc = len(nc_list[mask_nc])

            dpath = np.zeros(n_nc)
            dx_offset = np.zeros(n_nc)
            sc_dip_length = np.zeros(n_nc)
            nc_dip_length = np.zeros(n_nc)
            mcf_min = np.zeros(n_nc)
            mcf_max = np.zeros(n_nc)

            for i, nc in enumerate(nc_real):
                rcs.nb_cell_arc = int(nc / rcs.nb_arc)

                paths = [rcs.path_length(t) for t in np.linspace(0,1,10)]
                dpath[i] = np.max(paths) - np.min(paths) # Path length diff. of a half-cell
                dpath[i] *= 2 * nc # Total path length diff.

                dx_offset[i] = np.max([
                    np.max(rcs.extrema(1)[1] - rcs.extrema(0)[0]), # Case of highE orbit outside of lowE
                    np.max(rcs.extrema(0)[1] - rcs.extrema(1)[0])  # Case of highE orbit inside of lowE
                    ])

                sc_dip_length[i] = rcs.dipole_families['BSC']['length']
                nc_dip_length[i] = rcs.dipole_families['BNC']['length']

                mcf = compute_mcf(
                    rcs = rcs, 
                    mu = self.mu, 
                    time = self.time
                    )

                mcf_min[i] = np.min(mcf)
                mcf_max[i] = np.max(mcf)
            
            self.qty[key]['nc'] = nc_real   # Number of FODO cells
            self.qty[key]['path_length_diff'] = dpath   # Total path length difference [m]
            self.qty[key]['dx_offset'] = dx_offset  # Maximum trajectory offset [m]
            self.qty[key]['sc_dip_length'] = sc_dip_length  # SC dipole length [m]
            self.qty[key]['nc_dip_length'] = nc_dip_length  # NC dipole length [m]
            self.qty[key]['mcf_min'] = mcf_min
            self.qty[key]['mcf_max'] = mcf_max

    def _plot_qty(
        self, 
        x_name : str, 
        y_name : str, 
        x_label : str, 
        y_label : str, 
        yticks,
        figsize : tuple
        ):
        
        fig, ax = plt.subplots(1, figsize=figsize)

        for key, data in self.qty.items():

            dip = key.split()[0]
            tech = 'SC' if 'SC' in key else 'NC'

            ax.plot(
                    data[x_name],
                    data[y_name],
                    color = self.color_map[dip],
                    linestyle = self.style_map[tech],
                    label = f"{tech}, {dip} dipoles",
                    # alpha=0.7
                )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

        return fig

    def plot_mcf(
        self,
        yticks=None,
        figsize : tuple =  FIGSIZE
        ):

        figs = {}

        for tech in ["SC", "NC"]:

            fig, ax = plt.subplots(figsize=figsize)

            for key, data in self.qty.items():

                if tech not in key:
                    continue

                dip = key.split()[0]
                color = self.color_map[dip]

                ax.plot(
                    data['nc'],
                    data['mcf_max'],
                    color = color,
                    linestyle = '-',
                    label = f'{tech}, ' + r'$\alpha_{max}$, ' +  f'{dip} dipoles',
                    alpha = 0.7,
                )

                ax.plot(
                    data['nc'],
                    data['mcf_min'],
                    color = color,
                    linestyle = '--',
                    label = f'{tech}, ' + r'$\alpha_{min}$, ' + f'{dip} dipoles',
                    alpha = 1,
                )

            ax.set_yscale('log')
            ax.set_xlabel('Number of FODO cells')
            ax.set_ylabel(r'$\alpha_c$')

            if yticks is not None:
                ax.set_yticks(yticks)

            ax.legend(loc='upper right')
            fig.tight_layout()

            figs[tech] = fig

        plt.show()

        return figs

    def plot_dip_length(self, figsize : tuple = FIGSIZE, yticks = None):

        figs = {}

        for tech in ["SC", "NC"]:

            fig, ax = plt.subplots(figsize=figsize)

            for key, data in self.qty.items():

                if tech not in key:
                    continue

                dip = key.split()[0]
                color = self.color_map[dip]

                ax.plot(
                    data['nc'],
                    data['sc_dip_length'],
                    color = color,
                    linestyle = '-',
                    label = f'{tech}, ' + '$L_{SC}, $' + f'{dip} dipoles',
                    alpha = 0.7,
                )

                ax.plot(
                    data['nc'],
                    data['nc_dip_length'],
                    color = color,
                    linestyle = '--',
                    label = f'{tech}, ' + '$L_{NC}, $' + f'{dip} dipoles',
                    alpha = 1,
                )

            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlabel('Number of FODO cells')
            ax.set_ylabel('L [m]')

            if yticks is not None:
                ax.set_yticks(yticks[tech])

            ax.legend(loc='upper right')
            fig.tight_layout()

            figs[tech] = fig

        plt.show()

        return figs
    
    def plot_path_length(self, figsize : tuple = FIGSIZE, yticks = None):
        fig = self._plot_qty( 
            x_name = 'nc', 
            y_name = 'path_length_diff', 
            x_label = 'Number of FODO cells', 
            y_label = 'Total path length difference [m]',
            figsize = figsize,
            yticks = yticks
            )

        return fig

    def plot_sc_length(self, figsize : tuple = FIGSIZE, yticks = None):
        fig = self._plot_qty( 
            x_name = 'nc', 
            y_name = 'sc_dip_length', 
            x_label = 'Number of FODO cells', 
            y_label = 'SC dipole length [m]',
            figsize = figsize,
            yticks = yticks
            )
        
        return fig

    def plot_nc_length(self, figsize : tuple = FIGSIZE, yticks = None):
        fig = self._plot_qty( 
            x_name = 'nc', 
            y_name = 'nc_dip_length', 
            x_label = 'Number of FODO cells', 
            y_label = 'NC dipole length [m]',
            figsize = figsize,
            yticks = yticks
            )
        
        return fig

    def plot_dx_offset(self, figsize : tuple = FIGSIZE, yticks = None):
        fig = self._plot_qty( 
            x_name = 'nc', 
            y_name = 'dx_offset', 
            x_label = 'Number of FODO cells', 
            y_label = 'Maximum trajectory offset [m]',
            figsize = figsize,
            yticks = yticks)
        
        return fig

if __name__ == '__main__':

    file_input = {  
        "type": "hybrid",
        "circumference": 5990,
        "injection_energy": 300e9,
        "extraction_energy": 750e9,
        "length_short_insertion": 2.5,
        "dipole_spacing": 0.4,
        "nb_RF_section": 24,
        "nb_arc": 10,
        "nb_cell_arc": 13,
        "dipoles": {
            "BNC": [1.8, "linear"],
            "BSC": 10.0
        },
        "filling_ratio": 0.9,
        "filling_ratio_RF": 0.4,
        "pattern": ["BSC","BNC","BSC"]
        }

    dict_pattern = {
        '3 dip, NC' : ["BNC", "BSC", "BNC"],
        '5 dip, NC' : ["BNC", "BSC", "BNC", "BSC", "BNC"],
        '7 dip, NC' : ["BNC", "BSC", "BNC", "BSC", "BNC", "BSC", "BNC"],
        '3 dip, SC' : ["BSC", "BNC", "BSC"],
        '5 dip, SC' : ["BSC", "BNC", "BSC", "BNC", "BSC"],
        '7 dip, SC' : ["BSC", "BNC", "BSC", "BNC", "BSC", "BNC", "BSC"]
    }
    
    file_save = '/mnt/c/muco/Results/figures/chap4/'
    plot_save_traj = False
    if plot_save_traj:
        for key in dict_pattern.keys():
            file_input['pattern'] = dict_pattern[key]
            rcs = RingGeometry(file_input=file_input)
            print(key)
            print(rcs.nb_cell_rcs)
            fig = rcs.plot_traj(t_traj=np.linspace(0,1,6))
            fig.savefig(file_save + key)
    
    geo = GeoModel(
        file_input = file_input,
        n_slice = 10
    )

    n_traj = 6
    n_path = 20

    fig_comp_traj = geo.comp_traj(t_traj=np.linspace(0, 1, n_traj), plot_magnet_limit=True)
    fig_comp_traj_10 = geo.error_traj(t_traj=np.linspace(0, 1, n_traj), plot_magnet_limit=True, n_slice = 10)
    fig_comp_traj_50 = geo.error_traj(t_traj=np.linspace(0, 1, n_traj), plot_magnet_limit=True, n_slice = 50)
    fig_comp_traj_200 = geo.error_traj(t_traj=np.linspace(0, 1, n_traj), plot_magnet_limit=True, n_slice = 200)
    fig_path_length, fig_path_length_diff = geo.comp_path_length(t_traj=np.linspace(0, 1, n_path))
    fig_method_traj, fig_method_dz, fig_method_dx = geo.comp_method(slicing=True)
    fig_method_traj_nos, fig_method_dz_nos, fig_method_dx_nos = geo.comp_method(slicing=False)

    savef = False
    if savef:
        fig_comp_traj.savefig(file_save + 'comp_traj')
        fig_comp_traj_10.savefig(file_save + 'error_traj_10')
        fig_comp_traj_50.savefig(file_save + 'error_traj_50')
        fig_comp_traj_200.savefig(file_save + 'error_traj_200')
        fig_path_length.savefig(file_save + 'path_length')
        fig_path_length_diff.savefig(file_save + 'path_length_diff')
        fig_method_traj.savefig(file_save + 'method_traj')
        fig_method_dz.savefig(file_save + 'method_dz')
        fig_method_dx.savefig(file_save + 'method_dx')

        fig_method_traj_nos.savefig(file_save + 'method_traj_noslicing')
        fig_method_dz_nos.savefig(file_save + 'method_dz_noslicing')
        fig_method_dx_nos.savefig(file_save + 'method_dx_noslicing')

    mu = 90

    law = GeoLaw(
        file_input = file_input,
        patterns = dict_pattern,
        mu = mu
    )

    fig_path = law.plot_path_length(yticks=np.array([2, 1, 0.5, 0.1, 0.05]))
    fig_offset = law.plot_dx_offset(yticks=np.array([0.5, 0.1, 0.05]))
    fig_sc_length = law.plot_sc_length(yticks=np.array([10, 5, 1]))
    fig_nc_length = law.plot_nc_length(yticks=np.array([20, 10, 5, 1]))
    figs_dip_length = law.plot_dip_length(yticks=
                    {'SC' : np.array([30, 20, 10, 5, 1]),
                    'NC' : np.array([20, 10, 5, 1])})
    fig_mcf_varE = law.plot_mcf_varE()
    figs_mcf = law.plot_mcf(yticks=np.array([0.01, 0.001]))

    savef = False
    if savef:
        fig_path.savefig(file_save + 'path_length_scan')
        fig_offset.savefig(file_save + 'offset_scan')
        fig_sc_length.savefig(file_save + 'sclength_scan')
        fig_nc_length.savefig(file_save + 'nclength_scan')
        figs_dip_length['SC'].savefig(file_save + 'dip_length_SC_scan')
        figs_dip_length['NC'].savefig(file_save + 'dip_length_NC_scan')
        figs_mcf['SC'].savefig(file_save + 'mcf_SC_scan')
        figs_mcf['NC'].savefig(file_save + 'mcf_NC_scan')
        # fig_mcf_varE.savefig(file_save + 'mcf_E_scan')

    
    # if ciao = enrico then supp the program