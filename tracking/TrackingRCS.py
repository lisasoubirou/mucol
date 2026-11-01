import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xpart as xp
import scipy.constants as sc
from scipy.interpolate import CubicSpline
from class_geometry.ring_geometry.ring_geometry import RingGeometry
from optics.generate_fodo import plot_twiss
from EB_mismatch.EB_error import EB_error

def calc_separatrix_vphi(
    particles : xp.Particles, 
    rcs :  RingGeometry, 
    tw_arc : xt.TwissTable, 
    n_turn_rcs : int, 
    v_rf : float,
    phase_rf : float,
    frequency_rf : float = 1300*1e6,
    xlim : list = None,
    ylim : list = None
    ):
    """Plot initial particle distribution with separatrix.

    Args:
        particles (xp.Particles): Initial particle distribution.
        rcs (RingGeometry): RCS Geometry.
        tw_arc (xt.TwissTable): 4D twiss table of the arc.
        n_turn_rcs (int): Number of turn in the rcs.
        v_rf (float): Cavity voltage of 1 cavity (considering only 2 cavities in the arc) [V].
        phase_rf (float): Cavity phase [°].
        frequency_rf (float, optional): Cavity frequency [Hz]. Defaults to 1300*1e6.
        xlim (list, optional): Horizontal limits of the plot. Defaults to None.
        ylim (list, optional): Vertical limits of the plot. Defaults to None.
    """

    inj_E = rcs.injection_energy
    ext_E = rcs.extraction_energy
    energy_increment_per_arc = (ext_E - inj_E) / n_turn_rcs / rcs.nb_arc

    particles_gamma = particles.gamma0[0]
    particle_energy = particles.p0c[0]
    zeta = particles.zeta
    delta = particles.delta
    energy_increment_per_arc = (ext_E - inj_E) / n_turn_rcs / rcs.nb_arc
    #RF bucket defined here on 1 arc => careful with consistency!!
    machine_rf_bucket = xp.longitudinal.rf_bucket.RFBucket(
            circumference = rcs.circumference / rcs.nb_arc,
            gamma = particles_gamma, 
            mass_kg = sc.physical_constants['muon mass'][0],
            charge_coulomb = sc.e, 
            alpha_array = [tw_arc.momentum_compaction_factor], 
            p_increment = energy_increment_per_arc * sc.e / sc.c, #*nb_arc
            harmonic_list = [np.round(frequency_rf * tw_arc.T_rev0 * rcs.nb_arc) / rcs.nb_arc], 
            voltage_list = [v_rf*2], #*nb_arc
            phi_offset_list = np.atleast_1d((phase_rf) * np.pi / 180),
            z_offset=None)

    ufp = machine_rf_bucket.z_ufp
    xx = np.linspace(machine_rf_bucket.z_left, machine_rf_bucket.z_right, 1000)
    yy = machine_rf_bucket.separatrix(xx)
    plt.figure()
    # plt.scatter(rec.zeta*fac, rec.delta*particle_energy/1e9, color='C0',s=2)    
    plt.scatter(zeta, delta, s=2)
    plt.plot((xx-machine_rf_bucket.z_sfp), yy, color='tab:orange')
    plt.plot((xx-machine_rf_bucket.z_sfp), -yy, color='tab:orange')
    plt.xlabel('z [m]')
    plt.ylabel(r'$\delta$')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)    
    plt.show()

    return machine_rf_bucket


def plot_emit(ex, ey, es, tr, mon_s):
    """Plot a 4 subfigures graph, for the 3 emittances and transmission.

    Args:
        ex (_type_): Array of horizontal emittance.
        ey (_type_): Array of vertical emittance.
        es (_type_): Array of longitudinal emittance.
        tr (_type_): Array of transmission.
    """

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6.4, 6.4))
    
    axes[0].plot(mon_s * 1e-3, (ex-ex[0])/ex[0] * 100, color='tab:blue')
    axes[1].plot(mon_s * 1e-3, (ey-ey[0])/ey[0] * 100, color='tab:orange')
    axes[2].plot(mon_s * 1e-3, (es-es[0])/es[0] * 100, color='tab:green')
    axes[3].plot(mon_s * 1e-3, tr * 100, color='tab:purple')

    axes[0].set_ylabel(r'$\Delta \epsilon_x / \epsilon_x$ [%]')
    axes[1].set_ylabel(r'$\Delta \epsilon_y / \epsilon_y$ [%]')
    axes[2].set_ylabel(r'$\Delta \epsilon_s / \epsilon_s$ [%]')
    axes[3].set_ylabel(r'Transmission [%]')

    axes[3].set_xlabel('s [km]')

    axes[3].ticklabel_format(axis='y', style='plain', useOffset=False)

    plt.tight_layout()
    plt.show()

    return fig

def plot_emit_2rcs(res1, res2, label1="RCS 1", label2="RCS 2"):
    """
    Compare les évolutions des émittances et de la transmission
    pour deux configurations de RCS.

    Parameters
    ----------
    res1 : dict
        Dictionnaire des résultats de la première configuration.
    res2 : dict
        Dictionnaire des résultats de la seconde configuration.
    label1 : str
        Label de la première configuration.
    label2 : str
        Label de la seconde configuration.
    """

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6.4, 6.4), sharex=True)

    configs = [
        (res1, label1, "tab:blue"),
        (res2, label2, "tab:orange"),
    ]

    for res, cfg_label, cl in configs:

        ex = res["emitx"]
        ey = res["emity"]
        es = res["emits"]
        tr = res["transmission"]
        s = res["s"]

        # label uniquement sur la première courbe
        axes[0].plot(
            s * 1e-3,
            (ex - ex[0]) / ex[0] * 100,
            color=cl,
            label=cfg_label,
        )

        axes[1].plot(
            s * 1e-3,
            (ey - ey[0]) / ey[0] * 100,
            color=cl,
        )

        axes[2].plot(
            s * 1e-3,
            (es - es[0]) / es[0] * 100,
            color=cl,
        )

        axes[3].plot(
            s * 1e-3,
            tr * 100,
            color=cl,
        )

    axes[0].set_ylabel(r'$\Delta \epsilon_x / \epsilon_x$ [%]')
    axes[1].set_ylabel(r'$\Delta \epsilon_y / \epsilon_y$ [%]')
    axes[2].set_ylabel(r'$\Delta \epsilon_s / \epsilon_s$ [%]')
    axes[3].set_ylabel("Transmission [%]")
    axes[3].set_xlabel("s [km]")
    axes[3].ticklabel_format(axis='y', style='plain', useOffset=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.57, -0.026),
        ncol=2,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()

    return fig

def off_dE(line : xt.Line) -> float:
    dE_list = line.get_elements_of_type(xt.ReferenceEnergyIncrease)[1]
    delta_p0c = line[dE_list[0]].Delta_p0c
    for dE in dE_list: 
        line[dE].Delta_p0c = 0
    
    return delta_p0c

def on_dE(line : xt.Line, delta_p0c : float) -> None:
    dE_list = line.get_elements_of_type(xt.ReferenceEnergyIncrease)[1]
    for dE in dE_list: 
        line[dE].Delta_p0c = delta_p0c

def compute_action_list(u, v):
    """
    Compute single-particle action using beam moments (Twiss from covariance).
    """
    cov = np.cov(u, v)
    eps = np.sqrt(np.linalg.det(cov))

    bet = cov[0, 0] / eps
    alf = -cov[0, 1] / eps
    gam = cov[1, 1] / eps

    return 0.5 * (gam * u**2 + 2 * alf * u * v + bet * v**2)

def compute_action_perc(u, v, percentile):
    """
    Compute percentile action with one cleaning iteration.
    """
    action_list = compute_action_list(u, v)
    action_cut = np.percentile(action_list, percentile)

    mask = action_list < action_cut
    action_list = compute_action_list(u[mask], v[mask])

    return np.percentile(action_list, percentile)

def _calc_emit(u : np.ndarray, v : np.ndarray, delta : np.ndarray) -> float:
    """Compute dispersion-corrected emittance from particle coordinates."""

    cov = np.cov([u, v, delta])

    dx = cov[0, 2] / cov[2, 2]
    dpx = cov[1, 2] / cov[2, 2]

    eps2 = (
        (cov[0, 0] - dx * cov[0, 2])
        * (cov[1, 1] - dpx * cov[1, 2])
        - (cov[0, 1] - dx * cov[1, 2]) ** 2
    )

    return np.sqrt(eps2)

def calc_emit_x(mon, mask, turn):
    return _calc_emit(
        mon.x[mask, turn],
        mon.px[mask, turn],
        mon.delta[mask, turn],
    )

def calc_emit_y(mon, mask, turn):
    return _calc_emit(
        mon.y[mask, turn],
        mon.py[mask, turn],
        mon.delta[mask, turn],
    )

def calc_emit_s_particles(mon, n_perc=95):
    
    zeta = mon.zeta
    delta = mon.delta
    p0c = np.mean(mon.p0c)

    action = compute_action_perc(zeta, delta, n_perc)
    es = action / 2 * p0c / sc.c  # Normalized emittance
    
    return es

def calc_emit_s_per(mon, mask, turn, n_perc=95):
    
    zeta = mon.zeta[:, turn][mask]
    delta = mon.delta[:, turn][mask]
    p0c = np.mean(mon.p0c[:, turn][mask])

    action = compute_action_perc(zeta, delta, n_perc)
    es = action / 2 * p0c / sc.c  # Normalized emittance
    
    return es

class TrackingRCS():
    """Build the RCS line from an arc line and perform tracking of a matched particle distribution.
    """

    def __init__(
            self,
            line : xt.Line,
            rcs : RingGeometry,
            n_turn_rcs : int,
            n_part : int,
            nemitt_x : float,
            nemitt_y : float,
            sigma_z : float,
            t_acc : float,
            ramp : bool = False,
            s_jitter: float = 0,
            jitter_diagonal: bool = False,
            n_sigma : float = 5,
            method : str = 'var_ref',
            RF_tuning : bool = False,
            n_range : bool = None,
            n_tracking : int = 1,
            adapt_part = False,
            adapt_percentile : float = 95,
            plt_twiss : bool = False,
            plt_sepa : bool = False,
    ):
        """
        Args:
            line (xt.Line): Arc line.
            rcs (RingGeometry): Geometry of the rcs.
            n_turn_rcs (int): Number of turns in the rcs.
            n_part (int): Number of particles to track.
            nemitt_x (float): Normalized horizontal emittance [m]
            nemitt_y (float): Normalized vertical emittance [m].
            sigma_z (float): Bunch length [s].
            t_acc (float): Acceleration time [s]
            ramp (bool, optional): Apply fast ramp effect. Defaults to False.
            n_sigma (float, optional): Cut in sigma of longitudinal aperture. Defaults to 5.
            method (str, optional): Pulsing method: can be 'var_ref' or 'var_k'. Defaults to 'var_ref'.
            RF_tuning (bool, optional): Apply RF tuning for hybrid synchrotron. Defaults to False.
            n_range (bool, optional): Total number of arcs to reduce the rcs line to. Defaults to None.
            n_tracking (int, optional): Number of turns the rcs line is tracked (relevant when n_range is not None). Defaults to 1.
            adapt_part (bool, optional): Initial particle distribtion is matched to 2nd order. Defaults to False.
            adapt_percentile (float, optional): Define the max sigma delta to which the twiss adapation is done for particle matching. Defaults to 95.
            plt_twiss (bool, optional): Plot twiss functions. Defaults to False.
            plt_sepa (bool, optional): Plot chromatic functions. Defaults to False.
        """
        
        self.line = line
        self.rcs = rcs
        self.n_turn_rcs = n_turn_rcs
        self.nb_arc_tot = n_turn_rcs * rcs.nb_arc
        self.t_acc = t_acc

        self.n_part = n_part
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y
        self.sigma_z = sigma_z
        self.n_sigma = n_sigma
        self.adapt_part = adapt_part
        self.adapt_percentile = adapt_percentile

        self.ramp = ramp
        self.s_jitter = s_jitter
        self.jitter_diagonal = jitter_diagonal
        self.method = method
        self.RF_tuning = RF_tuning
        
        self.plt_twiss = plt_twiss
        self.plt_sepa = plt_sepa
        self.n_range = n_range
        self.n_tracking = n_tracking

        self.context = xo.ContextCpu(omp_num_threads='auto')

        delta_p0c = self.line['RF_acc_1_l'].Delta_p0c
        self.line['RF_acc_1_l'].Delta_p0c = 0
        self.line['RF_acc_1_r'].Delta_p0c = 0

        self.tw_arc = self.line.twiss4d(
            compute_chromatic_properties=True)
        
        if plt_twiss:
            plot_twiss(self.tw_arc)
    
        self.generate_part_distribution()

        self.line['RF_acc_1_l'].Delta_p0c = delta_p0c
        self.line['RF_acc_1_r'].Delta_p0c = delta_p0c

        if self.plt_sepa:
            self.fig_sepa = self.plot_separatrix()
        
        self.zeta_max = n_sigma * np.std(self.particles.zeta)
        self.delta_max = n_sigma * np.std(self.particles.delta)

        self.build_line_rcs()
        self.insert_mon_apert()
        self.tracking_rcs()
        self.compute_emit()
        
        self.fig_tracking = plot_emit(
            ex = self.track_res['emitx'],
            ey = self.track_res['emity'],
            es = self.track_res['emits'],
            tr = self.track_res['transmission'],
            mon_s = self.track_res['s']
        )

    def generate_part_distribution(self):
        """Generate matched particle distribution. 
        If adapt_part is true, the beam is matched at 2nd order.
        """
    
        print('----- Generating particle distribution -----')
        self.particles = xp.generate_matched_gaussian_bunch(
            num_particles = self.n_part, 
            nemitt_x = self.nemitt_x, 
            nemitt_y = self.nemitt_y, 
            sigma_z = self.sigma_z * sc.c,
            line = self.line
            )

        self.particles.zeta = self.particles.zeta - np.mean(self.particles.zeta)
        print('Emittance x at injection, before 2nd order matching', _calc_emit(self.particles.x, self.particles.px, self.particles.delta)*np.mean(self.particles.gamma0))
        print('Emittance s at injection', calc_emit_s_particles(self.particles))

        if self.adapt_part:
            limit = np.percentile(self.particles.delta, self.adapt_percentile)
            
            arr_delta = np.linspace(-limit, limit, 31)
            arr_twiss = np.zeros((31, 8))

            for i, delta in enumerate(arr_delta):
                tw = self.line.twiss(method='4d', matrix_stability_tol=0.1, delta0=delta)
                arr_twiss[i, 0] = tw.betx[0]
                arr_twiss[i, 1] = tw.alfx[0]
                arr_twiss[i, 2] = tw.bety[0]
                arr_twiss[i, 3] = tw.alfy[0]
                arr_twiss[i, 4] = tw.x[0]
                arr_twiss[i, 5] = tw.px[0]
                arr_twiss[i, 6] = tw.y[0]
                arr_twiss[i, 7] = tw.py[0]

            twiss_delta = CubicSpline(arr_delta, arr_twiss, bc_type='not-a-knot')
            tw_loc = twiss_delta(self.particles.delta)
            Ax = np.sqrt(tw_loc[:, 0] / self.tw_arc.betx[0])
            Bx = (self.tw_arc.alfx[0] - tw_loc[:,1]) / np.sqrt(self.tw_arc.betx[0] * tw_loc[:,0])
            Ay = np.sqrt(tw_loc[:, 2] / self.tw_arc.bety[0])
            By = (self.tw_arc.alfy[0] - tw_loc[:,3]) / np.sqrt(self.tw_arc.bety[0] * tw_loc[:,2])
            self.particles.px = Bx * self.particles.x + self.particles.px / Ax + tw_loc[:, 5]
            self.particles.x = Ax * self.particles.x  + tw_loc[:, 4]
            self.particles.py = By * self.particles.y + self.particles.py / Ay  + tw_loc[:, 7]
            self.particles.y = Ay * self.particles.y  + tw_loc[:, 6]

            print(f'Matched distribution adapted to second order with max delta: {limit}')

        self.particles0 =  self.particles.copy()
    
    def plot_separatrix(self):
        """Plot separatrix and initial particle distribution
        """
        print('----- Initial particle distribution and separatrix -----')
        self.rf_bucket = calc_separatrix_vphi(
                    particles = self.particles, 
                    rcs = self.rcs, 
                    tw_arc = self.tw_arc, # IN 4D FOR VALID MCF
                    n_turn_rcs = self.n_turn_rcs, 
                    v_rf = self.line['cavity_1_l'].voltage,
                    phase_rf = self.line['cavity_1_l'].lag,
                    frequency_rf = self.line['cavity_1_l'].frequency
            )
        
    def build_line_rcs(self):
        """Build the rcs line (start-to-end).
        If ramp is True, the fast ramp effect is taken into account.
        """
        print('----- Build RCS line -----')
        self.EB = EB_error(
            line = self.line, 
            rcs = self.rcs, 
            t_acc = self.t_acc, 
            n_turn_rcs = self.n_turn_rcs,
            method = self.method,
            RF_tuning = self.RF_tuning,
            jitter_diagonal = self.jitter_diagonal,
            s_jitter = self.s_jitter 
            )

        if self.ramp:
            self.EB.build_rcs_RMatrix_all(t_range = self.n_range)
            self.line_rcs = self.EB.rcs_RMatrix_all
            # self.EB.build_rcs_sliced_all(t_range = self.n_range)
            # self.line_rcs = self.EB.rcs_sliced_all
        else:
            self.EB.build_rcs_ref(t_range = self.n_range)
            self.line_rcs = self.EB.rcs_ref

    def insert_mon_apert(self):
        """Insert monitors and apertures in the rcs line.
        """
        
        print('----- Insert particle monitors and apertures -----')

        env = xt.Environment()
        env.import_line(
            self.line_rcs, 
            suffix_for_common_elements='', 
            line_name='line_env')

        # Extract markers from the line
        self.mon_dict = {}
        env.line_env.discard_tracker()
        all_marker = env.line_env.get_elements_of_type(xt.Marker)
        list_marker = all_marker[1][0::2] # Only keep markers at the beginning of each arc
        list_apert = []

        for marker in list_marker:
            # Particle monitor
            monitor_name = 'Monitor' + marker[8:]

            monitor = xt.ParticlesMonitor(
                start_at_turn=0, 
                stop_at_turn=self.n_tracking,
                num_particles=self.n_part, 
                _context=self.context)
            
            env.elements[monitor_name] = monitor
            self.mon_dict[monitor_name] = monitor

            # Longitudinal aperture
            apert_name = 'Apert' + marker[8:]

            apert = xt.LongitudinalLimitRect(
                        min_zeta=-self.zeta_max, 
                        max_zeta=self.zeta_max,
                        min_pzeta=-self.delta_max, 
                        max_pzeta=self.delta_max
                        )
            
            env.elements[apert_name] = apert
            list_apert.append('Apert' + marker[8:])

        list_mon = list(self.mon_dict.keys())

        for mon_name, marker_name in zip(list_mon, list_marker):
            idx = env.line_env.element_names.index(marker_name)
            env.line_env.element_names.insert(idx, mon_name)

        # for apert_name, marker_name in zip(list_apert, list_marker):
        #     idx = env.line_env.element_names.index(marker_name)
        #     env.line_env.element_names.insert(idx, apert_name)

        self.line_rcs = env.line_env
    
    def tracking_rcs(self):
        """Build tracker and tracks particles.
        """

        self.line_rcs.build_tracker(_context=self.context)

        self.line_rcs.track(
            self.particles, 
            num_turns = self.n_tracking,
            turn_by_turn_monitor = True,
            )

    def compute_emit(self):
        """Compute emittances and transmission.
        """

        part_mask = self.particles.state == 1
        id_part = self.particles.particle_id[part_mask]

        emitx = np.ravel([
            [calc_emit_x(mon, mask=np.isin(mon.particle_id[:, t], id_part), turn=t)
            for mon in self.mon_dict.values()]
            for t in range(self.n_tracking)
        ])

        emity = np.ravel([
            [calc_emit_y(mon, mask=np.isin(mon.particle_id[:, t], id_part), turn=t)
            for mon in self.mon_dict.values()]
            for t in range(self.n_tracking)
        ])

        gamma = np.ravel([
            [np.mean(mon.gamma0[:, t]) 
            for mon in self.mon_dict.values()]
            for t in range(self.n_tracking)
        ])

        emitx_norm = emitx * gamma
        emity_norm = emity * gamma

        emits_norm = np.ravel([
            [calc_emit_s_per(mon,  mask=np.isin(mon.particle_id[:, t], id_part), turn=t)
            for mon in self.mon_dict.values()]
            for t in range(self.n_tracking)
        ])

        lost_part_at_mon = np.ravel([
            [len(mon.state[:,t][mon.state[:,t] < 1]) 
            for mon in self.mon_dict.values()]
            for t in range(self.n_tracking)
        ])

        transmission = (self.n_part - np.array(lost_part_at_mon)) / self.n_part

        # n_factor = 1 if not self.n_range else int(self.n_turn_rcs * self.rcs.nb_arc/(np.max(self.n_range)+1)) 
        # mon_s = np.arange(0, self.n_turn_rcs * self.rcs.nb_arc / n_factor * self.n_tracking) * self.line.get_length()

        n = range(self.n_turn_rcs * self.rcs.nb_arc) if not self.n_range else self.n_range
        mon_s = (np.arange(n.start, n.stop, n.step) + 1) * self.line.get_length()

        self.track_res = {
            'emitx' : emitx_norm,
            'emity' : emity_norm,
            'emits' : emits_norm,
            'transmission' : transmission,
            's' : mon_s
        }

        print('----- Tracking results at injection -----')
        print(f'ex = {emitx_norm[0]}')
        print(f'ey = {emity_norm[0]}')
        print(f'es = {emits_norm[0]}')
        print('----- Tracking results at extraction -----')
        print(f'ex = {emitx_norm[-1]}')
        print(f'ey = {emity_norm[-1]}')
        print(f'es = {emits_norm[-1]}')
        print(f'tr = {transmission[-1]}')
        print('----- Growth at extraction -----')
        print(f'ex = {(emitx_norm[-1]-emitx_norm[0])/emitx_norm[0] * 100}')
        print(f'ey = {(emity_norm[-1]-emity_norm[0])/emity_norm[0] * 100}')
        print(f'es = {(emits_norm[-1]-emits_norm[0])/emits_norm[0] * 100}')
