
import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import xpart as xp
from class_geometry.ring_geometry.ring_geometry import RingGeometry, load_file
from hybrid_qp import length_qp_hybrid
from sklearn.linear_model import LinearRegression
from pathlib import Path



plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

step_quad = 1e-8

class GradientHFODO:
    def __init__(self, file_input, mu, sigma_delta, alpha = 8, 
                symmetric = False, gsc_max=400, option = 'var_ref', plot=False, 
                opt_geo_for_na = None, n_twiss=3):
        
        self.file_input = file_input #File input for geometry
        self.symmetric = symmetric #Symmetry of the quad (True or False)
        self.mu = mu #Phase advance
        self.sigma_delta = sigma_delta #Energy spread of the beam
        self.alpha = alpha #Ratio of SC pole field and NC pole field
        self.gsc_max = gsc_max #Maximum SC gradient for optimization
        self.option = option #Option for optics ('var_ref' or 'var_k')
        self.plot = plot # Plot the graph (True or False)

        if opt_geo_for_na is None:
            self.opt_geo = False
        else:
            self.nb_arc = opt_geo_for_na
            self.opt_geo = True
        
        if self.opt_geo:
            self.optimize_geometry()
        else:
            self.compute_geometry()

        self.match_gsc()
        self.tw_FODO_bf_matching = []
        self.t_scan_twiss = np.linspace(0, 1, n_twiss)
        for time_frac in self.t_scan_twiss:
            line_FODO = FODO_cell(
                rcs=self.rcs, 
                quad_dico=self.qp_para, 
                symmetric = self.symmetric,
                option= self.option, 
                time= time_frac
                )
            self.tw_FODO_bf_matching.append(line_FODO.twiss(method='4d'))
        if self.plot:
            self.fig_twiss_bf, ref_limits= plot_twiss_ftime(self.tw_FODO_bf_matching, self.t_scan_twiss, return_lim=True)
        self.match_gnc()

        self.tw_FODO_after_matching = []
        for time_frac in self.t_scan_twiss:
            line_FODO = FODO_cell(
                rcs=self.rcs, 
                quad_dico=self.qp_para, 
                symmetric = self.symmetric,
                option= self.option, 
                time= time_frac
                )
            self.tw_FODO_after_matching.append(line_FODO.twiss(method='4d'))
        if self.plot:
            self.fig_twiss_af = plot_twiss_ftime(self.tw_FODO_after_matching, self.t_scan_twiss, reference=ref_limits)

    def compute_geometry(self):
        self.rcs = RingGeometry(self.file_input)
        self.nb_arc = self.rcs.nb_arc
        self.nb_cell_arc = self.rcs.nb_cell_arc
        self.nb_cell_tot = self.rcs.nb_cell_rcs
        self.qp_para, self.radius_qp = length_qp_hybrid(self.rcs, n_cell = self.nb_cell_tot, mu = self.mu, sigma_delta = self.sigma_delta, alpha=self.alpha)[2:]
        self.l_sc, self.l_nc = self.qp_para['qp_sc']['length'], self.qp_para['qp_nc']['length']


    def optimize_geometry(self):
        """
        Solve the geometry for a given number of arc.
        Computes the length of quadrupoles. 
        """
        rcs = RingGeometry(file_input = self.file_input)

        # Scan the solutions
        n_cell_qpH, cell_length_qpH, qp_para_qpH, r_qp_qpH = length_qp_hybrid(
            rcs, 
            mu=self.mu, 
            sigma_delta = self.sigma_delta, 
            alpha=self.alpha
            )
        gradient_sc_qpH = qp_para_qpH['qp_sc']['gradient']
        gradient_nc_ext_qpH = qp_para_qpH['qp_nc']['gradient']
        maskH = (gradient_sc_qpH  < self.gsc_max) & (gradient_nc_ext_qpH * r_qp_qpH < 1) #Constraints on SC gradient
        
        # Find nc for a given number of arcs
        self.nb_cell_tot = max([nc for nc in n_cell_qpH[maskH] if nc % self.nb_arc == 0], default=None)
        self.nb_cell_arc = int(self.nb_cell_tot / self.nb_arc)

        #Compute QP parameter for geometry
        self.qp_para, self.radius_qp = length_qp_hybrid(
            rcs, 
            n_cell = self.nb_cell_tot, 
            mu = self.mu, 
            sigma_delta = self.sigma_delta, 
            alpha=self.alpha)[2:]
        self.l_sc, self.l_nc = self.qp_para['qp_sc']['length'], self.qp_para['qp_nc']['length']
        
        #Compute geometry
        self.rcs = RingGeometry(
            file_input = self.file_input, 
            nb_arc=self.nb_arc, 
            nb_cell_arc = self.nb_cell_arc, 
            length_short_insertion = self.l_sc + self.l_nc + rcs.dipole_spacing - 1e-4
            )
        
        if self.rcs.L_extra_arc > 0.1:
            raise ValueError('L_extra_arc > 0.1, check geometry! ' \
            'Expected geoemtry. Total number of cells: {self.nb_cell_tot}. Number of cells per arc: {self.nb_cell_arc}' \
            'Computed geoemtry. Total number of cells: { self.rcs.nb_cell_rcs}. Number of cells per arc: {self.rcs.nb_cell_arc}')

    def match_gsc(self):
        """
        Match the initial SC QP gradient, so phase advance is correct when Bnc=0 (t=0.5).
        """
        line_FODO = FODO_cell(rcs=self.rcs, quad_dico=self.qp_para, symmetric = self.symmetric,
                option= self.option, time= 0.5)
        #Attach knobs
        line_FODO.vars['ksc'] = line_FODO['qsc.f'].k1
        line_FODO.element_refs['qsc.f'].k1 = line_FODO.vars['ksc']
        line_FODO.element_refs['qsc.d'].k1 = -line_FODO.vars['ksc']
        #Match
        line_FODO.match(
            vary = [xt.VaryList(['ksc'], step=1e-8, tag='quad')],
            targets=[xt.TargetSet(qx=self.mu/2/np.pi, tol=1e-6)],
            method='4d',
            solve=True,
            )
        
        #Update SC QP gradient
        self.qp_para['qp_sc']['gradient'] = line_FODO['qsc.f'].k1*self.rcs.func_Brho(0.5)

    def match_gnc(self, n_point = 11):
        """
        Find law for NC gradient so that phase advance is constant in time.
        Approximate the exact law with linear regression, getting new gradient at t=0/1. 
        Args:
            n_point (int, optional): Number of points to fit and plot. Defaults to 11.
        """
        self.qx_unmatched = []
        self.qx_appr_lin = []
        self.qy_unmatched = []
        self.qy_appr_lin = []
        self.gnc_unmatched = []
        self.gnc_matched = []

        t = np.linspace(0, 1, n_point)
        for time_frac in t:
            Brho = self.rcs.func_Brho(time_frac)
            #Compute twiss for given t
            line_FODO = FODO_cell(
                rcs=self.rcs, 
                quad_dico=self.qp_para, 
                symmetric = self.symmetric,
                option= self.option, 
                time= time_frac
                )
            tw = line_FODO.twiss(method='4d')
            # tw.plot()
            # tw_unmatched.append(tw)
            self.qx_unmatched.append(tw.qx)
            self.qy_unmatched.append(tw.qy)

            # Attach knobs for matching
            if self.symmetric:
                self.gnc_unmatched.append(line_FODO['qnc.f1'].k1*Brho)
                line_FODO.vars['knc'] = line_FODO['qnc.f1'].k1
                line_FODO.element_refs['qnc.f1'].k1 = line_FODO.vars['knc']
                line_FODO.element_refs['qnc.d1'].k1 = -line_FODO.vars['knc']
                line_FODO.element_refs['qnc.f2'].k1 = line_FODO.vars['knc']
                line_FODO.element_refs['qnc.d2'].k1 = -line_FODO.vars['knc']
            else:
                self.gnc_unmatched.append(line_FODO['qnc.f'].k1*Brho)
                line_FODO.vars['knc'] = line_FODO['qnc.f'].k1
                line_FODO.element_refs['qnc.f'].k1 = line_FODO.vars['knc']
                line_FODO.element_refs['qnc.d'].k1 = -line_FODO.vars['knc']

            # Match
            matcher = line_FODO.match(
                    vary = [xt.VaryList(['knc'], step=1e-8, tag='quad')],
                    targets=[xt.TargetSet(qx=self.mu/2/np.pi, tol=1e-6)], #qy=0.25,
                    method='4d',
                    solve=True,
                    # matrix_stability_tol=0.1
                    )
            # matcher.step(80)
            # matcher.target_status()
            # matcher.vary_status()

            # matcher.step()
            tw = line_FODO.twiss(method='4d')
            # tw.plot()
            if self.symmetric:
                knc = line_FODO['qnc.f1'].k1
            else:
                knc = line_FODO['qnc.f'].k1
        
            self.gnc_matched.append(knc*Brho)   

        # Fit the gradient matched to get a linear approximation
        model = LinearRegression().fit(t.reshape((-1, 1)), self.gnc_matched)
        self.gnc_linear_appr = model.intercept_ + model.coef_*t

        #Update the NC gradient information
        self.qp_para['qp_nc']['gradient'] = self.gnc_linear_appr[-1]
        self.qp_para['qp_nc']['gradient_inj'] = self.gnc_linear_appr[0]

        #Compute the twiss functions and tune for the linearly appr. gradient
        for time_frac in t:
            Brho = self.rcs.func_Brho(time_frac)

            line_FODO = FODO_cell(
                rcs=self.rcs, 
                quad_dico=self.qp_para, 
                symmetric = self.symmetric,
                option=self.option, 
                time= time_frac
                )

            tw = line_FODO.twiss(method='4d')
            tw.plot()
            # tw_appr_lin.append(tw)
            self.qx_appr_lin.append(tw.qx)
            self.qy_appr_lin.append(tw.qy)

        if self.plot: 
            self.fig_matching = plt.figure(figsize=(6.4,4.8))
            plt.plot(t, self.gnc_unmatched, label='Thin lens appr.', color='tab:blue')
            plt.plot(t, self.gnc_matched, label='Xsuite matching', color='tab:orange')
            plt.plot(t, self.gnc_linear_appr, 'r+', label='Matching linear reg.' )
            plt.xlabel('Time normalized')
            plt.ylabel('Gradient [T/m]')
            plt.legend()
            plt.tight_layout()
            plt.grid()
            plt.show()

            self.fig_qx = plt.figure(figsize=(6.4,4.8))
            plt.plot(t, self.qx_unmatched, label='Thin lens appr.', color='tab:blue')
            plt.plot(t, self.qx_appr_lin, label='Matching linear reg.', color='tab:red')
            plt.xlabel('Time normalized')
            plt.ylabel(r'$Q_x$')
            plt.ylim(0.24,0.29)
            plt.legend()
            plt.tight_layout()
            plt.grid()
            plt.show()

def FODO_cell(rcs : RingGeometry, quad_dico : dict, symmetric = True,
                   option : str = 'var_ref', time : float = 0.5, t_ref : float = 0.5):
    """
    Args:
        rcs (RingGeometry): RCS object defining the geometry.
        option (str, optional): Trajectory option, either 'var_ref' or 'var_k'. Defaults to 'var_ref'.
        time (float, optional): Accelerating time. Defaults to 0.
        t_ref (float, optional): Reference time (useful if 'var_k' is used). Defaults to T_REF.

    Raises:
        ValueError: Invalid option. 

    Returns:
        _type_: Returns cell line. Xsuite format xt.line. 
    """

    #Extract quad info
    l_sc, l_nc = quad_dico['qp_sc']['length'], quad_dico['qp_nc']['length']
    g_sc = quad_dico['qp_sc']['gradient']
    k_sc = g_sc/rcs.func_Brho(time)

    g_nc_ext = quad_dico['qp_nc']['gradient']

    #If gradient_inj not precised, gradient at injection is negative value of g_nc_ext
    if 'gradient_inj' in quad_dico['qp_nc'].keys():
        g_nc_time = quad_dico['qp_nc']['gradient_inj'] + (
            g_nc_ext - quad_dico['qp_nc']['gradient_inj'])*time
    else:
        g_nc_time = g_nc_ext*(2*time-1)

    k_nc = g_nc_time/rcs.func_Brho(time)

    energy = rcs.injection_energy + (rcs.extraction_energy-rcs.injection_energy)*time

    env = xt.Environment()
    env.particle_ref = xp.Particles(p0c = energy, q0 = 1, mass0 = xp.MUON_MASS_EV)

    env.vars({  'ksc' : k_sc,
                'knc' : k_nc,
                'lsc' : l_sc,
                'lnc' : l_nc,
                'ldrift': rcs.dipole_spacing })
    
    if symmetric == False:

        qp_foc = env.new_line(components = [
                env.new('qsc.f', xt.Quadrupole, length='lsc', k1='ksc'),
                env.new('drift_qpf', xt.Drift, length='ldrift'),
                env.new('qnc.f', xt.Quadrupole, length='lnc', k1='knc'),
                ])
        
        # qp_defoc = env.new_line(components = [
        #         env.new('qnc.d',  xt.Quadrupole, length='lnc', k1='-knc'),
        #         env.new('drift_qpd', xt.Drift, length='ldrift'),
        #         env.new('qsc.d', xt.Quadrupole, length='lsc', k1='-ksc'),
        #         ])
        
        qp_defoc = env.new_line(components = [
                env.new('qsc.d', xt.Quadrupole, length='lsc', k1='-ksc'),
                env.new('drift_qpd', xt.Drift, length='ldrift'),
                env.new('qnc.d',  xt.Quadrupole, length='lnc', k1='-knc'),
                ])

    
    else: 
        qp_foc = env.new_line(components = [
                env.new('qnc.f1', xt.Quadrupole, k1='knc', length='lnc/2'),
                env.new('drift_qpf1', xt.Drift, length='ldrift'),
                env.new('qsc.f', xt.Quadrupole, k1='ksc', length='lsc'),
                env.new('drift_qpf2', xt.Drift, length='ldrift'),
                env.new('qnc.f2', xt.Quadrupole, k1='knc', length='lnc/2'),
                ])
        
        qp_defoc = env.new_line(components = [
                env.new('qnc.d1', xt.Quadrupole, k1='-knc', length='lnc/2'),
                env.new('drift_qpd1', xt.Drift, length='ldrift'),
                env.new('qsc.d', xt.Quadrupole, k1='-ksc', length='lsc'),
                env.new('drift_qpd2', xt.Drift, length='ldrift'),
                env.new('qnc.d2', xt.Quadrupole, k1='-knc', length='lnc/2'),
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

    #Construct half dipole cell
    for id, item in enumerate(rcs.pattern):
        cell_components.append( env.new(f'drift_{id}', xt.Drift, length=L_dd_path[id]) )
        
        if option == 'var_ref':
            cell_components.append(env.new(
                item+f'{id}', xt.Bend, 
                k0=list_hn[id], 
                # h=list_hn[id], 
                angle = L_dip_path[id]*list_hn[id],
                length=L_dip_path[id], 
                edge_entry_angle=epsilon[id], 
                edge_exit_angle=-epsilon[id+1]
                ))
        elif option == 'var_k':
            cell_components.append(env.new(item+f'{id}',
                xt.Bend, 
                k0=list_hn[id],
                # h=list_hn_ref[id], 
                angle=L_dip_path_ref[id]*list_hn_ref[id],
                length=L_dip_path_ref[id],
                edge_entry_angle=eps_ref[id], 
                edge_exit_angle=-eps_ref[id+1]
                ))
        else:
            raise ValueError('Invalid option: {}'.format(option))

    cell_components.append(env.new('drift_qd', xt.Drift, length=L_qd_path))

    cell = env.new_line(components = cell_components)

    fodo = qp_foc + cell + qp_defoc + cell

    return fodo

def plot_twiss_ftime(tw_fodo_list, t_scan, reference=None, return_lim = False):
    """
    Plot Twiss functions vs s for several time fractions.
    If reference is given, use its axis limits for consistency.
    
    Args:
        tw_fodo_list (list): list of twiss tables (before/after matching).
        t_scan (list/array): list of time fractions.
        reference (dict, optional): reference axis limits to reuse 
                                    {'xlim':(..), 'ylim_bet':(..), 'ylim_dx':(..)}.
    Returns:
        dict: dictionary with axis limits to be reused in another call.
    """
    # Collect data ranges if not using reference
    fac = 1.2
    if reference is None:
        all_s   = np.concatenate([tw.s for tw in tw_fodo_list])
        all_bet = np.concatenate([tw.betx for tw in tw_fodo_list])
        all_dx  = np.concatenate([tw.dx for tw in tw_fodo_list])

        # xlim     = (all_s.min(), all_s.max())
        ylim_bet = (all_bet.min()/fac, all_bet.max()*fac)
        ylim_dx  = (all_dx.min()/fac, all_dx.max()*fac)
    else:
        # xlim     = reference["xlim"]
        ylim_bet = reference["ylim_bet"]
        ylim_dx  = reference["ylim_dx"]

    # Make plot
    fig1, (bet, disp) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True)

    for i, t in enumerate(t_scan):
        bet.plot(tw_fodo_list[i].s, tw_fodo_list[i].betx, label=f't={t}')
        disp.plot(tw_fodo_list[i].s, tw_fodo_list[i].dx,   label=f't={t}')

    bet.set_ylabel(r'$\beta_{x}$ [m]')
    # bet.legend(loc='lower left')
    bet.set_ylim(ylim_bet)

    disp.set_ylabel(r'$D_{x}$ [m]')
    disp.set_xlabel(r"s [m]")
    disp.legend(loc='upper center')
    disp.set_ylim(ylim_dx)

    # bet.set_xlim(xlim)

    fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
    plt.tight_layout()
    plt.show()
    #"xlim": xlim, 
    # Return axis limits so the next call can reuse them
    if return_lim:
        return fig1, {"ylim_bet": ylim_bet, "ylim_dx": ylim_dx}
    else:
        return fig1

if __name__ == '__main__':

    path = Path(__file__).resolve().parent

    MU = np.pi/2
    n_size = 3
    rcs_name = 'RCS3_CERN'
    option = 'var_ref'

    # File input if you want to optimize the geometry
    input = load_file(path / 'FODO_cells_energy_spread.json')
    
    sigma_delta = input[rcs_name]['sig_delta_H_'+f'{n_size}']
    n_turn_rcs = input[rcs_name]['n_turn_rcs']
    file_input = input[rcs_name]['file']
    alpha = input[rcs_name]['alpha']

    rcs = RingGeometry(file_input = file_input)
    gradient_sc_max = 400
    nb_arc = 8
    symmetric = False

    gradient1 = GradientHFODO(
        file_input=file_input, 
        mu=MU, 
        sigma_delta=sigma_delta, 
        alpha=alpha,
        symmetric = symmetric, 
        gsc_max=gradient_sc_max, 
        option = 'var_ref', 
        plot=True, 
        opt_geo_for_na=nb_arc
        )

    gradient2 = GradientHFODO(
        file_input=file_input, 
        mu=MU, 
        sigma_delta=sigma_delta, 
        alpha=alpha,
        symmetric = True, 
        gsc_max=gradient_sc_max, 
        option = 'var_ref', 
        plot=True, 
        opt_geo_for_na=nb_arc
        )

    folder = '/mnt/c/muco/Results/figures/chap5/HFODO_var/'
    prename = 'hybrid_tvar'
    save_fig = False
    if save_fig:
        gradient1.fig_matching.savefig(folder+f'{prename}_{rcs_name}_gradient')
        gradient1.fig_qx.savefig(folder+f'{prename}_{rcs_name}_qx')
        gradient1.fig_twiss_bf.savefig(folder+f'{prename}_{rcs_name}_twiss_bf')
        gradient1.fig_twiss_af.savefig(folder+f'{prename}_{rcs_name}_twiss_after')
        gradient2.fig_matching.savefig(folder+f'{prename}_{rcs_name}_gradient_symm')
        gradient2.fig_qx.savefig(folder+f'{prename}_{rcs_name}_qx_symm')
        gradient2.fig_twiss_bf.savefig(folder+f'{prename}_{rcs_name}_twiss_bf_symm')
        gradient2.fig_twiss_af.savefig(folder+f'{prename}_{rcs_name}_twiss_after_symm')


    # # File input if you want to use a given geometry
    # file_input_geo = 'para_RCS3_CERN_25arc'
    
    # gradient2 = GradientHFODO(file_input=file_input_geo, mu=MU, sigma_delta=sigma_delta, 
    #                           alpha=alpha,
    #                         symmetric = symmetric, gsc_max=gradient_sc_max, 
    #                         option = 'var_ref', plot=True)