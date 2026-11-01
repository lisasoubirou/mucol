import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import xtrack as xt

from class_geometry.ring_geometry.ring_geometry import RingGeometry
from EB_mismatch.EB_error import EB_error
import optics.plot_optics as po
from pathlib import Path


plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

class EBPlotter:
    """
    Class to generate all diagnostic and comparison plots
    from an EB_error object.
    """

    def __init__(self,
        arc : xt.Line,
        rcs : RingGeometry,
        rcs_name : str ,
        t_acc : float,
        n_turn_rcs : int,
        n_slice : int = 10,
        method : str = 'var_ref',
        colorcode : dict = None,
        s_jitter : float = 0,
        RF_tuning : bool = False,
        ):
        """Initialize input parameters for EBPlotter

        Args:
            arc (xt.Line): Reference arc line.
            rcs (RingGeometry): Geometry object of the line.
            rcs_name (str): RCS name.
            t_acc (float): Acceleration time [s].
            n_turn_rcs (int): Number of turns in the RCS.
            n_slice (int, optional): Number of slices for pulsed elements. Defaults to 10.
            colorcode (dict, optional): Color code for plotting. A default is already set.
        """
        # Initialize EB_error instance
        self.EB = EB_error(
            line = arc, 
            rcs = rcs, 
            t_acc = t_acc,
            n_turn_rcs = n_turn_rcs, 
            n_slice = n_slice,
            method = method,
            s_jitter = s_jitter,
            RF_tuning = RF_tuning,
            )

        self.arc = arc
        self.rcs = rcs
        self.rcs_name = rcs_name
        self.t_acc = t_acc
        self.n_turn_rcs = n_turn_rcs
        self.n_slice = n_slice

        if colorcode: # Color code for plotting
            self.cc = colorcode
        else:
            self.cc = {
                'dip_sliced' : 'tab:blue',
                'dip_pert' : 'tab:red',
                'dip_quad_sliced' : 'tab:orange',
                'quad_av' : 'tab:green',
                'no_ramp' : 'k'
            }

    def build_arcs(self):
        """Build all arcs used in the plotting."""
        EB = self.EB
        self.tw_noramp = self.arc.twiss(matrix_stability_tol=0.1)
        self.arc1_sliced, self.tw_sliced = EB.build_first_arc('sliced', 0)
        self.arc1_sliced_all, self.tw_sliced_all = EB.build_first_arc('sliced_all', 0)
        self.arc1_Rmatrix, self.tw_Rmatrix = EB.build_first_arc('Rmatrix', 0)
        self.arc1_Rmatrix_all, self.tw_Rmatrix_all = EB.build_first_arc('Rmatrix_all', 0)

    def plot_arc_traj(self, plot_px : bool = False):
        """1 arc: plot trajectory for Sliced and Rmatrix method"""

        tw1 = self.tw_sliced_all
        tw2 = self.tw_Rmatrix_all
        mask1, mask2 = mask_common_s(tw1 = tw1, tw2 = tw2)

        self.fig_arc_x = plt.figure(figsize=(6.4, 4.8))
        plt.plot(
            tw1.s[mask1], 
            tw1.x[mask1] * 1e3, 
            label='Sliced', 
            color='tab:blue')
        plt.scatter(
            tw2.s[mask2], 
            tw2.x[mask2] * 1e3,
            label=r'Analytical model', 
            marker='+', 
            color='tab:red')
        plt.xlabel('s [m]')
        plt.ylabel('x [mm]')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

        self.fig_arc_diffx = plt.figure(figsize=(6.4, 4.8))
        plt.plot(
            tw1.s[mask1], 
            (tw1.x[mask1]-tw2.x[mask2]) * 1e3, 
            label='Sliced', 
            color='tab:blue')
        plt.xlabel('s [m]')
        plt.ylabel('x [mm]')
        plt.tight_layout()
        plt.show()

        if plot_px:
            self.fig_arc_px =plt.figure(figsize=(6.4, 4.8))
            plt.plot(
                tw1.s[mask1], 
                tw1.px[mask1], 
                label='Sliced', 
                color='tab:blue')
            plt.scatter(
                tw2.s[mask2], 
                tw2.x[mask2],
                label=r'Analytical model', 
                marker='+', 
                color='tab:red')
            plt.xlabel('s [m]')
            plt.ylabel("x'")
            plt.legend(loc='upper center')
            plt.tight_layout()
            plt.show()
        
        self.fig_arc_zeta = plt.figure(figsize=(6.4, 4.8))
        plt.plot(
            tw1.s[mask1], 
            tw1.zeta[mask1]* 1e3, 
            label='Sliced', 
            color='tab:blue')
        plt.scatter(
            tw2.s[mask2],
            tw2.zeta[mask2]* 1e3,
            label=r'Analytical model', 
            marker='+',
            color='tab:red'
        )
        plt.xlabel('s [m]')
        plt.ylabel("z [mm]")
        plt.legend(loc='upper center')
        plt.tight_layout()
        plt.show()

        self.fig_arc_diffzeta = plt.figure(figsize=(6.4, 4.8))
        plt.plot(
            tw1.s[mask1], 
            (tw1.zeta[mask1]-tw2.zeta[mask2])* 1e3, 
        )
        plt.xlabel('s [m]')
        plt.ylabel("z [mm]")
        plt.tight_layout()
        plt.show()

    def plot_arc_dip_vs_quad(self, plot_px : bool = False):
        """Compare Dip pulsed vs Dip + Quad pulsed trajectories."""

        self.fig_arc_dip_quad_x = plt.figure(figsize=(6.4, 4.8))
        plt.plot(
            self.tw_sliced.s, 
            self.tw_sliced.x * 1e3, 
            label='Dip pulsed', 
            color='tab:blue')
        plt.plot(
            self.tw_sliced_all.s, 
            self.tw_sliced_all.x * 1e3,
            label='Dip + Quad pulsed', 
            color='tab:orange')
        plt.xlabel('s [m]')
        plt.ylabel('x [mm]')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

        if plot_px:

            self.fig_arc_dip_quad_px = plt.figure(figsize=(6.4, 4.8))
            plt.plot(
                self.tw_sliced.s, 
                self.tw_sliced.px, 
                label='Dip pulsed', 
                color='tab:blue')
            plt.plot(
                self.tw_sliced_all.s, 
                self.tw_sliced_all.px,
                label='Dip + Quad pulsed', 
                color='tab:orange')
            plt.xlabel('s [m]')
            plt.ylabel("x'")
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.show()

    def plot_3twiss(self):
        """Plot beta and dispersion for no ramp, dip, and dip+quad cases."""
        labels = ['No pulsed magnets', 'Dip pulsed', 'Dip + Quad pulsed']

        fig = plt.figure(figsize=(10, 6))
        betx = plt.subplot(3, 1, 1)
        bety = plt.subplot(3, 1, 2, sharex=betx)
        disp = plt.subplot(3, 1, 3, sharex=betx)

        betx.plot(self.tw_noramp.s, self.tw_noramp.betx, linestyle='dashed', color=self.cc['no_ramp'], label=labels[0])
        betx.plot(self.tw_sliced.s,self.tw_sliced.betx, label=labels[1], color=self.cc['dip_sliced'])
        betx.plot(self.tw_sliced_all.s, self.tw_sliced_all.betx, label=labels[2], color=self.cc['dip_quad_sliced'])

        bety.plot(self.tw_noramp.s, self.tw_noramp.bety, linestyle='dashed', color=self.cc['no_ramp'])
        bety.plot(self.tw_sliced.s, self.tw_sliced.bety, color=self.cc['dip_sliced'])
        bety.plot(self.tw_sliced_all.s, self.tw_sliced_all.bety, color=self.cc['dip_quad_sliced'])

        disp.plot(self.tw_noramp.s, self.tw_noramp.dx, linestyle='dashed', color=self.cc['no_ramp'])
        disp.plot(self.tw_sliced.s, self.tw_sliced.dx, color=self.cc['dip_sliced'])
        disp.plot(self.tw_sliced_all.s, self.tw_sliced_all.dx, color=self.cc['dip_quad_sliced'])

        betx.set_ylabel(r'$\Delta \beta_{x} / \beta_{x}$')
        bety.set_ylabel(r'$\Delta \beta_{y} / \beta_{y}$')
        disp.set_ylabel(r'$\Delta D_{x}$ [m]')
        disp.set_xlabel('s [m]')
        fig.legend(loc='upper center', bbox_to_anchor=(0.48, 0.88))
        plt.subplots_adjust(right=0.8)
        plt.show()

    def plot_twiss_comparisons(self):
        """Plot comparisons for Dip vs Dip+Quad and Sliced vs Mean quadrupoles."""
        mref1, msliced = mask_common_s(self.tw_noramp, self.tw_sliced)
        mref2, m_qd_dip = mask_common_s(self.tw_noramp, self.tw_sliced_all)

        # Compute differences
        def compute_deltas(tw_a, mask_a, tw_b, mask_b):
            dbetx = (tw_b.betx[mask_b] - tw_a.betx[mask_a]) / tw_a.betx[mask_a]
            dbety = (tw_b.bety[mask_b] - tw_a.bety[mask_a]) / tw_a.bety[mask_a]
            ddx = (tw_b.dx[mask_b] - tw_a.dx[mask_a]) / np.sqrt(tw_a.betx[mask_a])
            return dbetx, dbety, ddx

        dbetax_sliced, dbetay_sliced, ddx_sliced = compute_deltas(self.tw_noramp, mref1, self.tw_sliced, msliced)
        dbetax_sliced_all, dbetay_sliced_all, ddx_sliced_all = compute_deltas(self.tw_noramp, mref2, self.tw_sliced_all, m_qd_dip)

        # --- Plot Dip vs Dip+Quad ---
        fig = plt.figure(figsize=(10, 8))
        fig.suptitle('Dipole pulsed vs Dipole + Quadrupole pulsed', y=0.92)
        betx = plt.subplot(4, 1, 2)
        x_traj = plt.subplot(4, 1, 1, sharex=betx)
        bety = plt.subplot(4, 1, 3, sharex=betx)
        disp = plt.subplot(4, 1, 4, sharex=betx)

        c1 = self.cc['dip_sliced']
        c2 = self.cc['dip_quad_sliced']
        # c3 = self.cc['quad_av']

        s = self.tw_noramp.s[mref1]
        x_traj.plot(self.tw_sliced.s, self.tw_sliced.x, color=c1, label='Dip')
        x_traj.plot(self.tw_sliced_all.s, self.tw_sliced_all.x, color=c2, label='Dip+Quad')
        betx.plot(s, dbetax_sliced, color=c1)
        betx.plot(s, dbetax_sliced_all, color=c2)
        bety.plot(s, dbetay_sliced, color=c1)
        bety.plot(s, dbetay_sliced_all, color=c2)
        disp.plot(s, ddx_sliced, color=c1)
        disp.plot(s, ddx_sliced_all, color=c2)

        x_traj.set_ylabel('x [m]')
        betx.set_ylabel(r'$\Delta \beta_{x} / \beta_{x}$')
        bety.set_ylabel(r'$\Delta \beta_{y} / \beta_{y}$')
        disp.set_ylabel(r'$\Delta D_{x} \ / \sqrt{\beta_{x}} [m^{1/2}]$')
        disp.set_xlabel('s [m]')
        x_traj.legend(loc='upper right')
        plt.subplots_adjust(right=0.8)
        plt.show()

    def plot_all_arc(self):
        """Generate all plots for 1 arc."""

        print("Building arcs...")
        self.build_arcs()

        print("First arc - X : Sliced dip+quad vs RMatrix dip+quad")
        self.plot_arc_traj()

        print("First arc - X : Sliced dip vs Sliced dip+quad")
        self.plot_arc_dip_vs_quad()

        # print("Twiss comparisons")
        # self.plot_twiss_comparisons()
    
    def plot_rcs(self, tw : xt.TwissTable, fig_name : str):
        """
        Plot trajectory and optics for start-to-end.
        """

        print('Start-to-end: x')
        fig1 = plt.figure(figsize=(6.4, 4.8))
        plt.plot(tw.s * 1e-3,
                 tw.x * 1e3)
        plt.xlabel('s [km]')
        plt.ylabel('x [mm]')
        plt.tight_layout()
        plt.show()

        figz = plt.figure(figsize=(6.4, 4.8))
        plt.plot(tw.s * 1e-3,
                 tw.zeta * 1e3)
        plt.xlabel('s [km]')
        plt.ylabel('z [mm]')
        plt.tight_layout()
        plt.show()
        
        figdelta = plt.figure(figsize=(6.4, 4.8))
        plt.plot(tw.s * 1e-3,
                 tw.delta)
        plt.xlabel('s [km]')
        plt.ylabel(r'$\delta$')
        plt.tight_layout()
        plt.show()

        print('Start-to-end: optics')
        fig2 = plot_twiss_3fig(tw = tw)
        
        print('Start-to-end: optics comparison to reference')
        fig3 = compare_optics(
            tw1 = self.EB.tw_rcs_ref, 
            tw2 = tw,
            )

        setattr(self, f'fig_{fig_name}_x', fig1)
        setattr(self, f'fig_{fig_name}_zeta', figz)
        setattr(self, f'fig_{fig_name}_delta', figdelta)
        setattr(self, f'fig_{fig_name}_optics', fig2)
        setattr(self, f'fig_{fig_name}_Doptics', fig3)

    def compare_rcs(self, tw1: xt.TwissTable, tw2:  xt.TwissTable, fig_name : str):
        mask1, mask2 = mask_common_s(tw1 = tw1, tw2 = tw2)

        print('Start-to-end: x')
        fig1 = plt.figure(figsize=(6.4, 4.8))
        plt.plot(tw1.s[mask1] * 1e-3,
                 (tw1.x[mask1]-tw2.x[mask2]) * 1e3
                 )
        plt.xlabel('s [km]')
        plt.ylabel('x [mm]')
        plt.tight_layout()
        plt.show()

        fig2 = plt.figure(figsize=(6.4, 4.8))
        plt.plot(tw1.s[mask1] * 1e-3,
                 (tw1.zeta[mask1]-tw2.zeta[mask2]) * 1e3
                 )
        plt.xlabel('s [km]')
        plt.ylabel('z [mm]')
        plt.tight_layout()
        plt.show()
        
        fig3 = plt.figure(figsize=(6.4, 4.8))
        plt.plot(tw1.s[mask1] * 1e-3,
                 (tw1.delta[mask1]-tw2.delta[mask2]) * 1e3
                 )
        plt.xlabel('s [km]')
        plt.ylabel(r'$\delta$')
        plt.tight_layout()
        plt.show()

        setattr(self, f'fig_{fig_name}_diffx', fig1)
        setattr(self, f'fig_{fig_name}_diffzeta', fig2)
        setattr(self, f'fig_{fig_name}_diffdelta', fig3)

    def build_rcs_dip(self, rg : int):
        EB = self.EB
        EB.build_rcs_RMatrix(t_range=range(rg))
        EB.build_rcs_ref(t_range=range(rg))
    
    def plot_rcs_dip(self, rg : int, plot_z : bool = False):
        print('Start-to-end results: dipoles pulsed only (RMatrix)')
        try:
            self.plot_rcs(tw=self.EB.tw_rcs_RMatrix, fig_name='rcs_dip')
    
        except:
            self.build_rcs_dip(rg)
            self.plot_rcs(tw=self.EB.tw_rcs_RMatrix, fig_name='rcs_dip')

        if plot_z:       
            self.plot_zeta_rcs(tw=self.EB.tw_rcs_RMatrix, fig_name='rcs_dip')

    def build_rcs_dip_quad(self, rg : int):
        EB = self.EB
        EB.build_rcs_RMatrix_all(t_range=range(rg))
        EB.build_rcs_sliced_all(t_range=range(rg))
        EB.build_rcs_ref(t_range=range(rg))
    
    def plot_rcs_dip_quad(self, rg : int, plot_z : bool = False):
        print('Start-to-end results: dipoles + quads pulsed (RMatrix_all)')
        try:
            self.plot_rcs(tw = self.EB.tw_rcs_RMatrix_all, fig_name='rcs_all')

        except:
            self.build_rcs_dip_quad(rg)
            self.plot_rcs(tw = self.EB.tw_rcs_RMatrix_all, fig_name='rcs_all')
            self.compare_rcs(
                tw1 = self.EB.tw_rcs_RMatrix_all, 
                tw2 = self.EB.tw_rcs_sliced_all,
                fig_name='rcs_all',
                )
        
        if plot_z:
            self.plot_zeta_rcs(tw=self.EB.tw_rcs_RMatrix_all)
    
    def plot_zeta_rcs(self, tw : xt.TwissTable):
        
        print('Start-to-end: z')
        fig1 = plt.figure(figsize=(6.4, 4.8))
        plt.plot(tw.s * 1e-3,
                 tw.zeta * 1e3,
                 color = 'tab:blue')
        plt.xlabel('s [km]')
        plt.ylabel("z [mm]")
        plt.tight_layout()
        plt.show()

        mmarker = ["Marker_2" in el for el in tw.name]
        f = 1.3e9               
        def z_to_phi(z):
            return z / sc.c * f * 180 / np.pi * 1e-3

        def phi_to_z(phi):
            return phi * np.pi/180 * sc.c / f * 1e3
        
        print('Start-to-end: z at end of the arc (RF insertion)')
        fig2, ax1 = plt.subplots(figsize=(6.4, 4.8))

        # plot ONLY z
        ax1.plot(
            tw.s[mmarker] * 1e-3, 
            tw.zeta[mmarker] * 1e3, 
            color='tab:blue')
        ax1.set_xlabel("s [km]")
        ax1.set_ylabel("z [mm]")
        ax1.tick_params(axis='y')

        # add the secondary y-axis without plotting the data again
        ax2 = ax1.secondary_yaxis('right', functions=(z_to_phi, phi_to_z))
        ax2.set_ylabel("φ [°]")

        plt.tight_layout()
        plt.show()

        fig3 = plt.figure(figsize=(6.4, 4.8))
        plt.plot(tw.s[mmarker] * 1e-3, 
                    tw.delta[mmarker]*100, 
                    color='tab:blue')
        plt.xlabel("s [km]")
        plt.ylabel(r"$\delta$ [%]")
        plt.tight_layout()
        plt.show()
    
        setattr(self, f'fig_z_RF', fig2)
        setattr(self, f'fig_delta_RF', fig3)

if __name__ == '__main__':

    path_lattices = Path(__file__).resolve().parent.parent

    path_08_25 = path_lattices / 'lattices/lattice_08_25'
    path_esppu = path_lattices / 'lattices/lattice_consolidated_ESPPU'

    input_rcs = {
        'GF RCS1': {
            'file_arc': path_08_25 / 'line_RCS1_gf_normal_qp_8_arc_v2.json',
            'file_geo': path_08_25 / 'para_RCS1_gf_normal_qp__8_arc.json',
            'n_turn_rcs': 17,
            't_acc': 0.34e-3
        },
        'GF RCS2': {
            'file_arc': path_esppu / 'GF_RCS2' / 'line_RCS2_gf_normal_qp_8_arc_v2.json',
            'file_geo': path_esppu / 'GF_RCS2' / 'para_RCS2_gf_normal_qp__8_arc.json',
            'n_turn_rcs': 55,
            't_acc': 1.1e-3
        },
        'GF RCS3': {
            'file_arc': path_esppu / 'GF_RCS3' / 'line_RCS3_gf_normal_qp_8_arc.json',
            'file_geo': path_esppu / 'GF_RCS3' / 'para_RCS3_gf_normal_qp__8_arc.json',
            'n_turn_rcs': 66,
            't_acc': 2.37e-3
        },
        'GF RCS4': {
            'file_arc': path_esppu / 'GF_RCS4' / 'line_RCS4_gf_normal_qp_8_arc.json',
            'file_geo': path_esppu / 'GF_RCS4' / 'para_RCS4_gf_normal_qp__8_arc.json',
            'n_turn_rcs': 55,
            't_acc': 6.37e-3
        }
    }

    rcs_name = 'GF RCS1'
    file_line = input_rcs[rcs_name]['file_arc']
    file_geo = input_rcs[rcs_name]['file_geo']
    n_turn_rcs = input_rcs[rcs_name]['n_turn_rcs']
    t_acc = input_rcs[rcs_name]['t_acc']

    # path_report = path_lattices / 'lattices/lattices_final_report'
    # file_line = path_report / 'RCS1_gf' / 'line_RCS1_gf_normal_qp_32_arc.json'
    # file_geo = path_report / 'RCS1_gf' / 'para_RCS1_gf_normal_qp__32_arc.json'

    arc = xt.Line.from_json(file_line)
    tw_noramp = arc.twiss(matrix_stability_tol=0.1)
    tw_noramp = arc.twiss(matrix_stability_tol=0.1)
    tab = arc.get_table()

    rcs = RingGeometry(file_geo)

    plotter = EBPlotter(
        arc=arc, 
        rcs=rcs, 
        rcs_name=rcs_name, 
        t_acc=t_acc, 
        n_turn_rcs=n_turn_rcs, 
        method='var_ref',
        RF_tuning=False,
        )
    
    plotter.plot_all_arc()

    nb_arc_tot = rcs.nb_arc * n_turn_rcs
    rg = int(nb_arc_tot / 1)
    # rg = nb_arc_tot
    # rg = 1

    ## Plot mu+ and mu- trajectories 
    # tw = plotter.tw_sliced_all
    # plt.figure(figsize=(6.4, 4.8))
    # plt.plot(
    #     tw.s, 
    #     tw.x * 1e3, 
    #     color='tab:blue',
    #     label=r'$\mu^+$',
    #     lw=1.75
    #     )
    # plt.plot(
    #     tw.s[::-1], 
    #     tw.x * 1e3, 
    #     color='tab:red',
    #     label=r'$\mu^-$',
    #     lw=1.75
    #     )
    # plt.xlabel('s [m]')
    # plt.ylabel('x [mm]')
    # plt.legend(loc='upper center')
    # plt.tight_layout()
    # plt.show()
    
    plotter.plot_rcs_dip_quad(rg = rg, plot_z=True)
    save_fig = False
    folder = f'/mnt/c/muco/Results/figures/chap6/RCS1_model/'
    if save_fig:
        plotter.fig_arc_x.savefig(folder+f'{rcs_name}_arc_x')
        plotter.fig_arc_diffx.savefig(folder+f'{rcs_name}_arc_diffx')
        plotter.fig_arc_zeta.savefig(folder+f'{rcs_name}_arc_zeta')
        plotter.fig_arc_diffzeta.savefig(folder+f'{rcs_name}_arc_diffzeta')

        plotter.fig_rcs_all_x.savefig(folder+f'{rcs_name}_rcs_x')
        plotter.fig_rcs_all_zeta.savefig(folder+f'{rcs_name}_rcs_zeta')
        plotter.fig_rcs_all_delta.savefig(folder+f'{rcs_name}_rcs_delta')
        plotter.fig_rcs_all_Doptics.savefig(folder+f'{rcs_name}_rcs_Doptics')

        plotter.fig_rcs_all_diffx.savefig(folder+f'{rcs_name}_rcs_diffx')
        plotter.fig_rcs_all_diffzeta.savefig(folder+f'{rcs_name}_rcs_diffzeta')
        plotter.fig_rcs_all_diffdelta.savefig(folder+f'{rcs_name}_rcs_diffdelta')

    diff_comp_optics_2ref = po.compare_optics(
        tw1=plotter.EB.tw_rcs_sliced_all, 
        tw2=plotter.EB.tw_rcs_RMatrix_all
        )
    
    plotter.EB.build_rcs_RMatrix()
    comp_all_dip = po.compare_optics2ref(
            plotter.EB.tw_rcs_ref, 
            plotter.EB.tw_rcs_RMatrix_all,
            plotter.EB.tw_rcs_RMatrix,
            label=['Dip + Quad', 'Dip'] )
    if save_fig:
        diff_comp_optics_2ref.savefig(folder+f'{rcs_name}_optics_diff')
        comp_all_dip.savefig(folder+f'{rcs_name}_dip_vs_all')
