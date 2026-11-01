import numpy as np
import matplotlib.pyplot as plt
from class_geometry.ring_geometry.ring_geometry import RingGeometry
import xtrack as xt
import sys
from EB_mismatch.EB_error import EB_error, adjust_qx_arc
from scipy.optimize import curve_fit
from pathlib import Path

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

class EB_scaling:
    """
    Class to perform scaling studies on EB_error simulations
    for varying numbers of arcs and ramp speed (B').
    """

    def __init__(self, file_path, rcs_name, nb_arc, t_acc, n_turn_rcs, n_slice=3, tunes=None, t_factor=[1, 2, 4, 6, 8, 10]):
        """
        Initialize scaling study setup.
        """
        self.file_path = file_path
        self.rcs_name = rcs_name
        self.nb_arc = np.array(nb_arc)
        self.t_acc = t_acc
        self.n_turn_rcs = n_turn_rcs
        self.n_slice = n_slice
        self.tunes = tunes
        self.t_factor = t_factor

        # Build file_input dictionary
        self.file_input = {
            str(na): {
                'file_line': f'{file_path}/line_{rcs_name}_{str(na)}_arc.json',
                'file_geo': f'{file_path}/para_{rcs_name}__{str(na)}_arc.json'
            }
            for na in nb_arc
        }

        rcs = RingGeometry(self.file_input[str(self.nb_arc[0])]['file_geo'])
        self.injection_energy = rcs.injection_energy
        self.extraction_energy = rcs.extraction_energy

    @staticmethod
    def func_inv(x, a):
        """Inverse model: a/x"""
        return a / x

    # ------------------ Scaling on number of arcs ------------------ #
    def law_narc(self):
        """
        Study scaling with number of arcs.
        Returns res (dict of results) and traj_ext (max trajectory values).
        """
        self.results_narc = {}

        for na in self.file_input.keys():
            arc = xt.Line.from_json(self.file_input[na]['file_line'])
            tw_noramp = arc.twiss4d(matrix_stability_tol=0.1)

            if self.tunes is not None:
                arc = adjust_qx_arc(
                    line = arc, 
                    qx = self.tunes['qx'], 
                    qy = self.tunes['qy'],
                    dqx = self.tunes['dqx'],
                    dqy = self.tunes['dqy'],
                    )

            rcs = RingGeometry(self.file_input[na]['file_geo'])

            EB = EB_error(
                line = arc, 
                rcs = rcs, 
                t_acc = self.t_acc, 
                n_turn_rcs = self.n_turn_rcs)
            EB.build_rcs_pert()

            tw_pert = EB.tw_rcs_pert
            mask_pert = EB.mask_pert

            self.results_narc[na] = {
                'tw': tw_pert,
                'mask': mask_pert,
                'nb_cell_rcs': rcs.nb_cell_rcs,
                'qx': tw_noramp.qx,
                'qy': tw_noramp.qy,
                'x': np.max(np.abs(tw_pert.x)),
                'px': np.max(np.abs(tw_pert.px))
            }


    # ------------------ Scaling on B' (ramp speed) ------------------ #
    def law_bprime(self):
        """
        Study scaling with ramp rate (B').
        Returns res (dict of results) and traj_ext (max trajectory values).
        """
        file_input = path / 'lattices/lattice_08_25/line_RCS1_gf_normal_qp_8_arc_v2.json'

        arc = xt.Line.from_json(file_input)
        rcs = RingGeometry(self.file_input[str(min(self.nb_arc))]['file_geo'])

        self.results_bprime = {}

        for tf in self.t_factor:
            EB = EB_error(
                line=arc, 
                rcs=rcs, 
                t_acc=self.t_acc * tf, 
                n_turn_rcs=self.n_turn_rcs)

            EB.build_rcs_pert()
            tw_pert = EB.tw_rcs_pert
            mask_pert = EB.mask_pert

            self.results_bprime[str(tf)] = {
                'tw': tw_pert, 
                'mask': mask_pert,
                'x': np.max(np.abs(tw_pert.x)),
                'px': np.max(np.abs(tw_pert.px))
                }


    # ------------------ Plotting methods ------------------ #
    def plot_dE_vs_narc(self):
        """Plot scaling of ΔE per arc."""

        self.narc_dE = np.arange(min(self.nb_arc), max(self.nb_arc) + 2)
        self.dE_narc = (self.extraction_energy - self.injection_energy) / self.n_turn_rcs / self.narc_dE

        plt.figure()
        plt.plot(self.narc_dE, self.dE_narc * 1e-9)
        plt.xlabel('Number of arcs')
        plt.ylabel(r'$\Delta E_{arc}$ [GeV]')
        plt.xticks(self.nb_arc)
        plt.show()


    def plot_na_results(self):
        """Plot scaling results for number of arcs."""

        res = self.results_narc
        narc_keys = res.keys()

        nb_cell_rcs = np.array([res[na]['nb_cell_rcs'] for na in narc_keys])
        qx = np.array([res[na]['qx'] for na in narc_keys])
        qy = np.array([res[na]['qy'] for na in narc_keys])
        dx = np.array([res[na]['x'] for na in narc_keys])
        dpx = np.array([res[na]['px'] for na in narc_keys])

        self.results_narc_fna = {
            'na' : self.nb_arc,
            'nb_cell_rcs' : nb_cell_rcs,
            'qx' : qx,
            'qy' : qy,
            'dx' : dx,
            'dpx' : dpx
        }

        popt_na, _ = curve_fit(self.func_inv, self.nb_arc, dx)
        nb_arc_fit = np.linspace(np.min(self.nb_arc), np.max(self.nb_arc), 20)
        dx_fit = self.func_inv(nb_arc_fit, *popt_na)

        # Optional: Tune plots
        if not self.tunes:
            plt.figure()
            plt.scatter(self.nb_arc, nb_cell_rcs)
            plt.xlabel('Number of arcs')
            plt.ylabel('Total number of cells')
            plt.xticks(self.nb_arc)
            plt.show()

            plt.figure()
            plt.scatter(self.nb_arc, qx, label=r'$Q_x$')
            plt.scatter(self.nb_arc, qy, label=r'$Q_y$')
            plt.xlabel('Number of arcs')
            plt.ylabel('Q')
            plt.legend()
            plt.xticks(self.nb_arc)
            plt.show()

        # Δx scaling
        plt.figure()
        title = f'Scan on number of arcs. '
        if self.tunes:
            title += f'$Q_x={self.tunes["qx"]}, Q_y={self.tunes["qy"]}$.'
        plt.title(title)
        plt.scatter(self.nb_arc, dx * 1e3, marker='+', color='tab:red', label='Twiss')
        plt.plot(nb_arc_fit, dx_fit * 1e3, color='tab:blue',
                 label=f'Fit a/x, a={popt_na[0]:.2f}')
        plt.xlabel('Number of arcs')
        plt.ylabel(r'Max $\Delta x$ [mm]')
        plt.legend()
        plt.xticks(self.nb_arc)
        plt.show()

        # Δpx scaling
        plt.figure()
        plt.title(title)
        plt.plot(self.nb_arc, dpx)
        plt.xlabel('Number of arcs')
        plt.ylabel("Max $\Delta x'$")
        plt.xticks(self.nb_arc)
        plt.show()


    def plot_bprime_results(self):
        """Plot scaling results for ramp rate (B')."""

        res = self.results_bprime
        tf_keys = res.keys()

        dx_bp = np.array([res[key]['x'] for key in tf_keys])
        dpx_bp = np.array([res[key]['px'] for key in tf_keys])

        self.results_bprime_tf = {
            'tf' : self.t_factor,
            'dx' : dx_bp,
            'dpx' : dpx_bp
        }

        popt_bp, _ = curve_fit(self.func_inv, self.t_factor, dx_bp)
        t_fit = np.linspace(np.min(self.t_factor), np.max(self.t_factor), 20)
        fit_dx = self.func_inv(t_fit, *popt_bp)

        plt.figure()
        # plt.title(f'Scan on ramp speed. {len(self.nb_arc)} arcs.')
        plt.scatter(self.t_factor, dx_bp * 1e3, marker='+', color='tab:red', label='Twiss')
        plt.plot(t_fit, fit_dx * 1e3, color='tab:blue', label=f'Fit') #a/x, a={popt_bp[0]:.2f}
        plt.xlabel(r'f * $t_{acc}$')
        plt.ylabel(r'Max $\Delta x$ [mm]')
        plt.legend()
        plt.show()

        plt.figure()
        # plt.title(f'Scan on ramp speed. {len(self.nb_arc)} arcs.')
        plt.plot(self.t_factor, dpx_bp)
        plt.xlabel(r'f * $t_{acc}$')
        plt.ylabel("Max $\Delta x'$")
        plt.show()

    # ------------------ Master run method ------------------ #
    def run_all(self):
        """
        Run the complete scaling study and plot results.
        """
        print("Running scaling with number of arcs...")
        self.law_narc()
        self.plot_dE_vs_narc()
        self.plot_na_results()

        print("Running scaling with ramp rate (B')...")
        self.law_bprime()
        self.plot_bprime_results()


if __name__ == '__main__':

    t_acc = 0.34 * 1e-3
    n_turn_rcs = 17
    n_slice = 3

    file_input = {}
    path = Path(__file__).resolve().parent.parent
    file_path = path / 'lattices/lattice_08_25/'

    rcs_name = 'RCS1_gf_normal_qp'

    nb_arc_list = np.array([8, 12, 16, 20, 24, 28])
    tunes_match = {'qx': 1.7, 'qy': 1.7, 'dqx': 0, 'dqy' : 0}

    scaling = EB_scaling(
        file_path = file_path,
        rcs_name = rcs_name,
        nb_arc = nb_arc_list,
        t_acc = t_acc,
        n_turn_rcs = n_turn_rcs,
        n_slice = n_slice,
        tunes = tunes_match
    )

    scaling.run_all()
    