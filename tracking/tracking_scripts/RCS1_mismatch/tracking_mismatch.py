import numpy as np
import xtrack as xt
import xpart as xp
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter
from class_geometry.ring_geometry.ring_geometry import RingGeometry
from TrackingRCS import TrackingRCS
import os
from tracking_scripts.RCS1_mismatch.plot_mismatch import save_2rcs, plot_ramp_vs_noramp

plt.rc('axes', labelsize=14)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize

import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":

    n_part = 5000
    n_turn_rcs = 17
    t_acc = 0.34 * 1e-3
    nemitt_x = 25e-6
    nemitt_y = 25e-6

    input_rcs = {
        '8arc_noramp_nomEl' : {
            'file_input' : '/mnt/c/muco/code/lattices/lattice_08_25/line_RCS1_gf_normal_qp_8_arc_v2.json',
            'file_geo' : '/mnt/c/muco/code//lattices/lattice_08_25/para_RCS1_gf_normal_qp__8_arc.json',
            'sz' : 30.2e-12,
            'ramp' : False,
            'label' : r'8a, No ramp, $\epsilon_s$',
        },
        '8arc_noramp_low_el' : {
            'file_input' : '/mnt/c/muco/code/lattices/lattice_08_25/line_RCS1_gf_normal_qp_8_arc_v2.json',
            'file_geo' : '/mnt/c/muco/code/lattices/lattice_08_25/para_RCS1_gf_normal_qp__8_arc.json',
            'sz' : 30.2e-12/3,
            'ramp' : False,
            'label' : r'8a, No ramp, $\epsilon_s$/10',
        },
        '8arc_ramp_low_el' : {
            'file_input' : '/mnt/c/muco/code/lattices/lattice_08_25/line_RCS1_gf_normal_qp_8_arc_v2.json',
            'file_geo' : '/mnt/c/muco/code/lattices/lattice_08_25/para_RCS1_gf_normal_qp__8_arc.json',
            'sz' : 30.2e-12/3,
            'ramp' : True,
            'label' : r'8a, Ramp, $\epsilon_s$/10',
        },
        '28arc_noramp_nomEl' : {
            'file_input' : '/mnt/c/muco/code/lattices/lattice_08_25/line_RCS1_gf_normal_qp_28_arc.json',
            'file_geo' : '/mnt/c/muco/code/lattices/lattice_08_25/para_RCS1_gf_normal_qp__28_arc.json',
            'sz' : 30.2e-12,
            'ramp' : False,
            'label' : r'28a, No ramp, $\epsilon_s$',
        },
        '28arc_ramp_nomEl' : {
            'file_input' : '/mnt/c/muco/code/lattices/lattice_08_25/line_RCS1_gf_normal_qp_28_arc.json',
            'file_geo' : '/mnt/c/muco/code/lattices/lattice_08_25/para_RCS1_gf_normal_qp__28_arc.json',
            'sz' : 30.2e-12,
            'ramp' : True,
            'label' : r'28a, Ramp, $\epsilon_s$',
        },
        '32arc_noramp_nomEl' : {
            'file_input' : '/mnt/c/muco/code/lattices/lattices_final_report/RCS1_gf/line_RCS1_gf_normal_qp_32_arc.json',
            'file_geo' : '/mnt/c/muco/code/lattices/lattices_final_report/RCS1_gf/para_RCS1_gf_normal_qp__32_arc.json',
            'sz' : 30.2e-12,
            'ramp' : False,
            'label' : r'32a, No ramp, $\epsilon_s$',
        },
        '32arc_ramp_nomEl' : {
            'file_input' : '/mnt/c/muco/code/lattices/lattices_final_report/RCS1_gf/line_RCS1_gf_normal_qp_32_arc.json',
            'file_geo' : '/mnt/c/muco/code/lattices/lattices_final_report/RCS1_gf/para_RCS1_gf_normal_qp__32_arc.json',
            'sz' : 30.2e-12,
            'ramp' : True,
            'label' : r'32a, Ramp, $\epsilon_s$',
        },
    }

    keys = [
        '8arc_noramp_low_el', '8arc_ramp_low_el', 
        '32arc_noramp_nomEl', '32arc_ramp_nomEl'
        ]
    adapt = [70, 95]
    n_sigma = 5
    method = 'var_ref'
    RF_tuning = False
    plt_twiss = False
    plt_sepa = True

    res = {}
    for key_rcs in keys:
        print(f'Study for {key_rcs}')
        file_input = input_rcs[key_rcs]['file_input']
        file_geo = input_rcs[key_rcs]['file_geo']

        line_arc = xt.Line.from_json(file_input)
        rcs = RingGeometry(file_geo)

        ramp = input_rcs[key_rcs]['ramp']
        sigma_z = input_rcs[key_rcs]['sz']

        track = TrackingRCS(
            line = line_arc,
            rcs =rcs,
            n_turn_rcs = n_turn_rcs,
            n_part=n_part,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            sigma_z=sigma_z,
            t_acc = t_acc,
            ramp = ramp,
            n_sigma = n_sigma,
            method = method,
            RF_tuning = RF_tuning,
            plt_twiss = plt_twiss,
            plt_sepa = plt_sepa,
            adapt_part = True,
            adapt_percentile=95
        )

        res[key_rcs] = track

    folder = '/mnt/c/muco/code/tracking_scripts/RCS1_mismatch/'
    save_2rcs(res, input_rcs, n_turn_rcs, folder + 'ramp_vs_noramp_32arc.json')


    pairs = {
        '8arc': ('8arc_noramp_low_el', '8arc_ramp_low_el'),
        '32arc': ('32arc_noramp_nomEl', '32arc_ramp_nomEl'),
    }
    tracks = {k: t.track_res for k, t in res.items()}
    figs = plot_ramp_vs_noramp(tracks, pairs, folder, n_turn_rcs=n_turn_rcs)