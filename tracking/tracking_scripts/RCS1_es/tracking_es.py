import numpy as np
import xtrack as xt
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter
from class_geometry.ring_geometry.ring_geometry import RingGeometry
from TrackingRCS import TrackingRCS
from pathlib import Path
from tracking_scripts.replot_es_scan import load_data, plot_part0, plot_es_scan

if __name__ == '__main__':

    path = Path(__file__).resolve().parent.parent


    file_geo = path / 'lattices/lattice_08_25/para_RCS1_gf_normal_qp__8_arc.json'
    file_arc = path / 'lattices/lattice_08_25/line_RCS1_gf_normal_qp_8_arc_v2.json'
    line_arc = xt.Line.from_json(file_arc)
    rcs = RingGeometry(file_geo)

    sys.exit(0)

    sz_list = np.array([30, 25, 20, 15, 10]) * 1e-12
    res = {}
    res_part = {}
    n_part = 25000
    n_turn_rcs = 17
    t_acc = 0.34 * 1e-3
    nemitt_x = 25e-6
    nemitt_y = 25e-6

    for sz in sz_list:

        track = TrackingRCS(
            line = line_arc,
            rcs = rcs,
            n_turn_rcs = n_turn_rcs,
            n_part = n_part,
            nemitt_x = nemitt_x,
            nemitt_y = nemitt_y,
            sigma_z = sz,
            t_acc = t_acc,
            plt_sepa=True
        )

        res[str(sz*1e12)] = track.track_res
        res_part[str(sz*1e12)] = {}
        res_part[str(sz*1e12)]['part0'] = track.particles0
        res_part[str(sz*1e12)]['rf_bucket'] = track.rf_bucket

    arc_length = line_arc.get_length()
    mon_s = np.arange(1, n_turn_rcs*rcs.nb_arc+1) * arc_length

    fig_part0 = plt.figure(figsize=(6.4,4.8))
    machine_rf_bucket = res_part[str(sz_list[0]*1e12)]['rf_bucket']
    ufp = machine_rf_bucket.z_ufp
    xx = np.linspace(machine_rf_bucket.z_left, machine_rf_bucket.z_right, 1000)
    yy = machine_rf_bucket.separatrix(xx)
    plt.plot((xx-machine_rf_bucket.z_sfp), yy, color='tab:orange')
    plt.plot((xx-machine_rf_bucket.z_sfp), -yy, color='tab:orange')

    for key in res_part.keys():
        es = res[key]['emits'][0]
        zeta = res_part[key]['part0'].zeta
        delta = res_part[key]['part0'].delta
        plt.scatter(zeta, delta, s=8, label=f'{es:.4f} eV.s')
    plt.xlabel('z [m]')
    plt.ylabel(r'$\delta$')
    fig_part0.legend(
        loc='lower center', 
        bbox_to_anchor=(0.56, -0.06),
        ncol=3,
        markerscale=2
        )   
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()

    folder = '/mnt/c/muco/Results/figures/chap6/tracking_v2/RCS1_'
    fig_part0.savefig(folder+f'es_scan_no_adapt_part0',bbox_inches='tight')

    fig_ex_es, axes = plt.subplots(nrows=4, ncols=1, figsize=(6.4, 6.4))

    for key in res.keys():
        ex = res[key]['emitx']
        ey = res[key]['emity']
        es = res[key]['emits']
        tr = res[key]['transmission']

        axes[0].plot(mon_s * 1e-3, (ex - ex[0]) / ex[0] * 100)
        axes[1].plot(mon_s * 1e-3, (ey - ey[0]) / ey[0] * 100)
        axes[2].plot(mon_s * 1e-3, (es - es[0]) / es[0] * 100)
        axes[3].plot(mon_s * 1e-3, tr, label=f'{es[0]:.4f} eV.s')

    # Set log scale for the first three subplots

    axes[0].set_yscale('log')
    axes[0].yaxis.set_major_locator(LogLocator(base=10))
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].yaxis.set_minor_formatter(NullFormatter())  # Use NullFormatter to hide minor tick labels

    # Set y-ticks for the first subplot
    axes[0].set_yticks([1, 10, 100, 500])

    # Set labels
    axes[0].set_ylabel(r'$\Delta \epsilon_x / \epsilon_x$ [%]')
    axes[1].set_ylabel(r'$\Delta \epsilon_y / \epsilon_y$ [%]')
    axes[2].set_ylabel(r'$\Delta \epsilon_s / \epsilon_s$ [%]')
    axes[3].set_ylabel('Transmission')
    axes[3].set_xlabel('s [km]')

    # Add a single legend for all subplots on the right
    fig_ex_es.legend(
        loc='lower center', 
        bbox_to_anchor=(0.57, -0.028),
        ncol=3,
        )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()

    save_fig = False
    if save_fig:
        fig_ex_es.savefig(folder + f'es_scan_no_adapt_v2',bbox_inches='tight')

    folder = '/mnt/c/muco/code/tracking_scripts/RCS1_es/'

    e_inj = rcs.injection_energy * 1e-9
    e_ext = rcs.extraction_energy * 1e-9

    b = res_part[str(sz_list[0]*1e12)]['rf_bucket']
    xx = np.linspace(b.z_left, b.z_right, 1000)

    out = {
        "s": mon_s.tolist(),
        "energy": np.linspace(e_inj, e_ext, len(mon_s)).tolist(),
        "circumference": float(rcs.circumference),
        "separatrix": {
            "z": (xx - b.z_sfp).tolist(),
            "delta": np.asarray(b.separatrix(xx)).tolist(),
        },
        "res": {
            key: {
                name: np.asarray(r[name]).tolist()
                for name in ("emitx", "emity", "emits", "transmission")
            }
            for key, r in res.items()
        },
        "part0": {
            key: {
                "zeta": np.asarray(p["part0"].zeta).tolist(),
                "delta": np.asarray(p["part0"].delta).tolist(),
            }
            for key, p in res_part.items()
        },
    }

    with open(folder + "es_scan_no_adapt_data.json", "w") as f:
        json.dump(out, f, indent=2)

    import sys
    sys.path.append("/mnt/c/muco/code/tracking_scripts/RCS1_es") 

    res, part0, separatrix, meta = load_data("/mnt/c/muco/code/tracking_scripts/RCS1_es/es_scan_no_adapt_data.json")
    fig_part0 = plot_part0(res, part0, separatrix)
    fig_es = plot_es_scan(res, **meta)

    fig_es.savefig(folder + f'es_scan_no_adapt_v2', bbox_inches='tight')