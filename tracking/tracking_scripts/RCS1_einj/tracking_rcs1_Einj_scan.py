import numpy as np
import xtrack as xt
import xpart as xp
import json
import matplotlib.pyplot as plt
from class_geometry.ring_geometry.ring_geometry import RingGeometry
from TrackingRCS import TrackingRCS
from pathlib import Path

def finish_layout(fig, handles, labels):
    """Same axes size in every figure, legend inside the bottom margin."""
    fig.subplots_adjust(left=0.16, right=0.97, top=0.98, bottom=0.2, hspace=0.12)
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.57, -0.028), ncol=3)

def plot_Escan(res, s, rcs_circumference, e_inj, e_ext):
    """
    Emittance and transmission evolution for an energy scan,
    with the (linear) energy ramp on top.

    Parameters
    ----------
    res : dict
        Results per reference energy, keyed by p0 [GeV].
    s : array
        Longitudinal position of the monitors [m].
    arc_length : float
        Length of one arc [m], used to convert s into a number of RCS turns
        (one turn = rcs.nb_arc arcs).
    e_inj, e_ext : float
        Injection / extraction energy of the ramp [GeV].
        e_inj defaults to rcs.injection_energy.
    """

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(7, 8.4), sharex=True)

    for key in res.keys():
        ex = res[key]["emitx"]
        ey = res[key]["emity"]
        es = res[key]["emits"]
        tr = res[key]["transmission"]

        # Linear ramp starting at each reference energy
        axes[0].plot(s * 1e-3, np.linspace(float(key), float(key) + e_ext - e_inj, len(s)))
        print(float(key), float(key) + e_ext - e_inj, len(s))
        axes[1].plot(s * 1e-3, (ex - ex[0]) / ex[0] * 100, label=f"{key} GeV")
        axes[2].plot(s * 1e-3, (ey - ey[0]) / ey[0] * 100)
        axes[3].plot(s * 1e-3, (es - es[0]) / es[0] * 100)
        axes[4].plot(s * 1e-3, tr * 100)

    axes[0].set_ylabel("Energy [GeV]")
    axes[0].set_ylim(50, 600)
    axes[0].set_yticks([50, 200, 400, 600])
    axes[1].set_ylabel(r'$\Delta \epsilon_x / \epsilon_x$ [%]')
    axes[2].set_ylabel(r'$\Delta \epsilon_y / \epsilon_y$ [%]')
    axes[3].set_ylabel(r'$\Delta \epsilon_s / \epsilon_s$ [%]')
    axes[4].set_ylabel("Transmission [%]")
    axes[4].set_xlabel("s [km]")
    axes[4].ticklabel_format(axis="y", style="plain", useOffset=False)

    # Number of turns, as a second x axis below the s axis of the last subplot
    turn_length_km = rcs_circumference * 1e-3
    sec = axes[4].secondary_xaxis(
        -0.55,
        functions=(lambda x: x / turn_length_km, lambda t: t * turn_length_km),
    )
    sec.set_xticks([0, 4, 8, 16])
    sec.set_xlabel("Number of turns")

    handles, labels = axes[1].get_legend_handles_labels()
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="lower center",
    #     bbox_to_anchor=(0.57, -0.026),
    #     ncol=3,
    # )
    finish_layout(fig, handles, labels)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()

    return fig

if __name__ == '__main__':

    n_part = 25000
    n_turn_rcs = 17
    t_acc = 0.34 * 1e-3
    nemitt_x = 25e-6
    nemitt_y = 25e-6

    path = Path(__file__).resolve().parent.parent
    
    # Initial lattice

    file_geo = path / 'lattices/lattice_08_25/para_RCS1_gf_normal_qp__8_arc.json'
    file_arc = path / 'lattices/lattice_08_25/line_RCS1_gf_normal_qp_8_arc_v2.json'
        
    line_arc = xt.Line.from_json(file_arc)
    rcs = RingGeometry(file_geo)

    p0_list = [63, 120, 180, 240, 300]
    sigma_z_ref = 30.2e-12

    method = 'var_ref'
    RF_tuning = False
    adapt_part = False
    adapt_per = 80
    plt_twiss = False
    plt_sepa = True

    res = {}
    res_adapt = {}
    n_part=25000

    for p0 in p0_list:
        print(f'----- Start study of {p0} GeV -----' )

        line_arc = xt.Line.from_json(file_arc)

        gamma0 = p0 * 1e9 / xp.MUON_MASS_EV
        line_arc.particle_ref = xp.Particles(
                p0c= p0 * 1e9, 
                q0=1, 
                mass0=xp.MUON_MASS_EV)

        fac_z = (rcs.injection_energy / p0 / 1e9)**0.25

        track = TrackingRCS(
            line = line_arc,
            rcs = rcs,
            n_turn_rcs = n_turn_rcs,
            n_part = n_part,
            nemitt_x = nemitt_x,
            nemitt_y = nemitt_y,
            sigma_z = sigma_z_ref * fac_z,
            t_acc = t_acc,
            method = method,
            RF_tuning = RF_tuning,
            adapt_part = adapt_part,
            adapt_percentile = adapt_per,
            plt_twiss = plt_twiss,
            plt_sepa = plt_sepa,
        )

        # track_adapt = TrackingRCS(
        #     line = line_arc,
        #     rcs = rcs,
        #     n_turn_rcs = n_turn_rcs,
        #     n_part = n_part,
        #     nemitt_x = nemitt_x,
        #     nemitt_y = nemitt_y,
        #     sigma_z = sigma_z_ref * fac_z,
        #     t_acc = t_acc,
        #     method = method,
        #     RF_tuning = RF_tuning,
        #     adapt_part = True,
        #     adapt_percentile = adapt_per,
        #     plt_twiss = plt_twiss,
        #     plt_sepa = plt_sepa,
        # )

        res[str(p0)] = track.track_res
        # res_adapt[str(p0)] = track_adapt.track_res

    arc_length = line_arc.get_length()
    mon_s = np.arange(0, n_turn_rcs * rcs.nb_arc) * arc_length
    e_inj = rcs.injection_energy * 1e-9
    e_ext = rcs.extraction_energy * 1e-9

    fig = plot_Escan(
        res, 
        mon_s, 
        rcs_circumference=rcs.circumference, 
        e_inj=e_inj, 
        e_ext=e_ext
        )

    ad = 'adapt' if adapt_part else 'no_adapt'
    folder = '/mnt/c/muco/code/tracking_scripts/RCS1_einj/'

    out = {
        "s": mon_s.tolist(),
        "circumference": rcs.circumference,
        "res": {
            key: {
                **{
                    name: np.asarray(r[name]).tolist()
                    for name in ("emitx", "emity", "emits", "transmission")
                },
                "energy": np.linspace(
                    float(key), float(key) + e_ext - e_inj, len(mon_s)
                ).tolist(),
            }
            for key, r in res.items()
        },
    }

    with open(folder + f"energy_scan_data_{ad}.json", "w") as f:
        json.dump(out, f, indent=2)

    folder = '/mnt/c/muco/Results/figures/chap6/tracking_v2/RCS1_'
    save_fig = False
    if save_fig:
        fig.savefig(folder + f'energy_scan_{ad}_v2', bbox_inches='tight')
