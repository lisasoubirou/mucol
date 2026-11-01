import os
import json
import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import scipy.constants as sc
from class_geometry.ring_geometry.ring_geometry import RingGeometry
from TrackingRCS import TrackingRCS

plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

scenario_labels = {
    's_jitter_plus_diag_False': 'I_{--}',
    's_jitter_plus_diag_True': 'I_{-+}',
    's_jitter_minus_diag_False': 'I_{++}',
    's_jitter_minus_diag_True': 'I_{+-}',
}

def plot_scan_noE(
    res,
    s,
    rcs_circumference,
    label=None,
    turn_ticks=(5, 10, 15, 17),
    ncol=2
):
    """
    Same format as plot_scan, without the energy panel and with linear axes.
 
    Parameters
    ----------
    res : dict
        res[key] has emitx, emity, emits, transmission.
    s : array
        Longitudinal position of the monitors [m].
    rcs_circumference : float
        RCS circumference [m].
    label : callable or None
        label(key, res[key]) -> legend text. Defaults to str(key).
    turn_ticks : tuple
        Ticks of the number-of-turns axis.
    """
    if label is None:
        label = lambda key, r: str(key)
 
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6.4, 6.9), sharex=True)
    x = s * 1e-3
 
    for key, r in res.items():
        ex, ey, es, tr = r["emitx"], r["emity"], r["emits"], r["transmission"]
 
        axes[0].plot(x, (ex - ex[0]) / ex[0] * 100, label=label(key, r))
        axes[1].plot(x, (ey - ey[0]) / ey[0] * 100)
        axes[2].plot(x, (es - es[0]) / es[0] * 100)
        axes[3].plot(x, tr * 100)
 
    axes[0].set_ylabel(r"$\Delta \epsilon_x / \epsilon_x$ [%]")
    axes[1].set_ylabel(r"$\Delta \epsilon_y / \epsilon_y$ [%]")
    axes[2].set_ylabel(r"$\Delta \epsilon_s / \epsilon_s$ [%]")
    axes[3].set_ylabel("Transmission [%]")
    axes[3].set_xlabel("s [km]")
    axes[3].ticklabel_format(axis="y", style="plain", useOffset=False)
 
    # Number of turns, as a second x axis below the s axis of the last subplot
    turn_length_km = rcs_circumference * 1e-3
    sec = axes[3].secondary_xaxis(
        -0.5,
        functions=(lambda v: v / turn_length_km, lambda t: t * turn_length_km),
    )
    sec.set_xticks(list(turn_ticks))
    sec.set_xlabel("Number of turns")
 
    # Fixed margins and legend inside the bottom margin
    fig.subplots_adjust(left=0.16, right=0.97, top=0.98, bottom=0.25, hspace=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.57, -0.04), ncol=ncol)
 
    return fig
 
def plot_scenario(res, nominal, label, rcs_circumference, turn_ticks=(5, 10, 15, 17)):
    per_ppm = res[label]
    plot_res = {'Nom.': nominal}
    plot_res.update({f'{p:g} ppm': tr for p, tr in per_ppm.items()})

    return plot_scan_noE(
        plot_res,
        nominal['s'],
        rcs_circumference,
        turn_ticks=turn_ticks,
    )


def scenario_key(s_sign, jitter_diag):
    return f"s_jitter_{'plus' if s_sign > 0 else 'minus'}_diag_{jitter_diag}"


def safe_name(label):
    """'I_{--}' -> 'I_mm' style filename-safe string."""
    return (label.replace('{', '').replace('}', '')
                 .replace('-', 'm').replace('+', 'p'))


def compute_sjit(rcs, ppm, t_acc):
    """Longitudinal jitter (m) corresponding to a field error of `ppm` ppm."""
    B_inj = rcs.dipole_families['BNC']['B'](0)
    B_ext = rcs.dipole_families['BNC']['B'](1)
    dB_ppm = B_ext * ppm * 1e-6
    return dB_ppm / (B_ext - B_inj) * t_acc * sc.c


def run_tracking(line_arc, rcs, params, s_sign, jitter_diag, ppm):
    sjit = compute_sjit(rcs, ppm, params['t_acc'])
    return TrackingRCS(
        line=line_arc,
        rcs=rcs,
        ramp=True,
        s_jitter=s_sign * sjit,
        jitter_diagonal=jitter_diag,
        adapt_part=True,
        adapt_percentile=90,
        **params,
    ).track_res


def run_scenarios(line_arc, rcs, params, ppm_list,
                  s_signs=(1, -1), jitter_diags=(False, True)):
    """
    Returns
    -------
    res : dict  res[label][ppm] = track_res
    nominal : dict  track_res without jitter
    """
    nominal = TrackingRCS(
        line=line_arc, rcs=rcs, ramp=True,
        adapt_part=True, adapt_percentile=90, **params,
    ).track_res

    res = {}
    for s_sign in s_signs:
        for jitter_diag in jitter_diags:
            label = scenario_labels[scenario_key(s_sign, jitter_diag)]
            print(f"\n--- Scenario: {label} ---")
            res[label] = {}
            for ppm in ppm_list:
                res[label][ppm] = run_tracking(
                    line_arc, rcs, params, s_sign, jitter_diag, ppm)
    return res, nominal


def save_scenarios(res, nominal, folder, rcs_name):
    """One .json per scenario label, holding all ppm values + the nominal case.
    Structure: {'nominal': {quantity: list}, '<ppm>': {quantity: list}}."""
    os.makedirs(folder, exist_ok=True)

    def to_json_dict(tr):
        out = {}
        for q, v in tr.items():
            try:
                out[q] = np.asarray(v).tolist()
            except Exception:
                pass
        return out

    for label, per_ppm in res.items():
        data = {'label': label, 'nominal': to_json_dict(nominal)}
        for ppm, tr in per_ppm.items():
            data[str(ppm)] = to_json_dict(tr)
        fname = os.path.join(folder, f'{rcs_name}_{safe_name(label)}_noramp.json')
        with open(fname, 'w') as f:
            json.dump(data, f, indent=2)
        print('saved', fname)


if __name__ == '__main__':
    file_geo = '/mnt/c/muco/code/lattices/lattices_final_report/RCS1_gf/para_RCS1_gf_normal_qp__32_arc.json'
    file_arc = '/mnt/c/muco/code/lattices/lattices_final_report/RCS1_gf/line_RCS1_gf_normal_qp_32_arc.json'
    rcs_name = 'RCS1_32arcs'
    folder = '/mnt/c/muco/code/tracking_scripts/RCS1_jitter/data_jitter'

    params = dict(
        t_acc=0.34e-3,
        n_turn_rcs=17,
        n_part=5000,
        nemitt_x=25e-6,
        nemitt_y=25e-6,
        sigma_z=30.2e-12,
    )

    ppm_list = [10, 100, 500]

    rcs = RingGeometry(file_geo)
    line_arc = xt.Line.from_json(file_arc)
    line_arc.twiss(method='6d', matrix_stability_tol=0.7)

    res, nominal = run_scenarios(line_arc, rcs, params, ppm_list, s_signs=[-1], jitter_diags=[True])
    save_scenarios(res, nominal, folder, rcs_name)

    # Circumference of one RCS turn [m]: adapt if RingGeometry exposes it directly
    rcs_circumference = rcs.circumference

    for label in res:
        fig = plot_scenario(res, nominal, label, rcs_circumference)
        fig.savefig(os.path.join(folder, f'{rcs_name}_{safe_name(label)}_track.png'),
                    bbox_inches='tight')