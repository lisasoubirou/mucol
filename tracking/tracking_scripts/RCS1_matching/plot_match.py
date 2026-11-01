"""Save / load / plot the matching comparison (adapt vs no_adapt, 8-arc vs 32-arc)."""
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, FixedLocator, FixedFormatter


ring_names = {
    '8arc_noramp_nomEl': '8-arc',
    '32arc_noramp_nomEl': '32-arc',
}
mode_names = {
    'adapt': '2nd order matching',
    'no_adapt': 'Linear matching',
}


def save_match(res, path):
    """res[key_rcs]['adapt' | 'no_adapt'] = TrackingRCS  ->  one json file."""
    data = {}
    for key_rcs, modes in res.items():
        data[key_rcs] = {}
        for mode, tracking in modes.items():
            data[key_rcs][mode] = {}
            for q, v in tracking.track_res.items():
                try:
                    data[key_rcs][mode][q] = np.asarray(v).tolist()
                except Exception:
                    pass
    with open(path, 'w') as f:
        json.dump(data, f)
    print('saved', path)


def load_match(path):
    """Returns data[key_rcs][mode] = track_res dict of arrays."""
    with open(path) as f:
        d = json.load(f)
    return {key_rcs: {mode: {q: np.asarray(v) for q, v in tr.items()}
                      for mode, tr in modes.items()}
            for key_rcs, modes in d.items()}


def plot_match(data, n_turn_rcs=17, rcs_circumference=None, keys=None,
               modes=('adapt', 'no_adapt'), turn_ticks=(5, 10, 15, 17),
               yticks_x=(-10, 0, 20, 100, 500), linthresh=10):
    """
    Same format as plot_scan_noE (4 panels, s [km] axis + number-of-turns
    axis below, same margins and legend). Each curve uses its own s, since the
    8-arc and 32-arc lines have different numbers of monitors.
 
    rcs_circumference : length of one RCS turn [m]. Default: s[-1] / n_turn_rcs
        of the first curve.
    yticks_x, linthresh : symlog scale of the eps_x panel (as in plot_emit_rcs).
        yticks_x=None gives a linear axis.
    """
    keys = keys or list(data)
 
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6.4, 6.9), sharex=True)
 
    s_ends = []
    for key_rcs in keys:
        for mode in modes:
            if mode not in data[key_rcs]:
                continue
            r = data[key_rcs][mode]
            name = f'{ring_names.get(key_rcs, key_rcs)}, {mode_names[mode]}'
 
            x = r["s"] * 1e-3
            s_ends.append(r["s"][-1])
            ex, ey, es, tr = r["emitx"], r["emity"], r["emits"], r["transmission"]
 
            axes[0].plot(x, (ex - ex[0]) / ex[0] * 100, label=name)
            axes[1].plot(x, (ey - ey[0]) / ey[0] * 100)
            axes[2].plot(x, (es - es[0]) / es[0] * 100)
            axes[3].plot(x, tr * 100)
 
    if not np.allclose(s_ends, s_ends[0], rtol=1e-2):
        warnings.warn(f'the curves do not end at the same s ({s_ends}): '
                      'the number-of-turns axis is only valid for one of them')
 
    # eps_x axis: symlog with fixed ticks
    if yticks_x is not None:
        ax = axes[0]
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.yaxis.set_major_locator(FixedLocator(yticks_x))
        ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in yticks_x]))
        ax.yaxis.set_minor_locator(NullLocator())
 
    axes[0].set_ylabel(r"$\Delta \epsilon_x / \epsilon_x$ [%]")
    axes[1].set_ylabel(r"$\Delta \epsilon_y / \epsilon_y$ [%]")
    axes[2].set_ylabel(r"$\Delta \epsilon_s / \epsilon_s$ [%]")
    axes[3].set_ylabel("Transmission [%]")
    axes[3].set_xlabel("s [km]")
    axes[3].ticklabel_format(axis="y", style="plain", useOffset=False)
 
    # Number of turns, as a second x axis below the s axis of the last subplot
    if rcs_circumference is None:
        rcs_circumference = s_ends[0] / n_turn_rcs
    turn_length_km = rcs_circumference * 1e-3
    sec = axes[3].secondary_xaxis(
        -0.5,
        functions=(lambda v: v / turn_length_km, lambda t: t * turn_length_km),
    )
    sec.set_xticks(list(turn_ticks))
    sec.set_xlabel("Number of turns")
 
    fig.subplots_adjust(left=0.16, right=0.97, top=0.98, bottom=0.25, hspace=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.57, -0.04), ncol=2)
 
    return fig