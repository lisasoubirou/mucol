"""Save / load / plot the ramp vs no-ramp comparison of two RCS configurations."""
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)


def save_2rcs(res, input_rcs, n_turn_rcs, path):
    """res[key_rcs] = TrackingRCS  ->  one json file with labels and track_res."""
    data = {'n_turn_rcs': n_turn_rcs, 'keys': list(res), 'configs': {}}
    for key_rcs, tracking in res.items():
        tr = {}
        for q, v in tracking.track_res.items():
            try:
                tr[q] = np.asarray(v).tolist()
            except Exception:
                pass
        data['configs'][key_rcs] = {'label': input_rcs[key_rcs]['label'], 'track_res': tr}
    with open(path, 'w') as f:
        json.dump(data, f)
    print('saved', path)


def load_2rcs(path):
    """Returns n_turn_rcs and an ordered list of (key_rcs, label, track_res)."""
    with open(path) as f:
        d = json.load(f)
    curves = [(k, d['configs'][k]['label'],
               {q: np.asarray(v) for q, v in d['configs'][k]['track_res'].items()})
              for k in d['keys']]
    return d['n_turn_rcs'], curves


def plot_emit_2rcs_turns(res1, res2, label1="RCS 1", label2="RCS 2",
                         rcs_circumference=None, n_turn_rcs=17,
                         turn_ticks=(5, 10, 15, 17)):
    """
    Same as plot_emit_2rcs (4 panels, tab:blue / tab:orange, same legend), with
    a number-of-turns axis added below the s [km] axis.

    rcs_circumference : length of one turn [m]. Default: s[-1] / n_turn_rcs of res1.
    """
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6.4, 6.9), sharex=True)

    configs = [
        (res1, label1, "tab:blue"),
        (res2, label2, "tab:orange"),
    ]

    for res, cfg_label, cl in configs:
        ex, ey, es, tr, s = (res["emitx"], res["emity"], res["emits"],
                             res["transmission"], res["s"])

        axes[0].plot(s * 1e-3, (ex - ex[0]) / ex[0] * 100, color=cl, label=cfg_label)
        axes[1].plot(s * 1e-3, (ey - ey[0]) / ey[0] * 100, color=cl)
        axes[2].plot(s * 1e-3, (es - es[0]) / es[0] * 100, color=cl)
        axes[3].plot(s * 1e-3, tr * 100, color=cl)

    axes[0].set_ylabel(r'$\Delta \epsilon_x / \epsilon_x$ [%]')
    axes[1].set_ylabel(r'$\Delta \epsilon_y / \epsilon_y$ [%]')
    axes[2].set_ylabel(r'$\Delta \epsilon_s / \epsilon_s$ [%]')
    axes[3].set_ylabel("Transmission [%]")
    axes[3].set_xlabel("s [km]")

    # Number of turns, as a second x axis below the s axis of the last subplot
    if rcs_circumference is None:
        rcs_circumference = res1["s"][-1] / n_turn_rcs
    turn_length_km = rcs_circumference * 1e-3
    sec = axes[3].secondary_xaxis(
        -0.5,
        functions=(lambda v: v / turn_length_km, lambda t: t * turn_length_km),
    )
    sec.set_xticks(list(turn_ticks))
    sec.set_xlabel("Number of turns")

    # Fixed margins, legend inside the bottom margin (nothing cropped on save)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.98, bottom=0.25, hspace=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.57, 0.0), ncol=2)

    return fig

def plot_ramp_vs_noramp(tracks, pairs, folder, prefix='ramp_vs_noramp',
                        labels=('No ramp', 'Fast ramp'), n_turn_rcs=17,
                        rcs_circumference=None, save=True):
    """
    One figure per lattice (no ramp vs fast ramp), each saved as a png.
 
    tracks : {key_rcs: track_res dict}
    pairs  : {name: (key_noramp, key_ramp)}, e.g.
             {'8arc': ('8arc_noramp_low_el', '8arc_ramp_low_el'),
              '32arc': ('32arc_noramp_nomEl', '32arc_ramp_nomEl')}
    Files: <folder><prefix>_<name>.png (figure saved as is, no cropping).
    Returns {name: fig}.
    """
    figs = {}
    for name, (key_noramp, key_ramp) in pairs.items():
        fig = plot_emit_2rcs_turns(
            tracks[key_noramp], tracks[key_ramp], labels[0], labels[1],
            rcs_circumference=rcs_circumference, n_turn_rcs=n_turn_rcs,
        )
        if save:
            fname = f'{folder}{prefix}_{name}.png'
            fig.savefig(fname, dpi=300)
            print('saved', fname)
        figs[name] = fig
    return figs