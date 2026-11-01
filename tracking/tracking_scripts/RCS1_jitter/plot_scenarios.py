"""Compare the 4 jitter scenarios (I--, I++, I+-, I-+) at a given jitter amplitude,
together with the nominal case, from the ramp-less scan saved as JSON."""
import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tracking_scripts.RCS1_jitter.tracking_jitter import plot_scan_noE

folder = '/mnt/c/muco/code/tracking_scripts/RCS1_jitter/data_jitter/'
rcs_name = 'RCS1_32arcs'
n_turn_rcs = 17
ppm = 100            # jitter amplitude to plot (must have been scanned)
save = False

scenario_order = ['I_{--}', 'I_{++}', 'I_{+-}', 'I_{-+}']


def load_noramp(folder, rcs_name):
    """Returns res[label][ppm] = track_res (arrays) and the nominal track_res."""
    to_arrays = lambda tr: {q: np.asarray(v) for q, v in tr.items()}
    res, nominal = {}, None

    for fname in sorted(glob.glob(os.path.join(folder, f'{rcs_name}_*_noramp.json'))):
        with open(fname) as f:
            d = json.load(f)
        nominal = to_arrays(d['nominal'])
        res[d['label']] = {float(k): to_arrays(v) for k, v in d.items()
                           if k not in ('label', 'nominal')}
    return res, nominal


def plot_all_scenarios(res, nominal, ppm, rcs_circumference,
                       turn_ticks=(5, 10, 15, 17)):
    """Nominal + every available scenario at `ppm`."""
    plot_res = {'Nom.': nominal}
    for label in scenario_order:
        if label in res and float(ppm) in res[label]:
            plot_res[f'${label}$'] = res[label][float(ppm)]
    return plot_scan_noE(plot_res, nominal['s'], rcs_circumference,
                         turn_ticks=turn_ticks, ncol=3)


res, nominal = load_noramp(folder, rcs_name)
rcs_circumference = nominal['s'][-1] / n_turn_rcs

fig = plot_all_scenarios(res, nominal, ppm, rcs_circumference)
if save:
    fig.savefig(os.path.join(folder, f'{rcs_name}_all_scenarios_{ppm}ppm_noramp.png'),
                bbox_inches='tight')

plt.show()