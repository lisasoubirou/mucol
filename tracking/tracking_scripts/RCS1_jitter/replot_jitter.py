import json
import os
import numpy as np
import matplotlib.pyplot as plt
from RCS1_jitter.tracking_jitter import plot_scenario

folder = '/mnt/c/muco/code/tracking_scripts/RCS1_jitter/data_jitter/RCS1_32arcs_I_mm_noramp.json'
rcs_name = 'RCS1_32arcs'
n_turn_rcs = 17
save = False


def load_noramp(folder, rcs_name):
    """Returns res[label][ppm] = track_res (arrays) and the nominal track_res."""
    to_arrays = lambda tr: {q: np.asarray(v) for q, v in tr.items()}
    res, nominal = {}, None

    with open(folder) as f:
            d = json.load(f)
    nominal = to_arrays(d['nominal'])
    res[d['label']] = {float(k): to_arrays(v) for k, v in d.items()
                           if k not in ('label', 'nominal')}
    return res, nominal

res, nominal = load_noramp(folder, rcs_name)
rcs_circumference = nominal['s'][-1] / n_turn_rcs   # [m], replace if you have the exact value

for label in res:
    fig = plot_scenario(res, nominal, label, rcs_circumference)
    if save:
        fig.savefig(os.path.join(folder, f'{rcs_name}_{label}_noramp_track.png'),
                    bbox_inches='tight')

plt.show()