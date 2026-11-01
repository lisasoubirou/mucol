import matplotlib.pyplot as plt
from tracking.tracking_scripts.RCS1_matching.plot_match import load_match, plot_match
from pathlib import Path

path0 = Path(__file__).resolve().parent
path = path0 / 'tracking_res_matching.json'
n_turn_rcs = 17
rcs_circumference = None   # [m]; None -> s[-1] / n_turn_rcs
save = False

data = load_match(path)
fig = plot_match(data, n_turn_rcs=n_turn_rcs, rcs_circumference=rcs_circumference)

if save:
    fig.savefig(path.replace('.json', '.png'), bbox_inches='tight')

plt.show()