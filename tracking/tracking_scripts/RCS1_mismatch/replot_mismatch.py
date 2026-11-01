import matplotlib.pyplot as plt
from plot_mismatch import load_2rcs, plot_emit_2rcs_turns
from pathlib import Path

path0 = Path(__file__).resolve().parent

path = path0 / 'ramp_vs_noramp_32arc.json'
rcs_circumference = None   # [m]; None -> s[-1] / n_turn_rcs
save = False

n_turn_rcs, curves = load_2rcs(path)
(_, label1, res1), (_, label2, res2) = curves[:2]

fig = plot_emit_2rcs_turns(res1, res2, label1, label2,
                           rcs_circumference=rcs_circumference,
                           n_turn_rcs=n_turn_rcs)

if save:
    fig.savefig(path.replace('.json', '.png'), dpi=300)

plt.show()