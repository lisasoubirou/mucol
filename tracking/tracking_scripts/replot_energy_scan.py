"""Replot the RCS1 energy scan from the data saved in a .json file.

Needs rcs_scan_plot.py in the same folder.

Import the functions:
    from replot_energy_scan import load_data, plot_Escan

Or run directly (edit `file` / `output` in the block at the bottom).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from tracking.tracking_scripts.rcs_scan_plot import plot_scan
from pathlib import Path

def load_data(file):
    with open(file) as f:
        d = json.load(f)
    res = {
        key: {name: np.asarray(vals) for name, vals in r.items()}
        for key, r in d["res"].items()
    }
    meta = {
        "s": np.asarray(d["s"]),
        "rcs_circumference": float(d["circumference"]),
    }
    return res, meta


def plot_Escan(res, s, rcs_circumference, **kwargs):
    """Energy scan: one ramp per curve (res[key]["energy"]), legend in GeV."""
    kwargs.setdefault("energy_ticks", (50, 200, 400, 600))
    return plot_scan(
        res, s, rcs_circumference,
        energy=None,
        label=lambda key, r: f"{key} GeV",
        **kwargs,
    )


if __name__ == "__main__":
    path0 = Path(__file__).resolve().parent

    file = path0 / "RCS1_einj/energy_scan_data_no_adapt.json"   # data to plot
    output = None                                   # e.g. "fig.png" to save

    res, meta = load_data(file)
    fig = plot_Escan(res, **meta)

    if output:
        fig.savefig(output)
    plt.show()
