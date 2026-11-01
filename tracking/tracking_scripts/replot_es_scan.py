"""Replot the RCS1 bunch-length (emittance) scan from the data saved in a .json file.

Needs rcs_scan_plot.py in the same folder.

Import the functions:
    from replot_es_scan import load_data, plot_part0, plot_es_scan

Or run directly (edit `file` / the output paths in the block at the bottom).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tracking.tracking_scripts.rcs_scan_plot import plot_scan


def load_data(file):
    with open(file) as f:
        d = json.load(f)

    res = {
        key: {name: np.asarray(vals) for name, vals in r.items()}
        for key, r in d["res"].items()
    }
    part0 = {
        key: {name: np.asarray(vals) for name, vals in p.items()}
        for key, p in d["part0"].items()
    }
    separatrix = {name: np.asarray(vals) for name, vals in d["separatrix"].items()}
    meta = {
        "s": np.asarray(d["s"]),
        "energy": np.asarray(d["energy"]),
        "rcs_circumference": float(d["circumference"]),    }
    return res, part0, separatrix, meta


def plot_part0(res, part0, separatrix):
    """Initial longitudinal distributions over the RF bucket separatrix."""
    fig = plt.figure(figsize=(6.4, 4.8))

    plt.plot(separatrix["z"], separatrix["delta"], color="tab:orange")
    plt.plot(separatrix["z"], -separatrix["delta"], color="tab:orange")

    for key in part0.keys():
        es = res[key]["emits"][0]
        plt.scatter(part0[key]["zeta"], part0[key]["delta"], s=8, label=f"{es:.4f} eV.s")

    plt.xlabel("z [m]")
    plt.ylabel(r"$\delta$")
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.56, -0.06),
        ncol=3,
        markerscale=2,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def plot_es_scan(res, s, energy, rcs_circumference, **kwargs):
    """Bunch-length scan: single energy ramp, legend in eV.s."""
    return plot_scan(
        res, s, rcs_circumference,
        energy=energy,
        label=lambda key, r: f"{r['emits'][0]:.4f} eV.s",
        **kwargs,
    )


if __name__ == "__main__":
    path0 = Path(__file__).resolve().parent
    
    file = path0 / 'RCS1_es/es_scan_no_adapt_data.json'   # data to plot
    out_part0 = None                            # e.g. "es_scan_part0.png" to save
    out_es = None                               # e.g. "es_scan.png" to save

    res, part0, separatrix, meta = load_data(file)

    fig_part0 = plot_part0(res, part0, separatrix)
    fig_es = plot_es_scan(res, **meta)

    if out_part0:
        fig_part0.savefig(out_part0, bbox_inches="tight")
    if out_es:
        fig_es.savefig(out_es)
    plt.show()
