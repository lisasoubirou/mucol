"""Shared plot for the RCS1 scans (energy scan and bunch-length scan).

Both replot scripts call plot_scan, so the two figures have the same
layout, margins and axes: energy ramp, emittances, transmission, and a
second x axis with the number of turns.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator


def plot_scan(
    res,
    s,
    rcs_circumference,
    energy=None,
    label=None,
    turn_ticks=(5, 10, 15, 17),
    energy_ticks=None,
    linthresh=1,
    yticks_x=(1, 10, 100, 500),
):
    """
    Parameters
    ----------
    res : dict
        Results per curve: res[key] has emitx, emity, emits, transmission
        (and energy, if `energy` is None).
    s : array
        Longitudinal position of the monitors [m].
    rcs_circumference : float
        RCS circumference  [m]
    energy : array or None
        Single energy ramp shared by all curves (black line). If None, the
        ramp of each curve is read from res[key]["energy"].
    label : callable or None
        label(key, res[key]) -> legend text. Defaults to str(key).
    turn_ticks : tuple
        Ticks of the number-of-turns axis.
    energy_ticks : tuple or None
        Fixed ticks of the energy axis (ylim set to first/last). None = auto.
    linthresh, yticks_x : float, tuple or None
        Symlog scale of the eps_x axis. yticks_x=None gives a linear axis.
    """
    if label is None:
        label = lambda key, r: str(key)

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(6.4, 8.4), sharex=True)
    x = s * 1e-3

    if energy is not None:
        axes[0].plot(x, energy, color="k")

    for key, r in res.items():
        ex, ey, es, tr = r["emitx"], r["emity"], r["emits"], r["transmission"]

        if energy is None:
            axes[0].plot(x, r["energy"])
        axes[1].plot(x, (ex - ex[0]) / ex[0] * 100, label=label(key, r))
        axes[2].plot(x, (ey - ey[0]) / ey[0] * 100)
        axes[3].plot(x, (es - es[0]) / es[0] * 100)
        axes[4].plot(x, tr * 100)

    # Energy axis
    axes[0].set_ylabel("Energy [GeV]")
    if energy_ticks is not None:
        axes[0].set_ylim(energy_ticks[0], energy_ticks[-1])
        axes[0].set_yticks(list(energy_ticks))

    # eps_x axis: symlog with fixed ticks
    ax = axes[1]
    if yticks_x is not None:
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.yaxis.set_major_locator(FixedLocator(yticks_x))
        ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in yticks_x]))
        ax.yaxis.set_minor_locator(NullLocator())

    axes[1].set_ylabel(r"$\Delta \epsilon_x / \epsilon_x$ [%]")
    axes[2].set_ylabel(r"$\Delta \epsilon_y / \epsilon_y$ [%]")
    axes[3].set_ylabel(r"$\Delta \epsilon_s / \epsilon_s$ [%]")
    axes[4].set_ylabel("Transmission [%]")
    axes[4].set_xlabel("s [km]")
    axes[4].ticklabel_format(axis="y", style="plain", useOffset=False)

    # Number of turns, as a second x axis below the s axis of the last subplot
    turn_length_km = rcs_circumference * 1e-3
    sec = axes[4].secondary_xaxis(
        -0.5,
        functions=(lambda v: v / turn_length_km, lambda t: t * turn_length_km),
    )
    sec.set_xticks(list(turn_ticks))
    sec.set_xlabel("Number of turns")

    # Fixed margins and legend inside the bottom margin: same size every time
    fig.subplots_adjust(left=0.16, right=0.97, top=0.98, bottom=0.22, hspace=0.12)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.57, 0.0), ncol=3)

    return fig
