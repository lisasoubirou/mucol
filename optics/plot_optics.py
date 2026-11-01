import xtrack as xt
import xpart as xp
import numpy as np
import matplotlib.pyplot as plt

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

def plot_twiss(tw : xt.TwissTable):
    """ Plot 1st order twiss functions. 
    1st subplot : betx and bety.
    2nd subplot : dx and dy.

    Args:
        tw (xt.TwissTable): _description_
    """
    fig1 = plt.figure(1, figsize=(6.4, 4.8*1.5))
    bet = plt.subplot(2,1,1)
    disp = plt.subplot(2,1,2, sharex=bet)

    bet.plot(tw.s, tw.betx, label=r'$\beta_x$')
    bet.plot(tw.s, tw.bety,label=r'$\beta_y$')
    bet.set_ylabel(r'$\beta_{x,y}$ [m]')
    bet.legend()

    disp.plot(tw.s, tw.dx,label=r'$D_x$')
    disp.plot(tw.s, tw.dy,label=r'$D_y$')
    disp.set_ylabel(r'$D_{x,y}$ [m]')
    disp.set_xlabel(r"s [m]")
    disp.legend(loc='lower center')

    fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
    plt.show()

    return fig1

def plot_chroma(tw : xt.TwissTable):
    """Plot 2nd order twiss functions.
    1st subplot : Wx and Wy.
    2nd subplot : Ddx and Ddy.

    Args:
        tw (xt.TwissTable): _description_
    """
    fig1 = plt.figure(1, figsize=(6.4, 4.8*1.5))
    bet = plt.subplot(2,1,1)
    disp = plt.subplot(2,1,2, sharex=bet)

    bet.plot(tw.s, tw.wx_chrom, label=r'$W_x$')
    bet.plot(tw.s, tw.wy_chrom,label=r'$W_y$')
    bet.set_ylabel(r'$W_{x,y}$')
    bet.legend()

    disp.plot(tw.s, tw.ddx,label=r"$D'_x$")
    disp.plot(tw.s, tw.ddy,label=r"$D'_y$")
    disp.set_ylabel(r"$D'_{x,y}$ [m]")
    disp.set_xlabel(r"s [m]")
    disp.legend()

    fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
    plt.show()

    return fig1

def plot_twiss_3fig(
        tw : xt.TwissTable, 
        title:str=None,
        figsize:tuple=(6.4,6.4),
        ) -> None:
    """Plot the betx, bety and dx in 3 subplots.

    Args:
        tw (xt.TwissTable): Twiss table.
        title (str, optional): Title of the plot. Defaults to None.
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize)
    if title:
        fig.suptitle(title, x=0.55)
    axes[0].plot(tw.s * 1e-3, tw.betx, color='k')
    axes[1].plot(tw.s * 1e-3, tw.bety, color='tab:red')
    axes[2].plot(tw.s * 1e-3, tw.dx, color='tab:blue')

    axes[0].set_ylabel(r'$\beta_{x}$ [m]')
    axes[1].set_ylabel(r'$\beta_{y}$ [m]')
    axes[2].set_ylabel(r'$D_{x}$ [m]')
    
    axes[2].set_xlabel('s [km]')

    plt.tight_layout()
    plt.show()

    return fig

def mask_common_s(
        tw1:xt.TwissTable, 
        tw2:xt.TwissTable, 
        tol:float=1e-5
        ):
    """Creates masks for common s coordinates for same arc structure but different slicing/added Rmatrix.

    Args:
        tw1 (xt.TwissTable): Twiss object for method 1.
        tw2 (xt.TwissTable): Twiss object for method 2.
        tol (float, optional): Tolerance on s coordinates. Defaults to 1e-5.
    """        
    # Step 1: find approximate common "s"
    common_s = []
    for s in tw1["s"]:
        if np.any(np.abs(tw2["s"] - s) < tol):
            common_s.append(s)
    common_s = np.unique(common_s)  # remove duplicates

    # Step 2: build masks keeping only last occurrence for each common s
    mask1 = np.zeros(len(tw1), dtype=bool)
    mask2 = np.zeros(len(tw2), dtype=bool)

    for s in common_s:
        # last index in tw_pert within tolerance
        idx1 = np.where(np.abs(tw1["s"] - s) < tol)[0]
        if len(idx1) > 0:
            mask1[idx1[-1]] = True

        # last index in tw_pulsed within tolerance
        idx2 = np.where(np.abs(tw2["s"] - s) < tol)[0]
        if len(idx2) > 0:
            mask2[idx2[-1]] = True

    return(mask1, mask2)

def compare_optics(
        tw1:xt.TwissTable, 
        tw2:xt.TwissTable, 
        title:str=None,
        figsize:tuple = (6.4, 6.4),
        ) -> None:
    """Plot the comparison between optics tw1 and tw2, tw1 being the reference.

    Args:
        tw1 (xt.TwissTable): Twiss 1 (reference)
        tw2 (xt.TwissTable): Twiss 2 
        title (str, optional): Title of the plot. Defaults to None.
    """
    mask1, mask2 = mask_common_s(tw1, tw2)

    dDx = (tw1.dx[mask1] - tw2.dx[mask2]) / np.sqrt(tw1.betx[mask1])
    dbetx = (tw1.betx[mask1] - tw2.betx[mask2]) / tw1.betx[mask1]
    dbety = (tw1.bety[mask1] - tw2.bety[mask2]) / tw1.bety[mask1]
    s = tw1.s[mask1] * 1e-3

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize)
    if title:
        fig.suptitle(title, x=0.55)

    axes[0].plot(s, dbetx * 100, color='k')
    axes[1].plot(s, dbety * 100, color='tab:red')
    axes[2].plot(s, dDx, color='tab:blue')

    axes[0].set_ylabel(r'$\Delta \beta_x$ / $\beta_x$ [%]')
    axes[1].set_ylabel(r'$\Delta \beta_y$ / $\beta_y$ [%]')
    axes[2].set_ylabel(r'$\Delta D_x$ / $\sqrt{\beta_x}$' + ' [$\sqrt{m}$ ]')
    
    axes[2].set_xlabel('s [km]')

    plt.tight_layout()
    plt.show()

    return fig

def compare_optics2ref(
        tw_ref:xt.TwissTable, 
        tw1:xt.TwissTable, 
        tw2:xt.TwissTable, 
        label:list,
        figsize:tuple=(6.4,6.4),
        ):
    """Compare two optics to a reference.

    Args:
        tw_ref (xt.TwissTable): Reference twiss.
        tw1 (xt.TwissTable): Twiss 1
        tw2 (xt.TwissTable): Twiss 2
        label (list): List with the legend description.
    """
    mref_1, m1 = mask_common_s(tw_ref, tw1)

    dDx_1 = (tw_ref.dx[mref_1] - tw1.dx[m1]) / np.sqrt(tw_ref.betx[mref_1])
    dbetx_1 = (tw_ref.betx[mref_1] - tw1.betx[m1]) / tw_ref.betx[mref_1]
    dbety_1 = (tw_ref.bety[mref_1] - tw1.bety[m1]) / tw_ref.bety[mref_1]

    mref_2, m2 = mask_common_s(tw_ref, tw2)

    dDx_2 = (tw_ref.dx[mref_2] - tw2.dx[m2]) / np.sqrt(tw_ref.betx[mref_2])
    dbetx_2 = (tw_ref.betx[mref_2] - tw2.betx[m2]) / tw_ref.betx[mref_2]
    dbety_2 = (tw_ref.bety[mref_2] - tw2.bety[m2]) / tw_ref.bety[mref_2]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize)

    color1 = 'tab:blue'
    color2 = 'tab:orange'
    al = 1

    axes[0].plot(tw_ref.s[mref_1]*1e-3, dbetx_1*100, color=color2, alpha=al, label=label[0])
    axes[1].plot(tw_ref.s[mref_1]*1e-3, dbety_1*100, color=color2, alpha=al)
    axes[2].plot(tw_ref.s[mref_1]*1e-3, dDx_1, color=color2, alpha=al)

    axes[0].plot(tw_ref.s[mref_2]*1e-3, dbetx_2*100, color=color1, alpha=al, label=label[1])
    axes[1].plot(tw_ref.s[mref_2]*1e-3, dbety_2*100, color=color1, alpha=al)
    axes[2].plot(tw_ref.s[mref_2]*1e-3, dDx_2, color=color1, alpha=al)

    axes[0].set_ylabel(r'$\Delta \beta_x$ / $\beta_x$ [%]')
    axes[1].set_ylabel(r'$\Delta \beta_y$ / $\beta_y$ [%]')
    axes[2].set_ylabel(r'$\Delta D_x$ / $\sqrt{\beta_x}$' + ' [$\sqrt{m}$ ]')

    axes[2].set_xlabel('s [km]')
    axes[0].legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    return fig

def doublecompare_optics2ref(
        tw_ref:xt.TwissTable, 
        tw1:xt.TwissTable, 
        tw2:xt.TwissTable,
        figsize:tuple=(6.4,6.4),
        ):
    """Compare the difference of two optics, compared to a reference.
    Comparison of beta-beating and dispersion perturbation.

    Args:
        tw_ref (xt.TwissTable): Reference twiss.
        tw1 (xt.TwissTable): Twiss 1
        tw2 (xt.TwissTable): Twiss 2
    """

    mref_1, m1 = mask_common_s(tw_ref, tw1)

    dDx_1 = (tw_ref.dx[mref_1] - tw1.dx[m1]) / np.sqrt(tw_ref.betx[mref_1])
    dbetx_1 = (tw_ref.betx[mref_1] - tw1.betx[m1]) / tw_ref.betx[mref_1]
    dbety_1 = (tw_ref.bety[mref_1] - tw1.bety[m1]) / tw_ref.bety[mref_1]

    mref_2, m2 = mask_common_s(tw_ref, tw2)

    dDx_2 = (tw_ref.dx[mref_2] - tw2.dx[m2]) / np.sqrt(tw_ref.betx[mref_2])
    dbetx_2 = (tw_ref.betx[mref_2] - tw2.betx[m2]) / tw_ref.betx[mref_2]
    dbety_2 = (tw_ref.bety[mref_2] - tw2.bety[m2]) / tw_ref.bety[mref_2]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize)
    axes[0].plot(tw_ref.s[mref_1]*1e-3, (dbetx_1 - dbetx_2) * 100, color='k')
    axes[1].plot(tw_ref.s[mref_1]*1e-3, (dbety_1 - dbety_2) * 100, color='tab:red')
    axes[2].plot(tw_ref.s[mref_1]*1e-3, dDx_1 - dDx_2, color='tab:blue')

    # y0_min, y0_max = axes[0].get_ylim()
    # y1_min, y1_max = axes[1].get_ylim()
    # common_min = min(y0_min, y1_min)
    # common_max = max(y0_max, y1_max)

    # axes[0].set_ylim(common_min, common_max)
    # axes[1].set_ylim(common_min, common_max)

    axes[0].set_ylabel(r'$\Delta \beta_x$ / $\beta_x$ [%]')
    axes[1].set_ylabel(r'$\Delta \beta_y$ / $\beta_y$ [%]')
    axes[2].set_ylabel(r'$\Delta D_x$ / $\sqrt{\beta_x}$' + ' [$\sqrt{m}$ ]')

    axes[2].set_xlabel('s [km]')

    plt.tight_layout()
    plt.show()

    return fig

def compare_traj(
    tw1:xt.TwissTable, 
    tw2:xt.TwissTable, 
    title:str=None,
    figsize:tuple=(6.4,4.8),
    ) -> None:
    """Plot the difference in trajectories (x) between two optics, tw1 being the reference.

    Args:
        tw1 (xt.TwissTable): Twiss 1 (reference)
        tw2 (xt.TwissTable): Twiss 2 
        title (str, optional): Title of the plot. Defaults to None.
    """
    mask1, mask2 = mask_common_s(tw1, tw2)

    fig = plt.figure(figsize=figsize)
    plt.plot(tw1.s[mask1] * 1e-3, tw1.x[mask1] - tw2.x[mask2], color='k')
    plt.ylabel(r'x [m]')  
    plt.xlabel('s [km]')
    if title:
        fig.suptitle(title, y=0.95)
    plt.tight_layout()
    plt.show()

    return fig

def compare_dx(
    tw1:xt.TwissTable, 
    tw2:xt.TwissTable,
    figsize:tuple=(6.4,4.8)
    ) -> None:
    """Plot the perturbation in normalized dispersion from two optics, tw1 being the reference.

    Args:
        tw1 (xt.TwissTable): Twiss 1 (reference)
        tw2 (xt.TwissTable): Twiss 2
        
    """

    mask1, mask2 = mask_common_s(tw1, tw2)

    dDx = (tw1.dx[mask1] - tw2.dx[mask2]) / np.sqrt(tw1.betx[mask1])
    s = tw1.s[mask1] * 1e-3

    fig = plt.figure(figsize=figsize)
    plt.plot(s, dDx, color='tab:blue')
    plt.ylabel(r'$\Delta D_x$ / $\sqrt{\beta_x}$' + ' [$\sqrt{m}$ ]')
    plt.xlabel('s [km]')
    plt.tight_layout()
    plt.show()

    return fig

def compare_delta(
        tw1:xt.TwissTable, 
        tw2:xt.TwissTable, 
        title:str=None,
        figsize:tuple=(6.4,4.8)) -> None:
    """Plot the difference in delta between two optics, tw1 being the reference.

    Args:
        tw1 (xt.TwissTable): Twiss 1 (reference)
        tw2 (xt.TwissTable): Twiss 2 
        title (str, optional): Title of the plot. Defaults to None.
    """
    mask1, mask2 = mask_common_s(tw1, tw2)

    fig = plt.figure(figsize=figsize)
    plt.plot(tw1.s[mask1] * 1e-3, tw1.delta[mask1] - tw2.delta[mask2], color='k')
    plt.ylabel(r'$\delta$')  
    plt.xlabel('s [km]')
    if title:
        fig.suptitle(title, y=0.95)
    plt.tight_layout()
    plt.show()

    return fig

def compare_z(
    tw1:xt.TwissTable, 
    tw2:xt.TwissTable, 
    figsize:tuple=(6.4,4.8),
    ) -> None:
    """Plot the difference in trajectories (x) between two optics, tw1 being the reference.

    Args:
        tw1 (xt.TwissTable): Twiss 1 (reference)
        tw2 (xt.TwissTable): Twiss 2 
        title (str, optional): Title of the plot. Defaults to None.
    """
    mask1, mask2 = mask_common_s(tw1, tw2)

    fig = plt.figure(figsize=figsize)
    plt.plot(tw1.s[mask1] * 1e-3, (tw1.zeta[mask1] - tw2.zeta[mask2]), color='k')
    plt.ylabel('z [m]')  
    plt.xlabel('s [km]')
    plt.tight_layout()
    plt.show()

    return fig