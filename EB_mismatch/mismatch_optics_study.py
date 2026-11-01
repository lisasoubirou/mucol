import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import json
from scipy.optimize import curve_fit
from class_geometry.ring_geometry.ring_geometry import RingGeometry
from EB_mismatch.EB_error import EB_error, adjust_qx_arc
import optics.plot_optics as po
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from pathlib import Path

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize


def compute_raw_data(n_slice_list, arc, rcs, t_acc, n_turn_rcs, rcs_ref):
    """Compute and store raw masked arrays for each n_slice."""
    data = {}

    for n_slice in n_slice_list:
        print(f"Computing for n_slice = {n_slice}")

        EB = EB_error(
            line=arc,
            rcs=rcs,
            t_acc=t_acc,
            n_turn_rcs=n_turn_rcs,
            n_slice=n_slice
        )
        EB.tw_init.element_name = rcs_ref.element_names[0]
        tw_ref = rcs_ref.twiss(init=EB.tw_init)

        # Build optics
        EB.build_rcs_sliced_all()
        EB.build_rcs_RMatrix_all()
        tw1 = EB.tw_rcs_sliced_all
        tw2 = EB.tw_rcs_RMatrix_all

        # Common masks for matching elements
        mref_1, m1 = po.mask_common_s(tw_ref, tw1)
        mref_2, m2 = po.mask_common_s(tw_ref, tw2)

        # Store raw matched data
        data[n_slice] = dict(
            s=tw_ref.s[mref_1],
            betx_ref=tw_ref.betx[mref_1],
            bety_ref=tw_ref.bety[mref_1],
            dx_ref=tw_ref.dx[mref_1],

            betx_sliced=tw1.betx[m1],
            bety_sliced=tw1.bety[m1],
            dx_sliced=tw1.dx[m1],

            betx_Rmatrix=tw2.betx[m2],
            bety_Rmatrix=tw2.bety[m2],
            dx_Rmatrix=tw2.dx[m2]
        )

    return data

def compute_differences(data:dict) -> dict:
    """From data, compute Δβ/β and ΔDx/√β."""
    twiss = {}

    for n_slice, d in data.items():
        betx_ref, bety_ref, dx_ref = d["betx_ref"], d["bety_ref"], d["dx_ref"]

        # Differences
        dbetx_sliced = (betx_ref - d["betx_sliced"]) / betx_ref
        dbety_sliced = (bety_ref - d["bety_sliced"]) / bety_ref
        dDx_sliced = (dx_ref - d["dx_sliced"]) / np.sqrt(betx_ref)

        dbetx_Rmatrix = (betx_ref - d["betx_Rmatrix"]) / betx_ref
        dbety_Rmatrix = (bety_ref - d["bety_Rmatrix"]) / bety_ref
        dDx_Rmatrix = (dx_ref - d["dx_Rmatrix"]) / np.sqrt(betx_ref)

        # Differences between the two methods
        twiss[n_slice] = dict(
            s = d["s"],
            dbetx_sliced = dbetx_sliced,
            dbety_sliced = dbety_sliced,
            dDx_sliced = dDx_sliced,
            dbetx_Rmatrix = dbetx_Rmatrix,
            dbety_Rmatrix = dbety_Rmatrix,
            dDx_Rmatrix = dDx_Rmatrix,
            diff_betx = dbetx_sliced - dbetx_Rmatrix,
            diff_bety = dbety_sliced - dbety_Rmatrix,
            diff_Dx = dDx_sliced - dDx_Rmatrix
        )

    return twiss

def plot_differences(derived_data:dict, save_path=None):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6.4, 6.4))

    for n_slice, d in derived_data.items():
        s = d["s"]*1e-3
        axes[0].plot(s, d["diff_betx"], label=f"{n_slice}s", alpha=0.7)
        axes[1].plot(s, d["diff_bety"], alpha=0.7)
        axes[2].plot(s, d["diff_Dx"], alpha=0.7)

    axes[0].set_ylabel(r'$\Delta \beta_x / \beta_x$')
    axes[1].set_ylabel(r'$\Delta \beta_y / \beta_y$')
    axes[2].set_ylabel(r'$\Delta D_x / \sqrt{\beta_x}$ [√m]')
    axes[2].set_xlabel('s [km]')
    axes[0].legend(prop={'size': 9}, loc='upper right')
    plt.tight_layout()

    plt.show()

def compute_relative_results(EB, rg):
    """Construit les matrices RCS et les résultats twiss alignés en s."""
    EB.build_rcs_RMatrix_all(rg)
    tw_rcs = EB.tw_rcs_RMatrix_all

    EB.build_rcs_ref(rg)
    tw_ref = EB.tw_rcs_ref

    mask1, mask2 = po.mask_common_s(tw1=tw_ref, tw2=tw_rcs)

    return {
        's':        tw_rcs.s[mask2],
        'betx':     tw_rcs.betx[mask2],
        'bety':     tw_rcs.bety[mask2],
        'dx':       tw_rcs.dx[mask2],
        's_ref':    tw_ref.s[mask1],
        'betx_ref': tw_ref.betx[mask1],
        'bety_ref': tw_ref.bety[mask1],
        'dx_ref':   tw_ref.dx[mask1],
    }


def plot_relative_diff(results, key, key_ref, ylabel, legend_suffix=" arcs"):
    plt.figure()
    for name, dat in results.items():
        rel = (dat[key] - dat[key_ref]) / dat[key_ref]
        plt.plot(dat['s'] * 1e-3, rel, label=f"{name}{legend_suffix}")
    plt.xlabel("s [km]")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def compute_max_variations(results):
    """Retourne les variations maximales pour betx, bety, Dx."""
    bx = []
    by = []
    dx = []
    for dat in results.values():
        bx.append(np.max(np.abs((dat['betx'] - dat['betx_ref']) / dat['betx_ref'])))
        by.append(np.max(np.abs((dat['bety'] - dat['bety_ref']) / dat['bety_ref'])))
        dx.append(np.max(np.abs((dat['dx']   - dat['dx_ref'])   / dat['betx_ref'])))
    return np.array(bx), np.array(by), np.array(dx)


def fit_and_plot(func, x, bx_max, by_max, dx_max, xlabel, xticks=None):
    """Fit des variations max + affichage."""
    x_fit = np.linspace(np.min(x), np.max(x), 20)

    bx_res, _ = curve_fit(func, x, bx_max)
    by_res, _ = curve_fit(func, x, by_max)
    dx_res, _ = curve_fit(func, x, dx_max)

    bx_fit = func(x_fit, *bx_res)
    by_fit = func(x_fit, *by_res)
    dx_fit = func(x_fit, *dx_res)

    plt.figure()
    plt.scatter(x, bx_max, marker='+', color='tab:blue')
    plt.plot(x_fit, bx_fit, color='tab:blue', label=r'$\Delta \beta_x$')

    plt.scatter(x, by_max, marker='+', color='tab:orange')
    plt.plot(x_fit, by_fit, color='tab:orange', label=r'$\Delta \beta_y$')

    plt.scatter(x, dx_max * 10, marker='+', color='tab:green')
    plt.plot(x_fit, dx_fit * 10, color='tab:green', label=r'$\Delta D_x \times 10$')

    if xticks is not None:
        plt.xticks(xticks)

    plt.xlabel(xlabel)
    plt.ylabel("Relative variation")
    plt.legend()
    plt.show()

def func_inv(x, a):
    """Inverse model: a/x"""
    return a / x

def func_lin(x, a):
    """Inverse model: a/x"""
    return a * x

def compare_n_slices(
    n_slices, 
    line_arc, 
    rcs, 
    t_acc, 
    n_turn_rcs
    ):
    
    res_sl = {}
    for n in n_slices:
        print(f'Compute line and twiss for {n} slices')
        res_sl[str(n)] = {}

        EB = EB_error(
            line = line_arc, 
            rcs = rcs,  
            t_acc = t_acc, 
            n_turn_rcs = n_turn_rcs, 
            n_slice =  n
            )
        EB.build_rcs_sliced_all(rg)
        res_sl[str(n)]['twiss'] = EB.tw_rcs_sliced_all
    
    ref_id = -1
    key_ref = str(n_slices[ref_id])
    for n in n_sl[:-1]:
        print(f'Compare twiss functions for {n} slices wrt {n_slices[ref_id]} slices ')
        tw1 = res_sl[key_ref]['twiss']
        tw2 = res_sl[str(n)]['twiss']
        # tw2 = tw_rcs_RMatrix
        mask1, mask2 = po.mask_common_s(tw1, tw2)

        res_sl[str(n)]['betx'] = np.max(np.abs((tw1.betx[mask1] - tw2.betx[mask2]) / tw1.betx[mask1]))
        res_sl[str(n)]['bety'] = np.max(np.abs((tw1.bety[mask1] - tw2.bety[mask2]) / tw1.bety[mask1]))
        res_sl[str(n)]['dx'] = np.max(np.abs((tw1.dx[mask1] - tw2.dx[mask2]) / np.sqrt(tw1.betx[mask1])))

    plt.figure()
    plt.plot(
        n_slices[:-1], 
        [res_sl[str(key)]['betx'] for key in n_slices[:-1]], 
        label=r'$\beta_x$',
        )
    plt.plot(
        n_slices[:-1], 
        [res_sl[str(key)]['bety'] for key in n_slices[:-1]], 
        label=r'$\beta_y$',
        )
    plt.plot(
        n_slices[:-1], 
        [res_sl[str(key)]['dx'] for key in n_slices[:-1]], 
        label=r'$D_x$ $[\sqrt{m}]$', 
        )
    plt.xticks(n_slices[:-1])
    plt.xlabel('Number of slices')
    plt.yscale('log')
    plt.ylabel(r'Relative variation')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    path_lattices = Path(__file__).resolve().parent.parent
    path_08_25 = path_lattices / 'lattices/lattice_08_25'
    
    file_line = path_08_25 / 'line_RCS1_gf_normal_qp_8_arc_v2.json'
    file_geo = path_08_25 / 'para_RCS1_gf_normal_qp__8_arc.json'

    arc = xt.Line.from_json(file_line)
    tab = arc.get_table()
    tw_noramp = arc.twiss(
        matrix_stability_tol=0.1, 
        compute_chromatic_properties=True
        )

    rcs = RingGeometry(file_geo)
    t_acc = 0.34*1e-3
    n_turn_rcs = 17
    nb_arc_tot = n_turn_rcs*rcs.nb_arc

    EB = EB_error(
            line = arc, 
            rcs = rcs,  
            t_acc = t_acc, 
            n_turn_rcs = n_turn_rcs, 
            )

    EB_3slice = EB_error(
        line = arc, 
        rcs = rcs,  
        t_acc = t_acc, 
        n_turn_rcs = n_turn_rcs, 
        n_slice =  1
        )

    # Number of arcs to compute 
    rg = None

    # Compute the lines and twiss
    EB.build_rcs_RMatrix_all(rg)
    rcs_RMatrix, tw_rcs_RMatrix = EB.rcs_RMatrix_all, EB.tw_rcs_RMatrix_all

    test_nslice = False
    if test_nslice:
        n_sl = [1, 2, 3, 5, 10, 20]
        res_sl = {}
        for n in n_sl:
            print(f'Compute line and twiss for {n} slices')
            res_sl[str(n)] = {}
            EB = EB_error(
                line = arc, 
                rcs = rcs,  
                t_acc = t_acc, 
                n_turn_rcs = n_turn_rcs, 
                n_slice =  n
                )
            EB.build_rcs_sliced_all(rg)
            tw_rcs_pulsed_sl = EB.tw_rcs_sliced_all
            res_sl[str(n)]['twiss'] = tw_rcs_pulsed_sl

        for n in n_sl:
            tw2 = tw_rcs_RMatrix
            tw1 = res_sl[str(n)]['twiss']
            mask1, mask2 = po.mask_common_s(tw1, tw2)

            res_sl[str(n)]['betx'] = np.max(np.abs((tw1.betx[mask1] - tw2.betx[mask2]) / tw1.betx[mask1]))
            res_sl[str(n)]['bety'] = np.max(np.abs((tw1.bety[mask1] - tw2.bety[mask2]) / tw1.bety[mask1]))
            res_sl[str(n)]['dx'] = np.max(np.abs((tw1.dx[mask1] - tw2.dx[mask2]) / np.sqrt(tw1.betx[mask1])))

        fig_nslice = plt.figure()
        plt.plot(
            n_sl, 
            [res_sl[str(key)]['betx'] for key in n_sl], 
            label=r'$\beta_x$',
            )
        plt.plot(
            n_sl, 
            [res_sl[str(key)]['bety'] for key in n_sl], 
            label=r'$\beta_y$',
            )
        plt.plot(
            n_sl, 
            [res_sl[str(key)]['dx'] for key in n_sl], 
            label=r'$D_x/\sqrt{\beta_x}$ $[\sqrt{m}]$', 
            )
        plt.xticks(n_sl)
        plt.xlabel('Number of slices')
        plt.yscale('log')
        plt.ylim(0.0001, 0.3)
        plt.ylabel(r'Relative variation')
        plt.legend()
        plt.show()

    folder = '/mnt/c/muco/Results/figures/chap6/RCS1_model'
    rcs_name = 'RCS1_gf'
    save_fig = False
    if save_fig:
        po.comp_optics.savefig(folder+rcs_name+'optics_rcs_model_vs_num')
        po.comp_traj.savefig(folder+rcs_name+'traj_rcs_model_vs_num')
        po.comp_z.savefig(folder+rcs_name+'z_rcs_model_vs_num')
        po.comp_delta.savefig(folder+rcs_name+'delta_rcs_model_vs_num')
        po.comp_all_dip.savefig(folder+rcs_name+'optics_rcs_all_vs_dip')

    # plotter = EBPlotter(
    #     arc = arc, 
    #     rcs = rcs, 
    #     t_acc = t_acc, 
    #     n_turn_rcs = n_turn_rcs, 
    #     n_slice = n_slice)
    # plotter.plot_all_arc()
    # plotter.plot_all_rcs()

    nb_arc = np.array([8, 12, 16, 20, 24, 28])

    rcs_name = 'RCS1_gf_normal_qp'
    qx = 1.7
    qy = 1.7
    dqx = tw_noramp.dqx
    dqy = tw_noramp.dqy
    tunes = {'qx': qx, 'qy': qy, 'dqx': dqx, 'dqy': dqy}

    results_narc = {}

    file_input = {
        str(na): {
            'file_line': path_08_25 / f'line_{rcs_name}_{str(na)}_arc.json',
            'file_geo': path_08_25 / f'para_{rcs_name}__{str(na)}_arc.json'
        }
        for na in nb_arc
    }

    # ============================================================
    #                PARTIE 1 : STUDY OF THE ARC NUMBER
    # ============================================================

    results_narc = {}

    for na in file_input.keys():

        arc = xt.Line.from_json(file_input[na]['file_line'])
        tw0 = arc.twiss4d(matrix_stability_tol=0.1)
        rcs = RingGeometry(file_input[na]['file_geo'])

        # Chromatic correction
        if tunes is not None:
            arc_chroma = adjust_qx_arc(
                line=arc,
                qx=tunes['qx'], qy=tunes['qy'],
                dqx=tunes['dqx'], dqy=tunes['dqy']
            )
        else:
            arc_chroma = arc

        tw_chroma = arc_chroma.twiss(matrix_stability_tol=0.1)
        print("Chromaticities after correction:", tw_chroma.dqx, tw_chroma.dqy)

        EB = EB_error(
            line=arc_chroma,
            rcs=rcs,
            t_acc=t_acc,
            n_turn_rcs=n_turn_rcs
        )

        results_narc[na] = compute_relative_results(EB, rg)


    # ----- Plots: betax, bety, Dx --- #
    plot_relative_diff(results_narc, 'betx', 'betx_ref', r'$\Delta \beta_x / \beta_x$')
    plot_relative_diff(results_narc, 'bety', 'bety_ref', r'$\Delta \beta_y / \beta_y$')

    plt.figure()
    for na, dat in results_narc.items():
        rel = (dat['dx'] - dat['dx_ref']) / dat['betx_ref']
        plt.plot(dat['s'] * 1e-3, rel, label=f"{na} arcs")
    plt.xlabel('s [km]')
    plt.ylabel(r'$\Delta D_x / \sqrt{\beta_x}$ [√m]')
    plt.legend()
    plt.show()

    # --- Variations vs nb_arc --- #
    bx_max, by_max, dx_max = compute_max_variations(results_narc)

    nb_arc_array = np.array(nb_arc)
    fit_and_plot(func_inv, nb_arc_array, bx_max, by_max, dx_max,
                xlabel="Number of arcs", xticks=nb_arc_array)


    # ============================================================
    #      PARTIE 2 : STUDY OF t_acc
    # ============================================================

    results_tf = {}

    # We use the line with 8 arcs
    na = '8'
    # file_input = self.file_input[str(min(self.nb_arc))]['file_line']
    file_input = '/mnt/c/muco/code/lattices/lattice_08_25/line_RCS1_gf_normal_qp_8_arc_v2.json'
    arc = xt.Line.from_json(file_input)
    rcs = RingGeometry('/mnt/c/muco/code/lattices/lattice_08_25/para_RCS1_gf_normal_qp__8_arc.json')

    # Chromaticity correction
    if tunes is not None:
        arc_chroma = adjust_qx_arc(
            line=arc,
            qx=tunes['qx'], qy=tunes['qy'],
            dqx=tunes['dqx'], dqy=tunes['dqy']
        )
    else:
        arc_chroma = arc

    tw_chroma = arc_chroma.twiss(matrix_stability_tol=0.1)
    print("Chromaticities:", tw_chroma.dqx, tw_chroma.dqy)

    t_factor = np.array([1, 2, 4, 6, 8, 10])

    for tf in t_factor:
        EB = EB_error(
            line=arc_chroma,
            rcs=rcs,
            t_acc=t_acc * tf,
            n_turn_rcs=n_turn_rcs
        )
        results_tf[str(tf)] = compute_relative_results(EB, rg)

    # --- Plots betax, bety, Dx --- #
    plot_relative_diff(results_tf, 'betx', 'betx_ref', r'$\Delta \beta_x / \beta_x$', legend_suffix=' × t_acc')
    plot_relative_diff(results_tf, 'bety', 'bety_ref', r'$\Delta \beta_y / \beta_y$', legend_suffix=' × t_acc')

    plt.figure()
    for tf, dat in results_tf.items():
        rel = (dat['dx'] - dat['dx_ref']) / dat['betx_ref']
        plt.plot(dat['s'] * 1e-3, rel, label=f"{tf}×t_acc")
    plt.xlabel('s [km]')
    plt.ylabel(r'$\Delta D_x / \sqrt{\beta_x}$ [√m]')
    plt.legend()
    plt.show()

    # # --- Variations vs t_factor --- #
    # bx_max, by_max, dx_max = compute_max_variations(results_tf)

    # fit_and_plot(func_inv, t_factor, bx_max, by_max, dx_max,
    #             xlabel=r"$f \times t_{acc}$", xticks=t_factor)

    # # --- Linear fit as a function of 1/t_factor --- #

    # inv_tf = 1 / t_factor
    # fit_and_plot(func_lin, inv_tf, bx_max, by_max, dx_max,
    #             xlabel=r"$1/f$")

    # with open('/mnt/c/muco/code/traj_bprime_rcs1_8arcs.json', 'r') as f:
    #     traj = json.load(f)
    # traj_pt = [traj[key]['x'] for key in traj.keys()]
    # x = t_factor
    # x_fit = np.linspace(np.min(x), np.max(x), 20)
    # func = func_inv

    # bx_res, _ = curve_fit(func, x, bx_max)
    # by_res, _ = curve_fit(func, x, by_max)
    # dx_res, _ = curve_fit(func, x, dx_max)
    # x_res, _ = curve_fit(func, x, traj_pt)

    # bx_fit = func(x_fit, *bx_res)
    # by_fit = func(x_fit, *by_res)
    # dx_fit = func(x_fit, *dx_res)
    # traj_fit = func(x_fit, *x_res)

    # fig, ax1 = plt.subplots(figsize=(8,5))
    # ax1.scatter(x, bx_max, s=25, color='tab:blue', alpha=0.7)
    # ax1.plot(x_fit, bx_fit, color='tab:blue', lw=2, label=r'$\Delta \beta_x$')
    # ax1.scatter(x, by_max, s=25, color='tab:orange', alpha=0.7)
    # ax1.plot(x_fit, by_fit, color='tab:orange', lw=2, label=r'$\Delta \beta_y$')
    # ax1.scatter(x, dx_max*10, s=25, color='tab:green', alpha=0.7)
    # ax1.plot(x_fit, dx_fit*10, color='tab:green', lw=2, label=r'$\Delta D_x \times 10$')
    # ax1.set_xlabel(r"$t$ / $t_{acc}$")
    # ax1.set_ylabel("Relative variation")
    # ax1.grid(alpha=0.25)
    # ax2 = ax1.twinx()
    # color='tab:red'
    # ax2.scatter(x, np.array(traj_pt)*1e3, s=25, color=color, alpha=0.7)
    # ax2.plot(x_fit, func(x_fit, *x_res)*1e3, color=color, lw=2, label=r'$\Delta x$')
    # ax2.set_ylabel(r'$\Delta x$ [mm]', color=color)
    # ax2.tick_params(axis='y', colors=color)
    # ax2.spines['right'].set_color(color)
    # ax2.spines['top'].set_visible(False)
    # ax1.legend(loc='upper right', bbox_to_anchor=(0.82, 1))
    # ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))
    # plt.tight_layout()
    # plt.show()

    # # --- Normalisation to the value at t/t_acc = 1 (in %) --- #
    # def norm(data, fit):
    #     ref = data[0]                      # value at t/t_acc = 1
    #     return 100 * data / ref, 100 * fit / ref
    # series = {
    #     r'$\Delta \beta_x / \beta_x$':            (bx_max,           bx_fit,   'tab:blue'),
    #     r'$\Delta \beta_y / \beta_y$':            (by_max,           by_fit,   'tab:orange'),
    #     r'$\Delta D_x / \sqrt{\beta_x}$':         (dx_max,           dx_fit,   'tab:green'),
    #     r'$\Delta x$':                            (np.array(traj_pt), traj_fit, 'tab:red'),
    # }

    # fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # for label, (data, fit, color) in series.items():
    #     data_n, fit_n = norm(np.asarray(data), np.asarray(fit))
    #     ax.scatter(x, data_n, color=color, alpha=1, marker='+', s=75)
    #     ax.plot(x_fit, fit_n, color=color, lw=1.2, label=label) #, linestyle='--')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_xlabel(r"$t \ / \ t_{acc}$")
    # ax.set_ylabel(r"Max. variation relative to nominal $t_{acc}$ [%]")
    # ax.set_xticks(x)
    # ax.set_xticklabels([str(v) for v in x])
    # ax.set_yticks([10, 20, 50, 100])
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    # ax.yaxis.set_minor_formatter(plt.NullFormatter()) 
    # ax.grid(alpha=0.25, which='both')
    # ax.legend(loc='upper right')
    # plt.tight_layout()
    # plt.show()

    # # Comparison for varying n_slice
    # n_slice_list = [3, 5, 10]

    # raw_data = compute_raw_data(n_slice_list, arc, rcs, t_acc, n_turn_rcs, rcs_ref)
    # derived = compute_differences(raw_data)
    # plot_differences(derived, save_path="beta_dx_plot.png")

    # n = 3
    # n2 = 10
    # s = derived[n]['s']
    # dbetx_1 = derived[n]['dbetx_sliced']
    # dbety_1 = derived[n]['dbety_sliced']
    # dDx_1 = derived[n]['dDx_sliced']
    # dbetx_2 = derived[n2]['dbetx_sliced']
    # dbety_2 = derived[n2]['dbety_sliced']
    # dDx_2 = derived[n2]['dDx_sliced']
        
    # fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6.4, 6.4))

    # color1 = 'tab:blue'
    # color2 = 'tab:orange'
    # al = 0.8

    # axes[0].plot(s*1e-3, dbetx_1-dbetx_2, color=color2, alpha=al, label=f'{str(n2)}-{str(n)}s')
    # axes[1].plot(s*1e-3, dbety_1-dbety_2, color=color2, alpha=al)
    # axes[2].plot(s*1e-3, dDx_1-dDx_2, color=color2, alpha=al)

    # axes[0].set_ylabel(r'$\Delta \beta_x$ / $\beta_x$')
    # axes[1].set_ylabel(r'$\Delta \beta_y$ / $\beta_y$')
    # axes[2].set_ylabel(r'$\Delta D_x$ / $\sqrt{\beta_x}$' + ' [$\sqrt{m}$ ]')

    # axes[2].set_xlabel('s [km]')
    # axes[0].legend()
    # plt.tight_layout()
    # plt.show()