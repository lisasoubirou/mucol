import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import scipy.constants as sc
from class_geometry.ring_geometry.ring_geometry import RingGeometry
from EB_mismatch.EB_error import EB_error
from plot_optics import compare_optics, compare_optics2ref, doublecompare_optics2ref, compare_traj, mask_common_s
from tracking.TrackingRCS import TrackingRCS

plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize


# %%
def plot_emit_nrcs(results, labels=None):
    """
    Compare les évolutions des émittances et de la transmission
    pour plusieurs configurations de RCS.

    Parameters
    ----------
    results : list of dict
        Liste des dictionnaires de résultats.
    labels : list of str, optional
        Labels associés aux configurations.
    """

    n = len(results)

    if labels is None:
        labels = [f"RCS {i+1}" for i in range(n)]

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(6.4, 6.4),
        sharex=True
    )

    colors = plt.cm.tab10(range(n))

    for res, cfg_label, cl in zip(results, labels, colors):

        ex = res["emitx"]
        ey = res["emity"]
        es = res["emits"]
        tr = res["transmission"]
        s = res["s"]

        curves = [
            (axes[0], (ex - ex[0]) / ex[0] * 100),
            (axes[1], (ey - ey[0]) / ey[0] * 100),
            (axes[2], (es - es[0]) / es[0] * 100),
            (axes[3], tr * 100),
        ]

        for ax, y in curves:
            ax.plot(
                s * 1e-3,
                y,
                color=cl,
                label=cfg_label if ax == axes[0] else None,
            )

    axes[0].set_ylabel(r'$\Delta \epsilon_x / \epsilon_x$ [%]')
    axes[1].set_ylabel(r'$\Delta \epsilon_y / \epsilon_y$ [%]')
    axes[2].set_ylabel(r'$\Delta \epsilon_s / \epsilon_s$ [%]')
    axes[3].set_ylabel("Transmission [%]")
    axes[3].set_xlabel("s [km]")
    axes[3].ticklabel_format(axis='y', style='plain', useOffset=False)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.57, -0.015),
        ncol=min(n, 2),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    plt.show()

    return fig

if __name__ == '__main__':
    # file_geo = '/mnt/c/muco/code/lattice_08_25/para_RCS1_gf_normal_qp__28_arc.json'
    # file_arc = '/mnt/c/muco/code/lattice_08_25/line_RCS1_gf_normal_qp_28_arc_matched.json'

    file_geo = '/mnt/c/muco/code/lattices/lattice_08_25/para_RCS1_gf_normal_qp__8_arc_v2.json'
    file_arc = '/mnt/c/muco/code/lattices/lattice_08_25/line_RCS1_gf_normal_qp_8_arc_v2.json'

    file_geo = '/mnt/c/muco/code/lattices/lattices_final_report/RCS1_gf/para_RCS1_gf_normal_qp__32_arc.json'
    file_arc = '/mnt/c/muco/code/lattices/lattices_final_report/RCS1_gf/line_RCS1_gf_normal_qp_32_arc.json'

    rcs_name = 'RCS1_32arcs'
    # Acceleration parameters
    n_turn_rcs = 17
    t_acc = 0.34 * 1e-3
         
    # RCS parameters
    rcs = RingGeometry(file_geo)
    nb_arc_tot = n_turn_rcs*rcs.nb_arc
        
    # Importing arc
    line_arc = xt.Line.from_json(file_arc)
    tw_arc = line_arc.twiss(method='6d', matrix_stability_tol=0.7)
    tw_init = tw_arc.get_twiss_init(0)

    ppm = 1e-4 #100 * 1e-6
    B_inj = rcs.dipole_families['BNC']['B'](0)
    B_ext = rcs.dipole_families['BNC']['B'](1)
    dB = B_ext * ppm

    plot_ramp = False
    if plot_ramp:

        pt_ramp = 20
        B_ramp = np.linspace(B_inj, B_ext, pt_ramp)
        t_ramp = np.linspace(0, 1, pt_ramp)

        fac = 200
        ppm_prog = B_ramp * ppm * fac
        ppm_max = B_ext * ppm * fac

        print('----- Plot B ramp -----')

        fig_ramp = plt.figure(figsize=(6.4, 4.8))
        # Main B-ramp
        plt.plot(
            t_ramp, 
            B_ramp, 
            label='Nominal magnetic ramp', 
            color='tab:blue', 
            linewidth=1
            )
        # Colored ppm band
        plt.fill_between( # Progressive
            t_ramp,
            B_ramp - ppm_prog,
            B_ramp + ppm_prog,
            alpha = 0.5,
            label=f'Instantaneous tolerance band * {fac}',
            color='tab:blue'
        )
        plt.fill_between( # Maxed out
            t_ramp,
            B_ramp - ppm_max,
            B_ramp + ppm_max,
            alpha = 0.3,
            label=f'Peaked tolerance band * {fac}',
            color='tab:red'
        )
        plt.xlabel(r'$t \ / \ t_{acc}$')
        plt.ylabel('B [T]')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        # Progressive ppm
        B_inj_min = B_inj * (1 - ppm)
        B_inj_max = B_inj * (1 + ppm)
        B_ext_min = B_ext * (1 - ppm)
        B_ext_max = B_ext * (1 + ppm)

        ramp_min = np.linspace(B_inj_min, B_ext_min, pt_ramp)
        ramp_max = np.linspace(B_inj_max, B_ext_max, pt_ramp)
        ramp_diag_min = np.linspace(B_inj_min, B_ext_max, pt_ramp)
        ramp_diag_max = np.linspace(B_inj_max, B_ext_min, pt_ramp)

        fig_inst_band = plt.figure(figsize=(6.4, 4.8))
        # plt.title('Progressive ppm band')
        plt.plot(t_ramp, ramp_min-B_ramp, label='$I_{--}$', color='tab:green', linewidth=1)
        plt.plot(t_ramp, ramp_diag_min-B_ramp, label='$I_{-+}$', color='tab:blue', linewidth=1)
        plt.plot(t_ramp, ramp_max-B_ramp, label='$I_{++}$', color='tab:red', linewidth=1)
        plt.plot(t_ramp, ramp_diag_max-B_ramp, label='$I_{+-}$', color='tab:purple', linewidth=1)
        plt.xlabel(r'$t \ / \ t_{acc}$')
        plt.ylabel(r'B - $B_{ref}$ [T]')
        plt.legend(loc = 'center right')
        plt.tight_layout()
        plt.show()

        # Max ppm
        B_inj_min = B_inj - dB
        B_inj_max = B_inj + dB
        B_ext_min = B_ext - dB
        B_ext_max = B_ext + dB

        ramp_min = np.linspace(B_inj_min, B_ext_min, pt_ramp)
        ramp_max = np.linspace(B_inj_max, B_ext_max, pt_ramp)
        ramp_diag_min = np.linspace(B_inj_min, B_ext_max, pt_ramp)
        ramp_diag_max = np.linspace(B_inj_max, B_ext_min, pt_ramp)

        fig_max_band = plt.figure(figsize=(6.4, 4.8))
        # plt.title('Max ppm band')
        plt.plot(t_ramp, ramp_min-B_ramp, label='$I_{--}$', color='tab:green', linewidth=1)
        plt.plot(t_ramp, ramp_diag_min-B_ramp, label='$I_{-+}$', color='tab:blue', linewidth=1)
        plt.plot(t_ramp, ramp_max-B_ramp, label='$I_{++}$', color='tab:red', linewidth=1)
        plt.plot(t_ramp, ramp_diag_max-B_ramp, label='$I_{+-}$', color='tab:purple', linewidth=1)
        plt.xlabel(r'$t \ / \ t_{acc}$')
        plt.ylabel(r'B - $B_{ref}$ [T]')
        plt.legend(loc = 'center right')
        plt.tight_layout()
        plt.show()

    folder = '/mnt/c/muco/Results/figures/chap6/jitter/'
    save_fig_ramp = False
    if save_fig_ramp:
        fig_ramp.savefig(folder+f'ramp_scenarios_{rcs_name}_{int(ppm*1e6)}ppm')
        fig_inst_band.savefig(folder+f'ramp_inst_{rcs_name}_{int(ppm*1e6)}ppm')
        fig_max_band.savefig(folder+f'ramp_max_{rcs_name}_{int(ppm*1e6)}ppm')

    # sys.exit(0)
    sjit = dB / (B_ext - B_inj) * t_acc * sc.c
    tjit = sjit / sc.c / t_acc
    Bjit = B_inj + (B_ext - B_inj) * tjit

    plot_scenario = False
    if plot_scenario:

        print('----- Build lines without jitter -----')
        EB = EB_error(
            line = line_arc, 
            rcs = rcs,
            t_acc = t_acc, 
            n_turn_rcs = n_turn_rcs,
            n_slice = 10,
            )

        EB.build_rcs_ref()
        tw_ref = EB.tw_rcs_ref
        EB.build_rcs_RMatrix_all()
        tw = EB.tw_rcs_RMatrix_all
        EB.build_rcs_sliced_all()

        print('----- Build lines with jitter -----')
        EB_jitter = EB_error(
            line = line_arc, 
            rcs = rcs,
            t_acc = t_acc, 
            n_turn_rcs = n_turn_rcs,
            n_slice = 10,
            s_jitter = sjit,
            jitter_diagonal = True
            )

        EB_jitter.build_rcs_RMatrix_all()
        tw_jitter = EB_jitter.tw_rcs_RMatrix_all
        EB_jitter.build_rcs_sliced_all()
        tw_sliced = EB_jitter.tw_rcs_sliced_all
        
        print('----- Plot jitter vs no jitter -----')
        compare_optics2ref(
            tw_ref = tw_ref,
            tw1 = tw_jitter,
            tw2 = tw,
            label=['Jitter', 'No Jitter']
        )

        print('----- RMatrix method -----')
        doublecompare_optics2ref(
            tw_ref = tw_ref,
            tw1 = tw_jitter,
            tw2 = tw)

        print('----- RMatrix vs Sliced method -----')
        doublecompare_optics2ref(
            tw_ref = tw_ref,
            tw1 = tw_jitter,
            tw2 = tw_sliced)

        compare_traj(tw, tw_jitter)

        rg = nb_arc_tot

        #Plot k0 for every dip in the line
        line_jitter = EB_jitter.rcs_sliced_all
        tw_jitter = EB_jitter.tw_rcs_sliced_all
        dips_jitter = line_jitter.get_elements_of_type(xt.Bend)[1]
        k0_jitter = np.array([line_jitter[dip].k0 for dip in dips_jitter])
        h_jitter = np.array([line_jitter[dip].h for dip in dips_jitter])
        s_jitter = np.array([tw_jitter['s', dip] for dip in dips_jitter])

        line = EB.rcs_sliced_all
        tw = EB.tw_rcs_sliced_all
        dips = line.get_elements_of_type(xt.Bend)[1]
        k0 = np.array([line[dip].k0 for dip in dips])
        h = np.array([line[dip].h for dip in dips])
        s = np.array([tw['s', dip] for dip in dips])

        plt.figure(figsize=(6.4, 4.8))
        plt.plot(s * 1e-3, k0_jitter - k0 , label='Jitter - No jitter')
        plt.xlabel('s [km]')
        plt.ylabel(r' $k_0$ [1/m]')
        plt.legend()
        plt.tight_layout()
        plt.show()

    #Scan on Scenarios and ppm
    ppm_list = [10, 100, 200, 500]
    scenario_labels = {
            's_jitter_plus_diag_False': 'I_{--}',
            's_jitter_plus_diag_True': 'I_{-+}',
            's_jitter_minus_diag_False': 'I_{++}',
            's_jitter_minus_diag_True': 'I_{+-}',
        }

    metrics = ['D_beta_x', 'D_beta_y', 'D_dx', 'D_x']
    metric_titles = {
            'D_beta_x': r'Relative increase $\Delta \beta_x \ / \ \beta_x $ [%]',
            'D_beta_y': r'Relative increase $\Delta \beta_y \ / \ \beta_y $ [%]' ,
            'D_dx': r'Relative increase $\Delta D_x \ / \ \sqrt{\beta_x}$ [%]',
            'D_x' : r'Relative increase $\Delta x$ [%]',
        }
    metric_values = {
            'D_beta_x': r'$\Delta \beta_x \ / \ \beta_x $ [%]',
            'D_beta_y': r'$\Delta \beta_y \ / \ \beta_y $ [%]' ,
            'D_dx': r'$\Delta D_x \ / \ \sqrt{\beta_x} [\sqrt{m}]$',
            'D_x' : r'$\Delta x$ [m]',
        }
    value_titles = {
            'beta_x': r'Max $\Delta \beta_x \ / \ \beta_x $ [%]',
            'beta_y': r'Max $\Delta \beta_y \ / \ \beta_y $ [%]' ,
            'dx': r'Max $\Delta D_x \ / \ \sqrt{\beta_x} [\sqrt{m}]$',
            'x' : r'Max $\Delta x$ [mm]',
        }
    scale = [100, 100, 1, 1e3]
    ppm_list = [500]
    plot_scan = False
    if plot_scan : 

        # Containers for results
        results = {}

        for ppm in ppm_list:
            print(f"\n================ PPM = {ppm} =================")

            # Scale dB according to ppm (assuming dB corresponds to 1e6 ppm full scale)
            dB_ppm = B_ext * ppm * 1e-6

            # Compute jitter quantities
            sjit = dB_ppm / (B_ext - B_inj) * t_acc * sc.c
            tjit = sjit / sc.c / t_acc
            Bjit = B_inj + (B_ext - B_inj) * tjit

            print(f"sjit = {sjit:.3e} m, tjit = {tjit:.3e}, Bjit = {Bjit:.6e}")

            # --- Reference (no jitter) ---
            EB_ref = EB_error(
                line=line_arc,
                rcs=rcs,
                t_acc=t_acc,
                n_turn_rcs=n_turn_rcs,
            )

            EB_ref.build_rcs_ref()
            tw_ref = EB_ref.tw_rcs_ref

            EB_ref.build_rcs_RMatrix_all()
            tw_nominal = EB_ref.tw_rcs_RMatrix_all

            mask1, mask2  = mask_common_s(tw_ref, tw_nominal)
            disp_beat_x_nom = (tw_nominal.dx[mask2] - tw_ref.dx[mask1]) / np.sqrt(tw_ref.betx[mask1])
            beta_beat_x_nom = (tw_nominal.betx[mask2]-tw_ref.betx[mask1]) / tw_ref.betx[mask1]
            beta_beat_y_nom = ( tw_nominal.bety[mask2]-tw_ref.bety[mask1]) / tw_ref.bety[mask1]
            x_nom = tw_nominal.x[mask2]
            s_nom = tw_nominal.s[mask2]

            results['nom'] = {
                'line' : EB_ref.rcs_RMatrix_all,
                'tw' : tw_nominal,
                'D_beta_x': beta_beat_x_nom,
                'D_beta_y': beta_beat_y_nom,
                'D_dx': disp_beat_x_nom,
                'D_x' : x_nom,
                's' : s_nom
                }

            # Store scenario results
            results[ppm] = {}

            # --- Scan jitter scenarios ---
            for s_sign in [+1, -1]:
                for jitter_diagonal in [True, False]:

                    scenario_key = f"s_jitter_{'plus' if s_sign>0 else 'minus'}_diag_{jitter_diagonal}"
                    print(f"\n--- Scenario: {scenario_key} ---")

                    EB_jitter = EB_error(
                        line=line_arc,
                        rcs=rcs,
                        t_acc=t_acc,
                        n_turn_rcs=n_turn_rcs,
                        s_jitter=s_sign * sjit,
                        jitter_diagonal=jitter_diagonal,
                    )

                    EB_jitter.build_rcs_RMatrix_all()
                    tw_jitter = EB_jitter.tw_rcs_RMatrix_all
                    line_jitter = EB_jitter.rcs_RMatrix_all

                    # --- Compute differences and plot comparison ---
                    print(f'----- Plot: PPM={ppm} | {scenario_key}')
                    mask_ref, mask_jit  = mask_common_s(tw_ref, tw_jitter)

                    disp_beat_x = (tw_jitter.dx[mask_jit] - tw_ref.dx[mask_ref]) / np.sqrt(tw_ref.betx[mask_ref])
                    beta_beat_x = (tw_jitter.betx[mask_jit]-tw_ref.betx[mask_ref]) / tw_ref.betx[mask_ref]
                    beta_beat_y = ( tw_jitter.bety[mask_jit]-tw_ref.bety[mask_ref]) / tw_ref.bety[mask_ref]
                    x = tw_jitter.x[mask_jit]

                    D_dbx = beta_beat_x - beta_beat_x_nom
                    D_dby = beta_beat_y - beta_beat_y_nom
                    D_ddx = disp_beat_x - disp_beat_x_nom
                    D_traj_dx = x - x_nom
                    s = tw_jitter.s[mask_jit]

                    # # --- Extract maxima ---
                    D_dbx_max = np.max(np.abs(D_dbx))
                    D_dby_max = np.max(np.abs(D_dby))
                    D_ddx_max = np.max(np.abs(D_ddx))
                    D_traj_dx_max = np.max(np.abs(D_traj_dx))

                    # beta_beat_x = np.max(np.abs(dbx)) - beta_beat_x_nom
                    # beta_beat_y = np.abs(np.max(np.abs(dby)) - beta_beat_y_nom
                    # disp_beat_x = np.abs(np.max(np.abs(ddx)) - disp_beat_x_nom
                    # traj_x_beat = np.abs(np.max(np.abs(x_traj)) - traj_x_beat_nom

                    fig = compare_optics(
                        tw1=tw_ref,
                        tw2=tw_jitter,
                        # tw2=tw_nominal,
                    )

                    results[ppm][scenario_key] = {
                        'line' : line_jitter,
                        'tw' : tw_jitter,
                        'D_beta_x': D_dbx_max,
                        'D_beta_y': D_dby_max,
                        'D_dx': D_ddx_max,
                        'D_x': D_traj_dx_max,
                        'beta_x' : beta_beat_x,
                        'beta_y' : beta_beat_y,
                        'dx' : disp_beat_x,
                        'x' : x,
                        's' : s
                    }

                    print(f"Max beta-beat x = {np.max(beta_beat_x - beta_beat_x_nom):.3e}")
                    print(f"Max beta-beat y = {np.max(beta_beat_y - beta_beat_y_nom):.3e}")
                    print(f"Max dispersion x = {np.max(disp_beat_x - disp_beat_x_nom):.3e}")
                    print(f"Max dispersion dx = {np.max(x - x_nom):4e}")

        # print("\n===== Summary =====")
        # for ppm, scen in results.items():
        #     print(f"\nPPM = {ppm}")
        #     for k, v in scen.items():
        #         print(scenario_labels[str(k)])
        #         for key in v.keys():
        #             print(key, v[key])

        print('Reference fast ramp')
        print('D_x', np.max(np.abs(disp_beat_x_nom)))
        print('$beta_x', np.max(np.abs(beta_beat_x_nom)))
        print('$beta_y$', np.max(np.abs(beta_beat_y_nom)))
        print('$x$', np.max(np.abs(x_nom)))

        save_metric = False
        scale_metric = [100, 100, 1, 1e3]
        for metric, fac in zip(metrics, scale_metric):
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            ax2 = ax.twinx()

            y_nom = np.max(np.abs(results['nom'][metric]))

            for scen_key, sc_label in scenario_labels.items():
                y = np.array([results[ppm][scen_key][metric] for ppm in ppm_list])

                ax.plot(
                    ppm_list,
                    y * fac,
                    marker='o',
                    label=rf'${sc_label}$'
                )

            ax.set_xlabel('ppm')
            ax.set_ylabel(metric_values[metric])
            ax.legend()
            ax.grid(True)

            # Secondary axis: percentage of nominal
            ymin, ymax = ax.get_ylim()
            ax2.set_ylim(ymin / fac / y_nom * 100,
                        ymax / fac / y_nom * 100)
            ax2.set_ylabel(metric_titles[metric])

            plt.tight_layout()
            plt.show()
            if save_metric:
                fig.savefig(folder+ f"{rcs_name}_{rcs.nb_arc}arcs_{metric}", bbox_inches="tight")
            
        print('----- Worst case scenarios for each metric -----')
        for metric in metrics:
            print(metric)
            for ppm in ppm_list:
                worst_scen = max(
                    scenario_labels,
                    key=lambda s: results[ppm][s][metric]
                )
                print(f"{ppm} ppm : {scenario_labels[worst_scen]}")

    ramp = True
    n_part = 5000
    n_turn_rcs = 17
    t_acc = 0.34 * 1e-3
    nemitt_x = 25e-6
    nemitt_y = 25e-6
    sigma_z = 30.2e-12
    ppm_list = [10,100,500]

    ramp_scenario_track = True
    if ramp_scenario_track:

        res = {}
        s_sign = +1
        for ppm in ppm_list:
            res[ppm] = {}
            for jitter_diag in [False, True]:

                dB_ppm = B_ext * ppm * 1e-6

                # Compute jitter quantities
                sjit = dB_ppm / (B_ext - B_inj) * t_acc * sc.c
                tjit = sjit / sc.c / t_acc
                Bjit = B_inj + (B_ext - B_inj) * tjit

                scenario_key = f"s_jitter_{'plus' if s_sign>0 else 'minus'}_diag_{jitter_diag}"
                print(f"\n--- Scenario: {scenario_labels[scenario_key]} ---")
                
                track = TrackingRCS(
                    line=line_arc,
                    rcs=rcs,
                    t_acc=t_acc,
                    n_turn_rcs=n_turn_rcs,
                    n_part = n_part,
                    nemitt_x = nemitt_x,
                    nemitt_y = nemitt_y,
                    sigma_z = sigma_z,
                    ramp = ramp,
                    s_jitter = s_sign * sjit,
                    jitter_diagonal=jitter_diag,
                    adapt_part = True,
                    adapt_percentile = 90
                )

                res[ppm][scenario_labels[scenario_key]] = track

        track_ref = TrackingRCS(
            line=line_arc,
            rcs=rcs,
            t_acc=t_acc,
            n_turn_rcs=n_turn_rcs,
            n_part = n_part,
            nemitt_x = nemitt_x,
            nemitt_y = nemitt_y,
            sigma_z = sigma_z,
            ramp = ramp,
            adapt_part = False,
            adapt_percentile = 90
        )

        ramp_scen = ['I_{--}', 'I_{-+}']
        ramp_sc = ramp_scen[0]
        fig_I1 = plot_emit_nrcs( 
            [track_ref.track_res] + [res[ppm][ramp_sc].track_res for ppm in ppm_list],
            ['Nom. ramp']+[f'{ppm} ppm' for ppm in ppm_list])
        
        ramp_sc = ramp_scen[1]
        fig_I2= plot_emit_nrcs( 
            [track_ref.track_res] + [res[ppm][ramp_sc].track_res for ppm in ppm_list],
            ['Nom. ramp']+[f'{ppm} ppm' for ppm in ppm_list])
        
        final_emit = {}
        for ramp_sc in ramp_scen:
            final_emit[ramp_sc] = {}
            for ppm in ppm_list:
                dt = res[ppm][ramp_sc].track_res
                final_emit[ramp_sc][ppm] = {}
                final_emit[ramp_sc][ppm]['ex'] = (dt['emitx'][-1]-dt['emitx'][0]) / dt['emitx'][0]
                final_emit[ramp_sc][ppm]['ey'] =(dt['emity'][-1]-dt['emity'][0]) / dt['emity'][0]
                final_emit[ramp_sc][ppm]['es'] = (dt['emits'][-1]-dt['emits'][0]) / dt['emits'][0]
                final_emit[ramp_sc][ppm]['tr'] = dt['transmission'][-1]
        tr_nom = track_ref.track_res
        final_emit['ramp_nominal'] = {
            'ex' : (tr_nom['emitx'][-1]-tr_nom['emitx'][0])/tr_nom['emitx'][0],
            'ey' : (tr_nom['emity'][-1]-tr_nom['emity'][0])/tr_nom['emity'][0],
            'es' : (tr_nom['emits'][-1]-tr_nom['emits'][0])/tr_nom['emits'][0],
            'tr' : tr_nom['transmission'][-1],
        }
        
        save = False
        if save:
            fig_I1.savefig(folder+f'{rcs_name}_{ramp_scen[0]}_track', bbox_inches="tight")
            fig_I2.savefig(folder+f'{rcs_name}_{ramp_scen[1]}_track', bbox_inches="tight")
