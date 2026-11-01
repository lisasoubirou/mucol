
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import xtrack as xt
import numpy as np
import re
from class_geometry.ring_geometry.ring_geometry import RingGeometry, load_file
from pathlib import Path

SIZE = 13
plt.rc('axes', labelsize=SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize

def extract_parameters(file, rcs_type="Hybrid QP"):
    """Return the parameters of the selected RCS type as a flat dict.

    rcs_type: "Hybrid QP" or "Normal QP" (also accepts "hybrid" / "normal").
    """
    aliases = {"hybrid": "Hybrid QP", "normal": "Normal QP", "cf": "Combined Function Magnets"}
    rcs_type = aliases.get(rcs_type.lower(), rcs_type)
    if rcs_type not in ("Hybrid QP", "Normal QP", "Combined Function Magnets"):
        raise ValueError(f"Unknown rcs_type: {rcs_type!r}")

    with open(file, "r", encoding="utf-8") as f:
        tex_content = f.read()

    patterns = {
        "normalized_transverse_emittance": r"Normalized transverse emittance\s*\[mm\.mrad\]\s*&\s*([\d.]+)\s*&\s*([\d.]+)",
        "beam_size_sigma": r"Beam size\s*\[\$\\sigma\$\]\s*&\s*([\d.]+)\s*&\s*([\d.]+)",
        "relative_energy_dispersion": r"Relative energy dispersion\s*&\s*([\d.]+)\s*&\s*([\d.]+)",
    }

    col = 1 if rcs_type == "Hybrid QP" else 2   # regex group index

    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, tex_content)
        if match is None:
            raise ValueError(f"Parameter '{key}' not found in {file}")
        results[key] = float(match.group(col))

    print(f"{rcs_type} parameters:")
    print(f"  Normalized transverse emittance: {results['normalized_transverse_emittance']} mm·mrad")
    print(f"  Beam size (σ): {results['beam_size_sigma']}")
    print(f"  Relative energy dispersion: {results['relative_energy_dispersion']}")

    return results

def beam_optics(tw, sigma_delta, n_sigma, tr_emit_n, inj_gamma):
    beta_x = tw.betx
    disp_x = tw.dx
    beam_size_x = sigma_delta*(disp_x) + n_sigma*np.sqrt(tr_emit_n / inj_gamma*beta_x)
    aperture_x = 2*beam_size_x

    beta_y = tw.bety
    disp_y = tw.dy
    beam_size_y = sigma_delta*(disp_y) + n_sigma*np.sqrt(tr_emit_n / inj_gamma*beta_y)
    aperture_y = 2*beam_size_y

    return aperture_x, aperture_y

def make_thick(line):
    thick_el = []
    names = []
    unique_parents = set()
    for index, element in line.to_pandas().iterrows():
        try:
            el = line.get(element['name'])
        except:
            pass
        
        if element['element_type'] in ['ThinSliceBendEntry', 'DriftSliceQuadrupole']:
            name = element['parent_name']
            if name not in unique_parents:
                unique_parents.add(name)
                el_data = line[element['parent_name']]
                names.append(element['parent_name'])
                thick_el.append(el_data)

        if element['element_type'] in ['Drift', 'Bend', 'Quadrupole'] :
            thick_el.append(el)
            names.append(element['name'])
    
    return thick_el, names

def format_k1(value, _):
    return f"{value:.1g}"

def plotLatticeSeries(ax, element, height, v_offset=0, color='r'):
    """Function to plot a lattice element as a patch

    Args:
        ax (_type_): _description_
        element (_type_): _description_
        height (_type_): _description_
        v_offset (int, optional): _description_. Defaults to 0.
        color (str, optional): _description_. Defaults to 'r'.
    """
    if not hasattr(element, 'length') or element.length == 0:
        return
    rect = Rectangle(
        (element.s, v_offset - height / 2),  # Bottom-left corner
        element.length,
        height,
        color=color,
        alpha=0.6
    )
    ax.add_patch(rect)

def plot_cell_k_twiss(line, myTwiss,
                figsize = (6,4),
                 color_k1 = 'red', 
                 color_k0 = 'green',
                 color_betx = 'k',
                 color_bety = 'r',
                 color_dx = 'b',
                 limit_k0 = 0.15,
                 limit_k1 = 0.005):

    fig = plt.figure(figsize=figsize)

    # Subplot n°1 : k0 and k1.
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)
    ax1.plot(myTwiss['s'], 0 * myTwiss['s'], 'k')  # reference line at y=0

    ax1.set_ylabel('$k_1$ [m$^{-2}$]', color=color_k1)  # k1 axis (on the left)
    ax1.tick_params(axis='y', labelcolor=color_k1)
    ax1.yaxis.set_major_formatter(FuncFormatter(format_k1))

    ax2 = ax1.twinx()   # k0 axis (on the right)
    ax2.set_ylabel('$k_0$ [m$^{-1}$]', color=color_k0)
    ax2.tick_params(axis='y', labelcolor=color_k0)
    ax2.yaxis.set_major_formatter(FuncFormatter(format_k1))

    s_pos = line.get_s_position()
    for index, element in line.to_pandas()[:-1].iterrows():
        el = line.get(element['name'])
        el.s = s_pos[index]

        if element['element_type'] == 'Bend':
            plotLatticeSeries(ax1, el, height=el.k1, v_offset=el.k1 / 2, color=color_k1)
            plotLatticeSeries(ax2, el, height=el.k0, v_offset=el.k0 / 2, color=color_k0)

        elif element['element_type'] == 'Quadrupole':
            plotLatticeSeries(ax1, el, height=el.k1, v_offset=el.k1 / 2, color=color_k1)
            # plotLatticeSeries(plt.gca(), el, height=el.knl[2], v_offset=el.knl[2] / 2, color='b')
    
    limit_k0 = max(abs(el.k0) for el in line.elements if hasattr(el, 'k0'))*2
    limit_k1 = max(abs(el.k1) for el in line.elements if hasattr(el, 'k1'))*1.1
    ax1.set_ylim(-limit_k1, limit_k1)
    ax2.set_ylim(-limit_k0, limit_k0)

    plt.subplot2grid((3, 3), (1, 0), colspan=3, rowspan=2, sharex=ax1)

    # Beta functions (on the left)
    plt.plot(myTwiss['s'], myTwiss['betx'], color=color_betx, label=r'$\beta_x$')
    plt.plot(myTwiss['s'], myTwiss['bety'], color=color_bety, label=r'$\beta_y$')
    plt.grid()
    plt.ylabel(r'$\beta$ [m]')
    plt.xlabel('s [m]')
    # plt.grid()
    plt.legend(loc='lower left')
    plt.ylim(0)

    # Dispersion (on the right)
    ax3 = plt.gca().twinx()
    plt.plot(myTwiss['s'], myTwiss['dx'], color=color_dx, label='$D_x$', lw=2)
    ax3.set_ylabel('$D_x$ [m]', color=color_dx)
    ax3.tick_params(axis='y', labelcolor=color_dx)
    plt.ylim(0)

    plt.tight_layout()
    plt.show()

    return fig

def plot_k_twiss(line, myTwiss,
                figsize = (6,4),
                 color_k1 = 'red', 
                 color_k0 = 'green',
                 color_betx = 'k',
                 color_bety = 'r',
                 color_dx = 'b',
                 limit_k0 = 0.15,
                 limit_k1 = 0.005):

    fig = plt.figure(figsize=figsize)

    # Subplot n°1 : k0 and k1.
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)
    ax1.plot(myTwiss['s'], 0 * myTwiss['s'], 'k')  # reference line at y=0

    ax1.set_ylabel('$k_1$ [m$^{-2}$]', color=color_k1)  # k1 axis (on the left)
    ax1.tick_params(axis='y', labelcolor=color_k1)
    ax1.yaxis.set_major_formatter(FuncFormatter(format_k1))

    ax2 = ax1.twinx()   # k0 axis (on the right)
    ax2.set_ylabel('$k_0$ [m$^{-1}$]', color=color_k0)
    ax2.tick_params(axis='y', labelcolor=color_k0)
    ax2.yaxis.set_major_formatter(FuncFormatter(format_k1))

    s_pos = line.get_s_position()
    for index, element in line.to_pandas()[:-1].iterrows():
        el = line.get(element['name'])
        el.s = s_pos[index]

        if element['element_type'] == 'Bend':
            plotLatticeSeries(ax1, el, height=el.k1, v_offset=el.k1 / 2, color=color_k1)
            plotLatticeSeries(ax2, el, height=el.k0, v_offset=el.k0 / 2, color=color_k0)

        elif element['element_type'] == 'Quadrupole':
            plotLatticeSeries(ax1, el, height=el.k1, v_offset=el.k1 / 2, color=color_k1)
            # plotLatticeSeries(plt.gca(), el, height=el.knl[2], v_offset=el.knl[2] / 2, color='b')
    
    limit_k0 = max(abs(el.k0) for el in line.elements if hasattr(el, 'k0'))*2
    limit_k1 = max(abs(el.k1) for el in line.elements if hasattr(el, 'k1'))*1.1
    ax1.set_ylim(-limit_k1, limit_k1)
    ax2.set_ylim(-limit_k0, limit_k0)

    plt.subplot2grid((3, 3), (1, 0), colspan=3, rowspan=2, sharex=ax1)

    # Beta functions (on the left)
    plt.plot(myTwiss['s'], myTwiss['betx'], color=color_betx, label=r'$\beta_x$', lw=2)
    plt.plot(myTwiss['s'], myTwiss['bety'], color=color_bety, label=r'$\beta_y$', lw=2)
    plt.ylabel(r'$\beta$ [m]')
    plt.xlabel('s [m]')
    # plt.grid()
    plt.legend(loc='upper left')
    plt.ylim(0)

    # Dispersion (on the right)
    ax3 = plt.gca().twinx()
    plt.plot(myTwiss['s'], myTwiss['dx'], color=color_dx, label='$D_x$', lw=2.5)

    ax3.set_ylabel('$D_x$ [m]', color=color_dx)
    ax3.tick_params(axis='y', labelcolor=color_dx)
    plt.ylim(0)

    plt.tight_layout()
    plt.show()

    return fig

def plot_enveloppe(myTwiss, beamX, beamY, figsize=(10,6), xylim=None, yylim=None):
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # --- Horizontal ---
    x_center = myTwiss.x * 1e3           # mm
    x_halfsize = beamX / 2 * 1e3         # mm
    x_upper = x_center + x_halfsize
    x_lower = x_center - x_halfsize

    axes[0].plot(myTwiss.s, x_center, color='k') #, label="x")
    axes[0].fill_between(myTwiss.s, x_lower, x_upper, color='k', alpha=0.3) #, label="Hor. beam Size")

    if not xylim:
        ymin_x = np.min(x_lower)
        ymax_x = np.max(x_upper)
        padding_x = 0.05 * (ymax_x - ymin_x)  # 5% padding
        axes[0].set_ylim(ymin_x - padding_x, ymax_x + padding_x)
    else:
        axes[0].set_ylim(-xylim, xylim)

    axes[0].set_ylabel("x [mm]")
    axes[0].grid()

    # --- Vertical ---
    y_center = myTwiss.y * 1e3
    y_halfsize = beamY / 2 * 1e3
    y_upper = y_center + y_halfsize
    y_lower = y_center - y_halfsize

    axes[1].plot(myTwiss.s, y_center, color='red') #, label="y")
    axes[1].fill_between(myTwiss.s, y_lower, y_upper, color='red', alpha=0.3) #, label="Ver. beam Size")

    if not yylim:
        ymin_y = np.min(y_lower)
        ymax_y = np.max(y_upper)
        padding_y = 0.05 * (ymax_y - ymin_y)
        axes[1].set_ylim(ymin_y - padding_y, ymax_y + padding_y)
    else:
        axes[1].set_ylim(-yylim, yylim)

    axes[1].set_xlabel("s [m]")
    axes[1].set_ylabel("y [mm]")
    # axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()
    
    return fig



if __name__ == '__main__':

    path = Path(__file__).resolve().parent.parent

    # --- For a cell ---
    rcs_name = 'RCS1_gf'

    # type_rcs = 'cf'
    file = path / f'lattices/latex_table_CF/{rcs_name}/cellCF_{rcs_name}_dipbent.json'
    table = path / f'lattices/latex_table_CF/{rcs_name}/tab_CF_{rcs_name}.tex'
    geo = path / f'lattices/latex_table_CF/{rcs_name}/para_CF_{rcs_name}.txt'

    type_rcs = 'normal'
    file = path / f'lattices/latex_table_hyb_qd_v2/{rcs_name}/{type_rcs}_{rcs_name}.json'
    table = path / f'lattices/latex_table_hyb_qd_v2/{rcs_name}/tab_hyb_normal_{rcs_name}.tex'
    geo = path / f'lattices/latex_table_hyb_qd_v2/{rcs_name}/para_{type_rcs}_{rcs_name}.txt'
    
    rcs = RingGeometry(geo)
    line = xt.Line.from_json(file)
    line_sliced = xt.Line.from_json(file)
    line_sliced.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(10))])
    myTwiss = line_sliced.twiss4d(matrix_stability_tol=0.1)
    thick_el, thick_names = make_thick(line)
    line = xt.Line(elements=thick_el, element_names=thick_names)
    line.particle_ref = line.particle_ref

    input_rcs_type = {"normal": "Normal QP", "hybrid": "Hybrid QP", "cf": "Combined Function Magnets"}
    data_input=extract_parameters(table, rcs_type=type_rcs)
    sigma_delta = data_input['relative_energy_dispersion']
    n_sigma = data_input['beam_size_sigma']
    tr_emit_n = data_input['normalized_transverse_emittance'] * 1e-6

    figsize = (6.4,5)

    fig = plot_cell_k_twiss(line, myTwiss, figsize=figsize)
    # Assuming beamX and beamY contain beam sizes at each s position
    tr_emit_n = 25 * 1e-6
    beamX, beamY = beam_optics(
        tw = myTwiss, 
        sigma_delta = sigma_delta, 
        n_sigma = n_sigma, 
        tr_emit_n = tr_emit_n, 
        inj_gamma = rcs.inj_gamma,
        )
    
    fig_env = plot_enveloppe(myTwiss, beamX, beamY, figsize=(6.4, 5), xylim=50, yylim=6)

    # --- For an arc ---
    # Geometry
    rcs_name = 'RCS1_gf'
    geo = path / 'lattices/lattice_08_25/para_RCS1_gf_normal_qp__8_arc.json'
    rcs = RingGeometry(geo)

    input = load_file(path / 'optics/FODO_cells_energy_spread.json')
    n_sigma = 3
    sigma_delta = input[rcs_name]["sig_delta_N_3"]
    # sigma_delta = input[rcs_name]["sig_delta_H_3"]

    #Line
    file = path / 'lattices/lattice_08_25/line_RCS1_gf_normal_qp_8_arc.json'
    file = path / 'lattices/lattice_08_25/line_RCS1_gf_normal_qp_8_arc_v2.json'
    line = xt.Line.from_json(file)
    line_sliced = xt.Line.from_json(file)
    line_sliced.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(10))])
    myTwiss = line_sliced.twiss(matrix_stability_tol=0.1)
    figsize = (10,6)

    fig = plot_k_twiss(line, myTwiss, figsize=figsize)
    # sys.exit(0)
    # Assuming beamX and beamY contain beam sizes at each s position
    tr_emit_n = 25 * 1e-6
    beamX, beamY = beam_optics(
        tw = myTwiss, 
        sigma_delta = sigma_delta, 
        n_sigma = n_sigma, 
        tr_emit_n = tr_emit_n, 
        inj_gamma = rcs.inj_gamma)
    
    fig_env = plot_enveloppe(myTwiss, beamX, beamY, figsize=(10,6))

    # save_fig = True
    # if save_fig:
    #     fig.savefig(folder + f'{rcs_name}_twiss_k')
    #     fig_env.savefig(folder + f'{rcs_name}_envelope')