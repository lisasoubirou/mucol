
import xtrack as xt
from class_geometry.ring_geometry.ring_geometry import RingGeometry
from tracking.TrackingRCS import TrackingRCS
from tracking.tracking_scripts.RCS1_matching.plot_match import save_match, load_match, plot_match
from pathlib import Path

path = Path(__file__).resolve().parent.parent

input_rcs = {
    '8arc_noramp_nomEl' : {
        'file_input' : path / 'lattices/lattice_08_25/line_RCS1_gf_normal_qp_8_arc_v2.json',
        'file_geo' : path / '/lattices/lattice_08_25/para_RCS1_gf_normal_qp__8_arc.json',
        'sz' : 30.2e-12,
        'ramp' : False,
        'label' : r'8a, No ramp, $\epsilon_s$',
    },
    '32arc_noramp_nomEl' : {
        'file_input' : path / 'lattices/lattices_final_report/RCS1_gf/line_RCS1_gf_normal_qp_32_arc.json',
        'file_geo' : path / 'lattices/lattices_final_report/RCS1_gf/para_RCS1_gf_normal_qp__32_arc.json',
        'sz' : 30.2e-12,
        'ramp' : False,
        'label' : r'32a, No ramp, $\epsilon_s$',
    },
}

keys = ['8arc_noramp_nomEl', '32arc_noramp_nomEl']
adapt = [70, 95]
n_sigma = 5
method = 'var_ref'
RF_tuning = False
plt_twiss = False
plt_sepa = True
ramp = False

n_part = 5000
n_turn_rcs = 17
t_acc = 0.34 * 1e-3
nemitt_x = 25e-6
nemitt_y = 25e-6

res = {}
for key_rcs, adapt_per in zip(keys, adapt):
    file_input = input_rcs[key_rcs]['file_input']
    file_geo = input_rcs[key_rcs]['file_geo']

    line_arc = xt.Line.from_json(file_input)
    rcs = RingGeometry(file_geo)

    ramp = input_rcs[key_rcs]['ramp']
    sigma_z = input_rcs[key_rcs]['sz']

    tr_adapt = TrackingRCS(
        line = line_arc,
        rcs =rcs,
        n_turn_rcs = n_turn_rcs,
        n_part=n_part,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        t_acc = t_acc,
        ramp = ramp,
        n_sigma = n_sigma,
        method = method,
        RF_tuning = RF_tuning,
        adapt_part = True,
        adapt_percentile = adapt_per,
        plt_twiss = plt_twiss,
        plt_sepa = plt_sepa,
    )

    tr_no_adapt = TrackingRCS(
        line = line_arc,
        rcs =rcs,
        n_turn_rcs = n_turn_rcs,
        n_part=n_part,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        t_acc = t_acc,
        ramp = ramp,
        n_sigma = n_sigma,
        method = method,
        RF_tuning = RF_tuning,
        adapt_part = False,
        adapt_percentile = adapt_per,
        plt_twiss = plt_twiss,
        plt_sepa = plt_sepa,
    )

    res[key_rcs] = {}
    res[key_rcs]['adapt'] = tr_adapt
    res[key_rcs]['no_adapt'] = tr_no_adapt

folder = '/mnt/c/muco/code/tracking_scripts/RCS1_matching/'
save_match(res, folder +'tracking_res_matching.json')
fig = plot_match(load_match(folder+ 'tracking_res_matching.json'))
fig.savefig(folder+'tracking_match_8_32.png')

