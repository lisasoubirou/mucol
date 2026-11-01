from class_geometry.ring_geometry.ring_geometry import RingGeometry, load_file
import xtrack as xt
import xpart as xp
from EB_error import EB_error
from pathlib import Path
path_lattices = Path(__file__).resolve().parent

# 8 arcs lattice
# path_conso = path_lattices / 'lattice_consolidated_ESPPU'
# input_rcs = {
#     'RCS1': {
#         'n_turn_rcs': 17,
#         't_acc': 0.34e-3,
#         'file_arc': path_conso / 'GF_RCS1' / 'line_RCS1_gf_normal_qp_8_arc.json',
#         'file_geo': path_conso / 'GF_RCS1' / 'para_RCS1_gf_normal_qp__8_arc.json'
#     },
#     'RCS2': {
#         'n_turn_rcs': 55,
#         't_acc': 1.1e-3,
#         'file_arc': path_conso / 'GF_RCS2' / 'line_RCS2_gf_normal_qp_8_arc.json',
#         'file_geo': path_conso / 'GF_RCS2' / 'para_RCS2_gf_normal_qp__8_arc.json'
#     },
#     'RCS3': {
#         'n_turn_rcs': 66,
#         't_acc': 2.37e-3,
#         'file_arc': path_conso / 'GF_RCS3' / 'line_RCS3_gf_normal_qp_8_arc.json',
#         'file_geo': path_conso / 'GF_RCS3' / 'para_RCS3_gf_normal_qp__8_arc.json'
#     },
#     'RCS4': {
#         'n_turn_rcs': 55,
#         't_acc': 6.37e-3,
#         'file_arc': path_conso / 'GF_RCS4' / 'line_RCS4_gf_normal_qp_8_arc.json',
#         'file_geo': path_conso / 'GF_RCS4' / 'para_RCS4_gf_normal_qp__8_arc.json'
#     }
# }

# 32 arcs lattice
path_freport = path_lattices / 'lattices_final_report'
input_rcs = {
    'RCS1': {
        'n_turn_rcs': 17,
        't_acc': 0.34e-3,
        'file_arc': path_freport / 'RCS1_gf' / 'line_RCS1_gf_normal_qp_32_arc.json',
        'file_geo': path_freport / 'RCS1_gf' / 'para_RCS1_gf_normal_qp__32_arc.json'
    },
    'RCS2': {
        'n_turn_rcs': 55,
        't_acc': 1.1e-3,
        'file_arc': path_freport / 'RCS2_gf' / 'line_RCS2_gf_normal_qp_32_arc.json',
        'file_geo': path_freport / 'RCS2_gf' / 'para_RCS2_gf_normal_qp__32_arc.json'
    },
    'RCS3': {
        'n_turn_rcs': 66,
        't_acc': 2.37e-3,
        'file_arc': path_freport / 'RCS3_gf' / 'line_RCS3_gf_normal_qp_32_arc.json',
        'file_geo': path_freport / 'RCS3_gf' / 'para_RCS3_gf_normal_qp__32_arc.json'
    },
    'RCS4': {
        'n_turn_rcs': 55,
        't_acc': 6.37e-3,
        'file_arc': path_freport / 'RCS4_gf' / 'line_RCS4_gf_normal_qp_32_arc.json',
        'file_geo': path_freport / 'RCS4_gf' / 'para_RCS4_gf_normal_qp__32_arc.json'
    }
}

rcs_name = 'RCS4'
file_line = input_rcs[rcs_name]['file_arc']
file_geo = input_rcs[rcs_name]['file_geo']
t_acc = input_rcs[rcs_name]['t_acc']
n_turn_rcs = input_rcs[rcs_name]['n_turn_rcs']

arc = xt.Line.from_json(file_line)
rcs = RingGeometry(file_geo)

EB = EB_error(
    line=arc, 
    rcs=rcs, 
    t_acc=t_acc, 
    n_turn_rcs=n_turn_rcs, 
    )

el_inj, name_inj = EB.build_arc_ref(
    t_ref = 0,
    i_arc = 0
)
el_ext, name_ext = EB.build_arc_ref(
    t_ref = 1,
    i_arc = 0
)
line_inj = xt.Line(
    elements=el_inj,
    element_names=name_inj
)
line_inj.particle_ref = xt.Particles(
        p0c=rcs.injection_energy,
        mass0=xp.MUON_MASS_EV,
        q0=1,
    )
line_ext = xt.Line(
    elements=el_ext,
    element_names=name_ext
)
line_ext.particle_ref = xt.Particles(
        p0c=rcs.extraction_energy,
        mass0=xp.MUON_MASS_EV,
        q0=1,
    )

line_inj.to_json(Path(file_line).parent / f'{rcs_name}_DS_inj.json')
line_ext.to_json(Path(file_line).parent /f'{rcs_name}_DS_ext.json')
