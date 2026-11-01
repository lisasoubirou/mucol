"""
Estimates the momentum acceptance of a lattice by tracking particles
with different initial momentum deviations and identifying the
momentum range for which particles remain stable over the tracking period.
"""

from EB_mismatch.EB_error import EB_error
import xtrack as xt
from class_geometry.ring_geometry.ring_geometry import RingGeometry
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def scan_momentum_acceptance(
    line_rcs,
    delta_range=(-0.20, 0.20),
    n_delta=401,
):
    delta0 = np.linspace(delta_range[0], delta_range[1], n_delta)

    particles = xt.Particles(
        p0c=line_rcs.particle_ref.p0c,
        mass0=line_rcs.particle_ref.mass0,
        q0=line_rcs.particle_ref.q0,

        x=np.zeros(n_delta),
        px=np.zeros(n_delta),
        y=np.zeros(n_delta),
        py=np.zeros(n_delta),

        delta=delta0,

        # Unique ID corresponding to the original delta array
        particle_id=np.arange(n_delta),
    )

    # Track the complete RCS once
    line_rcs.track(
        particles,
        num_turns=1,
    )

    # Recover the original index from particle_id.
    # Do NOT use the current particle ordering.
    particle_id = np.asarray(particles.particle_id)
    state = np.asarray(particles.state)

    survived = np.zeros(n_delta, dtype=bool)

    valid = particle_id >= 0
    survived[particle_id[valid]] = state[valid] > 0

    lost = ~survived

    return delta0, survived, lost

path_lattices = Path(__file__).resolve().parent.parent

path_08_25 = path_lattices / 'lattices/lattice_08_25'
path_esppu = path_lattices / 'lattices/lattice_consolidated_ESPPU'
path_report = path_lattices / 'lattices/lattices_final_report'

rcs_files = {
    'RCS1': {
        '8': {
            'file_arc': path_08_25 / 'line_RCS1_gf_normal_qp_8_arc_v2.json',
            'file_geo': path_08_25 / 'para_RCS1_gf_normal_qp__8_arc.json',
            'sigma_z': 30.2e-12,
        },
        '32': {
            'file_arc': path_report / 'RCS1_gf' / 'line_RCS1_gf_normal_qp_32_arc.json',
            'file_geo': path_report / 'RCS1_gf' / 'para_RCS1_gf_normal_qp__32_arc.json',
            'sigma_z': 30.2e-12,
        }
    },
    'RCS2': {
        '8': {
            'file_arc': path_esppu / 'GF_RCS2' / 'line_RCS2_gf_normal_qp_8_arc_v2.json',
            'file_geo': path_esppu / 'GF_RCS2' / 'para_RCS2_gf_normal_qp__8_arc.json',
            'sigma_z': 23.3e-12,
        },
        '32': {
            'file_arc': path_report / 'RCS2_gf' / 'line_RCS2_gf_normal_qp_32_arc.json',
            'file_geo': path_report / 'RCS2_gf' / 'para_RCS2_gf_normal_qp__32_arc.json',
            'sigma_z': 23.3e-12,
        }
    },
    'RCS3': {
        '8': {
            'file_arc': path_esppu / 'GF_RCS3' / 'line_RCS3_gf_normal_qp_8_arc.json',
            'file_geo': path_esppu / 'GF_RCS3' / 'para_RCS3_gf_normal_qp__8_arc.json',
            'sigma_z': 16e-12,
        },
        '32': {
            'file_arc': path_report / 'RCS3_gf' / 'line_RCS3_gf_normal_qp_32_arc.json',
            'file_geo': path_report / 'RCS3_gf' / 'para_RCS3_gf_normal_qp__32_arc.json',
            'sigma_z': 16e-12,
        }
    },
    'RCS4': {
        '8': {
            'file_arc': path_esppu / 'GF_RCS4' / 'line_RCS4_gf_normal_qp_8_arc.json',
            'file_geo': path_esppu / 'GF_RCS4' / 'para_RCS4_gf_normal_qp__8_arc.json',
            'sigma_z': 10e-12,
        },
        '32': {
            'file_arc': path_report / 'RCS4_gf' / 'line_RCS4_gf_normal_qp_32_arc.json',
            'file_geo': path_report / 'RCS4_gf' / 'para_RCS4_gf_normal_qp__32_arc.json',
            'sigma_z': 10e-12,
        }
    },
}

input = {
    'RCS1': {
        'n_turn_rcs' : 17,
        't_acc' : 0.34e-3,
    },
    'RCS2': {
        'n_turn_rcs' : 55,
        't_acc' : 1.1e-3,
    },
    'RCS3': {
        'n_turn_rcs' : 66,
        't_acc' : 2.37e-3,
    },
    'RCS4': {
        'n_turn_rcs' : 55,
        't_acc' : 6.37e-3,
    }
}

rcs_name = 'RCS1'
n_arc = '32'
n_turn_rcs = input[rcs_name]['n_turn_rcs']
t_acc = input[rcs_name]['t_acc']
file_arc = rcs_files[rcs_name][n_arc]['file_arc']
file_geo = rcs_files[rcs_name][n_arc]['file_geo']

line = xt.Line.from_json(file_arc)
rcs = RingGeometry(file_geo)
nb_arc_tot = n_turn_rcs*rcs.nb_arc

EB = EB_error(
    line = line, 
    rcs = rcs, 
    t_acc = t_acc, 
    n_turn_rcs = n_turn_rcs,
    )

EB.build_rcs_ref()
line_rcs = EB.rcs_ref

delta, survived, lost = scan_momentum_acceptance(
    line_rcs,
    delta_range=(-0.20, 0.20),
    n_delta=401,
)

plt.figure(figsize=(9, 3))

plt.scatter(
    delta[survived],
    np.ones(np.sum(survived)),
    s=12,
    label="Transported",
)

plt.scatter(
    delta[lost],
    np.zeros(np.sum(lost)),
    s=12,
    marker="x",
    label="Lost",
)

plt.xlabel(r"Momentum deviation $\delta$")
plt.yticks([0, 1], ["Lost", "Transported"])
plt.ylim(-0.3, 1.3)
plt.grid(True, axis="x", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

if np.any(survived):
    delta_min = delta[survived].min()
    delta_max = delta[survived].max()

    print(f"Transported range: {delta_min:.5f} < delta < {delta_max:.5f}")
else:
    print("No particles transported.")