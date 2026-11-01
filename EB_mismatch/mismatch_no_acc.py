"""
Tracking study without acceleration for a given lattice.
Includes the EB error with slicing method. 

Removes RefEnergyIncrease and RF cavity elements to perform
a tracking study at constant reference energy.
"""

import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xpart as xp
from class_geometry.ring_geometry.ring_geometry import RingGeometry
import copy
from EB_mismatch.EB_error import EB_error
from tracking.track_function import emit_x_per, emit_y_per
from optics.generate_fodo import plot_twiss
from pathlib import Path


if __name__ == '__main__':
    
    path = Path(__file__).resolve().parent

    # RCS Parameters
    file_geo = path / 'lattices/lattice_08_25/para_RCS1_gf_normal_qp__8_arc_v2.json'
    rcs = RingGeometry(file_geo)
    n_turn_rcs = 17
    t_acc = 0.34*1e-3
    nb_arc_tot = n_turn_rcs*rcs.nb_arc

    # Importing arc
    file_arc =  path / 'lattices/lattice_08_25/line_RCS1_gf_normal_qp_8_arc_v2.json'
    line_arc = xt.Line.from_json(file_arc)
    tw_arc = line_arc.twiss(method='6d', matrix_stability_tol=0.7)

    n_slice = 10
    EB = EB_error(
        line=line_arc, 
        rcs=rcs, 
        t_acc=t_acc, 
        n_turn_rcs=n_turn_rcs, 
        n_slice=n_slice
        )
    arc1_pert, tw_pert = EB.build_first_arc('sliced', 0)

    tab_arc_pert = arc1_pert.get_table()
    
    arc1_pert.discard_tracker()
    for el_type, el_name in zip(tab_arc_pert.element_type, tab_arc_pert.name):
        if el_type == 'ReferenceEnergyIncrease':
            arc1_pert.remove(name=el_name)
        if el_type == 'Cavity':
            arc1_pert.remove(name=el_name)
    
    tw_arc_prop = arc1_pert.twiss6d(init=EB.tw_init)
    tw_arc_period = arc1_pert.twiss4d()

    plt.figure()
    plt.plot(tw_arc_prop.s, tw_arc_prop.x * 1e3, label='Propagating')
    plt.plot(tw_arc_period.s, tw_arc_period.x * 1e3, label='Periodic')
    plt.legend()
    plt.xlabel('s [m]')
    plt.ylabel('x [mm]')
    plt.show()

    env = xt.Environment()
    env.import_line(arc1_pert, suffix_for_common_elements='', line_name='arc')
    env.particle_ref = arc1_pert.particle_ref
    components = [
        env.new(f'arc_{i}', env.arc, mode='replica')
        for i in range(nb_arc_tot)
    ]
    rcs = env.new_line(
        name='rcs',
        components=components
    )

    EB.tw_init.element_name = rcs.element_names[0]
    EB.tw_init.delta=0
    tw_prop = rcs.twiss4d(init=EB.tw_init)
    plot_twiss(tw_prop)
    plt.figure()
    plt.plot(tw_prop.s, tw_prop.x)
    plt.xlabel('s [m]')
    plt.ylabel('x [m]')
    plt.show()
    tw_period = rcs.twiss4d()
    plot_twiss(tw_period)
    plt.figure()
    plt.plot(tw_period.s, tw_period.x)
    plt.xlabel('s [m]')
    plt.ylabel('x [m]')
    plt.show()

    #Generate particle distribution matched at injection
    n_part = 5000
    nemitt_x = 25e-6
    nemitt_y = 25e-6
    nemitt_s = 0.025
    context = xo.ContextCpu()

    (x_gaussian, px_gaussian) = xp.generate_2D_gaussian(n_part)
    (y_gaussian, py_gaussian) = xp.generate_2D_gaussian(n_part)

    particles = arc1_pert.build_particles(
                x_norm = x_gaussian, px_norm = px_gaussian,
                y_norm = y_gaussian, py_norm = py_gaussian,
                # matrix_stability_tol = 0.1,
                nemitt_x = nemitt_x, 
                nemitt_y = nemitt_y, 
                zeta = 0, delta = 0,
                method='4d'
    )

    particles0 = copy.deepcopy(particles)

    arc1_pert.build_tracker(_context=context)

    n_turns = nb_arc_tot
    arc1_pert.track(
        particles, 
        num_turns=n_turns, 
        turn_by_turn_monitor=True, 
        freeze_longitudinal=True)
    rec = arc1_pert.record_last_track
    particles.sort(interleave_lost_particles=True)

    emitx = emit_x_per(rec)
    emitx_norm = emitx*rec.gamma0[0][0]
    emity = emit_y_per(rec)
    emity_norm = emity*rec.gamma0[0][0]
    emits_norm = np.zeros(len(emitx_norm))
