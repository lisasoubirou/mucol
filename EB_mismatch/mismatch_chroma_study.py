import matplotlib.pyplot as plt
import xtrack as xt
from class_geometry.ring_geometry.ring_geometry import RingGeometry
from EB_mismatch.EB_error import EB_error, adjust_qx_arc
from optics.plot_optics import mask_common_s, compare_optics2ref
from pathlib import Path

if __name__ == '__main__':

    path_lattices = Path(__file__).resolve().parent.parent
    path_08_25 = path_lattices / 'lattices/lattice_08_25'

    file_line = path_08_25 / 'line_RCS1_gf_normal_qp_8_arc_v2.json'
    arc = xt.Line.from_json(file_line)
    tab = arc.get_table()
    tw_noramp = arc.twiss(matrix_stability_tol=0.1)

    file_geo = path_08_25 / 'para_RCS1_gf_normal_qp__8_arc_v2.json'
    rcs = RingGeometry(file_geo)

    t_acc = 0.34 * 1e-3
    n_turn_rcs = 17

    EB = EB_error(
            line = arc, 
            rcs = rcs,  
            t_acc = t_acc, 
            n_turn_rcs = n_turn_rcs, 
            )

    # rg = range(8)
    rg = range(n_turn_rcs * rcs.nb_arc)
    EB.build_rcs_RMatrix_all(rg)
    rcs_RMatrix, tw_rcs_RMatrix = EB.rcs_RMatrix_all, EB.tw_rcs_RMatrix_all
    EB.build_rcs_ref(rg)
    rcs_ref, tw_ref = EB.rcs_ref, EB.tw_rcs_ref

    arc = xt.Line.from_json(file_line)
    tunes = {'qx': 1.7, 'qy': 1.7, 'dqx': 0., 'dqy': 0.}

    if tunes is not None:
        arc_chroma = adjust_qx_arc(
            line = arc,
            qx = tunes['qx'],
            qy = tunes['qy'],
            dqx = tunes['dqx'],
            dqy = tunes['dqy'],
            )
    
    tw_chroma = arc_chroma.twiss(matrix_stability_tol=0.1)

    EB_chroma = EB_error(
        line = arc_chroma, 
        rcs = rcs, 
        t_acc = t_acc, 
        n_turn_rcs = n_turn_rcs,
        )
    
    EB_chroma.build_rcs_RMatrix_all(rg)
    rcs_RMatrix_nochroma = EB_chroma.rcs_RMatrix_all
    EB_chroma.build_rcs_ref(rg)
    rcs_ref_nochroma = EB_chroma.rcs_ref

    name_map = rcs_ref_nochroma.get_elements_of_type(xt.LineSegmentMap)[1]
    for el in name_map:
        name_marker = 'Marker' + el[14:]
        mmap = rcs_ref_nochroma[el]
        mmap2 = rcs_RMatrix_nochroma[el]
        mmap.betx = tw_ref['betx', name_marker]
        mmap2.betx = tw_ref['betx', name_marker]
        mmap.bety  = tw_ref['bety', name_marker]
        mmap2.bety = tw_ref['bety', name_marker]
        mmap.dx = tw_ref['dx', name_marker]
        mmap2.dx = tw_ref['dx', name_marker]
        mmap.alfx = tw_ref['alfx', name_marker]
        mmap2.alfx = tw_ref['alfx', name_marker]
        mmap.alfy = tw_ref['alfy', name_marker]
        mmap2.alfy = tw_ref['alfy', name_marker]
    
    tw_ref_nochroma = rcs_ref_nochroma.twiss(init=EB_chroma.tw_init)
    tw_rcs_RMatrix_nochroma = rcs_RMatrix_nochroma.twiss(init=EB_chroma.tw_init)

    mask1, mask2 = mask_common_s(tw1=tw_rcs_RMatrix, tw2=tw_rcs_RMatrix_nochroma)

    print('Compare x and delta, with and without chromaticity correction')
    plt.figure(figsize=(6.4, 5))
    plt.plot(tw_rcs_RMatrix.s[mask1]*1e-3, tw_rcs_RMatrix_nochroma.x[mask2] - tw_rcs_RMatrix.x[mask1], color='k')
    plt.ylabel(r'x [m]')  
    plt.xlabel('s [km]')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6.4, 5))
    plt.plot(tw_rcs_RMatrix.s[mask1]*1e-3, tw_rcs_RMatrix_nochroma.delta[mask2] - tw_rcs_RMatrix.delta[mask1], color='k')
    plt.ylabel(r'$\delta$')  
    plt.xlabel('s [km]')
    plt.tight_layout()
    plt.show()

    #Compare to refs
    print('Differences from reference')
    compare_optics2ref(tw_ref, tw_rcs_RMatrix, tw_rcs_RMatrix_nochroma, ['No correction', 'Corrected'])

    tw_rcs_RMatrix_nochroma = rcs_RMatrix_nochroma.twiss(init=EB_chroma.tw_init, compute_chromatic_properties=True)
    tw_rcs_RMatrix = rcs_RMatrix.twiss(init=EB.tw_init, compute_chromatic_properties=True)
    
    plt.figure()
    plt.plot(tw_rcs_RMatrix.s * 1e-3, tw_rcs_RMatrix.dx, label='No correction', color='tab:blue')
    plt.plot(tw_rcs_RMatrix_nochroma.s * 1e-3, tw_rcs_RMatrix_nochroma.dx, label='Correction', color='tab:orange')
    plt.xlabel('s [km]')
    plt.ylabel('$D_x$ [m]')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()




