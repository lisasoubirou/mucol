import numpy as np
import xpart as xp
import xtrack as xt
import scipy.constants as sc
from scipy.interpolate import CubicSpline
from class_geometry.ring_geometry.ring_geometry import RingGeometry

muon_mass_kg=sc.physical_constants['muon mass'][0]

def generate_particles(
        rcs : RingGeometry, 
        line : xt.Line, 
        tw : xt.TwissTable, 
        n_part : int, 
        n_turn_rcs : int, 
        e_x : float, 
        e_y : float, 
        e_s : float, 
        RF_freq : float =1.3e9,
        gamma : float = None, 
        sync_phase: float = 45, 
        adapt : bool =False) -> xp.Particles: 
    """Generate matched particle distribution.

    Args:
        rcs (RingGeometry): RCS geometry.
        line (xt.Line): RCS line.
        tw (xt.TwissTable): Twiss table.
        n_part (int): Number of particles.
        n_turn_rcs (int): Number of turns
        e_x (float): Normalised horizontal emittance. 
        e_y (float): Normalised vertical emittance. 
        e_s (float): Normalised longitudinal emittance. 
        RF_freq (float, optional): RF frequency. Defaults to 1.3e9.
        gamma (float, optional): Gamma factor. Defaults to None, will be initialised to rcs.injection_energy. 
        sync_phase (float, optional): Synchronous phase. Defaults to 45.
        adapt (bool, optional): Transverse adaptation to higher orders. Defaults to False.

    Returns:
        xp.Particles: Matched particle distribution. 
    """
    
    if not gamma:
        gamma = rcs.inj_gamma
    
    line.build_tracker()
    # Generate transverse coordinates
    (x_gaussian, px_gaussian) = xp.generate_2D_gaussian(n_part)
    (y_gaussian, py_gaussian) = xp.generate_2D_gaussian(n_part)
    particles = line.build_particles(
                x_norm = x_gaussian, px_norm = px_gaussian,
                y_norm = y_gaussian, py_norm = py_gaussian,
                matrix_stability_tol = 0.1,
                nemitt_x = e_x, 
                nemitt_y = e_y, 
                zeta = 0, delta = 0,
    )

    try:
        volt_arc = line['cavity_1_l'].voltage * 2
        energy_increment_per_arc = line['RF_acc_1_l'].Delta_p0c * 2
    except:
        energy_increment_per_arc = (rcs.extraction_energy-rcs.injection_energy)/n_turn_rcs/rcs.nb_arc
        phase=180-sync_phase
        volt_arc=energy_increment_per_arc/np.sin(phase*np.pi/180)
    
    h_rf = np.round(RF_freq*tw.T_rev0*rcs.nb_arc)

    #Compute RF bucket 
    rfbucket = xp.longitudinal.rf_bucket.RFBucket(
            circumference = rcs.circumference / rcs.nb_arc,
            gamma = gamma, #rcs.inj_gamma,
            mass_kg = muon_mass_kg,
            charge_coulomb = sc.e,
            alpha_array = np.atleast_1d(tw['momentum_compaction_factor']),
            harmonic_list = np.atleast_1d(h_rf / rcs.nb_arc),
            voltage_list = np.atleast_1d(volt_arc),
            phi_offset_list =  np.atleast_1d(sync_phase),
            # phi_offset_list = np.atleast_1d(tw.zeta[0]*RF_freq/sc.c*2*np.pi),
            p_increment = energy_increment_per_arc * sc.e / sc.c) 

    #Compute matched longitudinal coordinates to the line
    matcher = xp.longitudinal.rfbucket_matching.RFBucketMatcher(
            rfbucket = rfbucket,
            distribution_type = xp.longitudinal.rfbucket_matching.ThermalDistribution,
            #sigma_z=sigma_z,
            epsn_z = 4 * np.pi * e_s
                        )

    #Overwrite longitudinal coordinates
    z_particles, delta_particles, _, _ = matcher.generate(macroparticlenumber=n_part)
    particles.zeta = z_particles - np.mean(z_particles) #-z_sync
    particles.delta = delta_particles

    #Adapt to higher order of delta
    if adapt:
        limit = np.percentile(particles.delta, 95)
        print(limit)
        arr_delta = np.linspace(-0.04, 0.04, 41)
        # arr_delta = np.linspace(-limit, limit, 41)
        arr_twiss = np.zeros((41, 8))
        for i, delta in enumerate(arr_delta):
            tw_6d = line.twiss(method='4d', matrix_stability_tol=0.1, delta0=delta)
            arr_twiss[i, 0] = tw_6d.betx[0]
            arr_twiss[i, 1] = tw_6d.alfx[0]
            arr_twiss[i, 2] = tw_6d.bety[0]
            arr_twiss[i, 3] = tw_6d.alfy[0]
            arr_twiss[i, 4] = tw_6d.x[0]
            arr_twiss[i, 5] = tw_6d.px[0]
            arr_twiss[i, 6] = tw_6d.y[0]
            arr_twiss[i, 7] = tw_6d.py[0]
        twiss_delta = CubicSpline(arr_delta, arr_twiss, bc_type='not-a-knot')
        tw_loc = twiss_delta(particles.delta)
        Ax = np.sqrt(tw_loc[:, 0] / tw.betx[0])
        Bx = (tw.alfx[0] - tw_loc[:,1]) / np.sqrt(tw.betx[0] * tw_loc[:,0])
        Ay = np.sqrt(tw_loc[:, 2] / tw.bety[0])
        By = (tw.alfy[0] - tw_loc[:,3]) / np.sqrt(tw.bety[0] * tw_loc[:,2])
        particles.px = Bx * particles.x + particles.px / Ax + tw_loc[:, 5]
        particles.x = Ax * particles.x  + tw_loc[:, 4]
        particles.py = By * particles.y + particles.py / Ay  + tw_loc[:, 7]
        particles.y = Ay * particles.y  + tw_loc[:, 6]

    return particles
