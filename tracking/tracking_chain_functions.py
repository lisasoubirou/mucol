import xpart as xp
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc

def calculate_transmission_cumul(particles, total_particles, n_turn, cumulative_lost_particles=0):
    particles.unhide_lost_particles()
    lost_turns = sorted(particles.at_turn[particles.state < 0])
    particles.hide_lost_particles()
    transmissions = []
    lost_particles = cumulative_lost_particles  # Start with any previously lost particles
    for turn in range(1, n_turn + 1):
        if turn in lost_turns:
            lost_particles += lost_turns.count(turn)
        transmission = (total_particles - lost_particles) / total_particles
        transmissions.append(transmission)
    return transmissions, lost_particles

def transfer_matrix(tw0, tw1, particles):

    Ax = np.sqrt(tw1.betx[0]/tw0.betx[0])
    Bx = (tw0.alfx[0]-tw1.alfx[0])/np.sqrt(tw0.betx[0]*tw1.betx[0])
    Ay = np.sqrt(tw1.bety[0]/tw0.bety[0])
    By = (tw0.alfy[0]-tw1.alfy[0])/np.sqrt(tw0.bety[0]*tw1.bety[0])

    particles.px = Bx*particles.x + particles.px/Ax 
    particles.x = Ax*particles.x  
    particles.py = By*particles.y + particles.py/Ay
    particles.y = Ay*particles.y

def calc_separatrix(line, rcs, tw, particles, title = None):
        h_rf = np.round(1.3e9*tw.T_rev0*rcs.nb_arc)
        energy = particles.p0c[0]
        # print('energy', energy)
        phase = line['cavity_1_l'].lag
        # print('phase', phase)
        energy_increment_per_arc = line['RF_acc_1_l'].Delta_p0c*2
        # print('energy incr', energy_increment_per_arc)
        volt_arc = line['cavity_1_l'].voltage*2

        #RF bucket defined here on RCS => careful with consistency!!
        machine_rf_bucket = xp.longitudinal.rf_bucket.RFBucket(
                circumference = rcs.circumference/rcs.nb_arc,
                gamma = energy/xp.MUON_MASS_EV, mass_kg = sc.physical_constants['muon mass'][0],
                charge_coulomb = sc.e, alpha_array = [tw.momentum_compaction_factor], 
                p_increment = energy_increment_per_arc * sc.e / sc.c, 
                harmonic_list = [h_rf/rcs.nb_arc], voltage_list = [volt_arc], 
                phi_offset_list = np.atleast_1d((phase)*np.pi/180),
                z_offset = None
                )
        xx = np.linspace(machine_rf_bucket.z_left, machine_rf_bucket.z_right, 1000)
        yy = machine_rf_bucket.separatrix(xx)

        plt.figure()
        plt.title(title)    
        plt.scatter(particles.zeta, particles.delta*particles.p0c/1e9, label='Particle distribution', s=2)
        plt.plot((xx-machine_rf_bucket.z_sfp), yy*energy/1e9, color='tab:orange', label='RF bucket')
        plt.plot((xx-machine_rf_bucket.z_sfp), -yy*energy/1e9, color='tab:orange')
        plt.xlabel('z [m]')
        plt.ylabel(r'$\Delta E$ [GeV]')
        plt.legend()
        plt.show()
        return (machine_rf_bucket)

