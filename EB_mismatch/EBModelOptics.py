import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import xpart as xp
from class_geometry.ring_geometry.ring_geometry import RingGeometry
import scipy.constants as sc
from pathlib import Path

class EBModelDip:
    """
    Class for modeling a dipole magnet with a magnetic field ramp (B'(s)).
    Includes analytical dispersion models, transfer matrices, 
    perturbation matrices, and offset computations.
    """

    def __init__(
            self, 
            h : float, 
            L : float, 
            s0 : float, 
            sr : float, 
            bprime : float, 
            brho : float, 
            gamma0 : float
            ):
        """
        Initialize dipole parameters.

        Args:
            h (float): Reference bending [1/m].
            L (float): Dipole length [m].
            s0 (float): Entrance face coordinate [m].
            sr (float): Magnetic reference coordinate [m].
            bprime (float): Magnetic field ramp [T/m].
            brho (float): Magnetic rigidity [Tm].
            gamma0 (float): Relativistic gamma factor.
        """
        self.h = h
        self.L = L
        self.s0 = s0
        self.sr = sr
        self.bprime = bprime
        self.brho = brho
        self.gamma0 = gamma0
        self.dk = bprime / brho
        self.beta0 = np.sqrt(1 - 1 / gamma0**2)

        self.dR = self.compute_dR_matrix()
        self.dX = self.compute_offset_dX()
        self.Mdip = self.compute_Mdip()
        self.dM_error = np.identity(6) + np.matmul(self.dR, np.linalg.inv(self.Mdip))

    def compute_Mdip(self):
        """
        Compute the nominal transfer matrix of a dipole.

        Args:
            u (float): Propagation length [m].
        """
        h, beta0, gamma0, L = self.h, self.beta0, self.gamma0, self.L
        M = np.array([
            [np.cos(h*L),  np.sin(h*L) / h,  0, 0, 0, (1 - np.cos(h*L)) / (h * beta0)],
            [-h * np.sin(h*L), np.cos(h*L),  0, 0, 0, np.sin(h*L) / beta0],
            [0, 0, 1, L, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [-np.sin(h*L) / beta0, -(1 - np.cos(h*L)) / (h * beta0), 0, 0,
             1, (-(h*L - np.sin(h*L)) / (h * beta0**2)) + L / (beta0**2 * gamma0**2)],
            [0, 0, 0, 0, 0, 1]
        ])
        return M

    def compute_dR_matrix(self):
        """
        Compute the perturbation dR matrix of a dipole with varying k0(s).
        Returns a 6x6 NumPy array.
        """
        h, L, s0, sr, dk, beta0, gamma0 = (
            self.h, self.L, self.s0, self.sr, self.dk, self.beta0, self.gamma0
        )
        if abs(h) < 1e-6:
            M = np.array([
                [
                    -dk/3 * L**2 * (L + 3*s0 - 3*sr)* h + dk/120 * L**4 * (13*L + 40*(s0 - sr)) * h**3,
                    -dk/24* L**3 *(5*L + 12*(s0 - sr))* h + dk/360* L**5 *(14*L + 45*(s0 - sr)) * h**3,
                    0, 
                    0, 
                    0,
                    dk / 6 / beta0 * L**2 *(L + 3*s0 - 3*sr) - dk / 120 / beta0 * L**4 *(12*L + 35*(s0 - sr))* h**2
                ],
                [
                    -dk/2 * L * (L + 2*s0 - 2*sr) * h + dk/12 * L**3 * (L + 2*s0 - 2*sr) * h**3,
                    -dk / 6 * L**3 * h + dk/60 * L**5 * h**3,
                    0, 
                    0, 
                    0,
                    - dk /24 / beta0 * L**4 * h**2
                ],
                [
                    0, 
                    0, 
                    0,
                    -dk / 24 * L**3 *(L + 4*s0 - 4*sr) * h + dk/720 * L**5 * (L + 6*s0 - 6*sr)* h**3,
                    0, 
                    0
                ],
                [0, 0, 0, 0, 0, 0],
                [
                    - dk / 24 / beta0 * L**4 * h**2,
                    dk / 6 / beta0 * L**2 * (L + 3*s0 - 3*sr)- dk / 120 / beta0 * L**4 *(2*L + 5*s0 - 5*sr)*h**2,
                    0, 
                    0, 
                    0,
                    dk / 24 / (-1 + gamma0**2) * L**3 * (-L - 4*s0 + 4*sr+ 2*(L + 2*s0 - 2*sr)*gamma0**2) * h \
                        - dk / 720 / (-1 + gamma0**2) * L**5 * (L + 6*s0 - 6*sr - 3*(L + 2*s0 - 2*sr)*gamma0**2) * h**3
                ],
                [0, 0, 0, 0, 0, 0]
            ])
        else: 
            M = np.array([
                [
                    - (dk * (h * L * np.cos(h*L)
                            + np.sin(h*L) * (1 - 2 * np.cos(h*L) + 2 * h * (s0 - sr) * np.sin(h*L))
                            )) / (2 * h**2),
                    - (dk * (-2 * np.cos(h*L) + 2 * np.cos(h*L)**2
                            + h * (L + 2 * (s0 - sr) - 2*(s0 - sr)*np.cos(h*L)) * np.sin(h*L)
                            )) / (2 * h**3),
                    0, 
                    0, 
                    0,
                    (dk * (-2 * h * L - h * s0 + h * sr
                            + h * (L + 2 * (s0 - sr)) * np.cos(h*L)
                            - h * (s0 - sr) * np.cos(2*h*L)
                            + 3 * np.sin(h*L) - np.sin(2*h*L)
                        )) / (2 * h**3 * beta0)
                ],
                [
                    -0.5 * dk * (L + 2 * (s0 - sr)) * np.sin(h*L),
                    (dk * (h * L * np.cos(h*L) - np.sin(h*L))) / (2 * h**2),
                    0, 
                    0, 
                    0,
                    (dk * (-2 + 2 * np.cos(h*L) + h * L * np.sin(h*L))) / (2 * h**2 * beta0)
                ],
                [0, 
                0, 
                0,
                (dk * (2 - h**2 * L * (L + 2*(s0 - sr))
                        - 2 * np.cos(h*L) + 2 * h *(s0 - sr) * np.sin(h*L))) / (2 * h**3),
                0, 
                0],
                [0, 0, 0, 0, 0, 0],
                [
                    (dk * (-2 + 2 * np.cos(h*L) + h * L * np.sin(h*L))) / (2 * h**2 * beta0),
                    (dk * (2 * h * (s0 - sr)
                        - h * (L + 2*(s0 - sr)) * np.cos(h*L)
                        + np.sin(h*L))) / (2 * h**3 * beta0),
                    0, 
                    0, 
                    0,
                    - ( dk * ( -2 + h**2 * L**2 + 2 * h**2 *L * s0 - 2 * h**2* L * sr
                            + 2 * np.cos(h*L) - 2 * h * s0 * np.sin(h*L) + 2 * h * sr * np.sin(h*L)
                            + h * (L + 2*(s0 - sr)) * (-h * L + np.sin(h*L)) * gamma0**2
                            )
                    ) / (2*h**3 * (-1 + gamma0**2))
                ],
                [0, 0, 0, 0, 0, 0]
            ])
        return M

    def compute_offset_dX(self):
        """
        Compute the offset dX vector for a dipole with varying k0(s).
        Returns a 6x1 NumPy array.
        """
        h, L, s0, sr, dk, beta0 = (
            self.h, self.L, self.s0, self.sr, self.dk, self.beta0
        )

        if abs(h) < 1e-6:
            dX =  np.array([
                -dk * (L**3 / 6 + (s0 - sr) * L**2 / 2) + dk / 120 * L**4 * (L + 5*s0 - 5*sr)* h**2,
                -dk * (L**2 / 2 + (s0 - sr) * L) + dk /24 * L**3 * (L + 4*s0 - 4*sr) * h**2,
                0,
                0,
                dk / 24 * L**3 *(L + 4*s0 - 4*sr) * h - dk / 720 * L**5 * (L + 6*s0 - 6*sr) * h**3,
                0
            ])
        else:
            dX = np.array([
                    (dk * (-h * (L + s0 - sr) + h * (s0 - sr) * np.cos(h*L) + np.sin(h*L))) / h**3,
                    (dk * (-1 + np.cos(h*L) - h * (s0 - sr) * np.sin(h*L))) / h**2,
                    0, 0,
                    (dk * (-2 + h**2 * L * (L + 2*(s0 - sr))
                        + 2 * np.cos(h*L) + 2 * h * (-s0 + sr) * np.sin(h*L))) / (2 * h**3 * beta0),
                    0
                ])
        return dX        


def Ddx_1(L, s0, sr, bprime, brho):
    return 1 / 6 * L**2 * (L + 3*(s0 -sr)) * bprime / brho

def Ddx_matrix(L, s0, sr, bprime, brho, h):
    return - bprime / brho * (np.cos(h*L) * (-h*L + 4 * np.sin(h*L)) + 
    np.sin(h*L) * (-3 + h**2 * L * (L + 2*(s0 - sr)) - 4 * h * (s0 - sr) * np.sin(h*L))) / 4 / h**3

class EBModelQuad:
    """
    Class for modeling a quadrpole magnet with a magnetic field ramp (B'(s)).
    Includes analytical dispersion models, transfer matrices, 
    perturbation matrices, and offset computations.
    """

    def __init__(
            self, 
            kref : float, 
            L : float, 
            s0 : float, 
            sr : float, 
            Gprime : float, 
            brho : float, 
            gamma0 : float
            ):
        """Initialize quadrupole parameters

        Args:
            kref (float): k1 strength of the quadrupole [1/m²].
            L (float): Length of the quadrupole [m].
            s0 (float): Entrance face coordinate [m].
            sr (float): Magnetic reference coordinate [m].
            Gprime (float): Gradient ramp [T/m²]
            brho (float): Magnetic rigidity [Tm].
            gamma0 (float): Relativistic gamma factor.
        """

        self.kref = kref
        self.L = L
        self.s0 = s0
        self.sr = sr
        self.Gprime = Gprime
        self.brho = brho
        self.gamma0 = gamma0
        self.dk1 = Gprime / brho
        self.beta0 = np.sqrt(1 - 1 / gamma0**2)

        self.dR = self.compute_dR_matrix()
        self.dX = np.zeros(6)
        self.Mquad = self.compute_Mquad()
        self.dM_error = np.identity(6) + np.matmul(self.dR, np.linalg.inv(self.Mquad))
    
    def compute_dR_matrix(self):

        kref, L, s0, sr = self.kref, self.L, self.s0, self.sr
        dk = self.Gprime / self.brho

        defoc = False
        if kref < 0:
            defoc = True
            kref = - kref
        
        sqrt_kref = np.sqrt(kref)
        M = np.array([
            [
                - (dk * (sqrt_kref * L * np.cos(sqrt_kref * L) + (-1 + kref * L * (L + 2*s0 - 2*sr)) * np.sin(sqrt_kref * L))) / (4 * kref**(3/2)),
                (dk * (L + 2*s0 - 2*sr) * (sqrt_kref * L * np.cos(sqrt_kref * L) - np.sin(sqrt_kref * L))) / (4 * kref**(3/2)),
                0, 0, 0, 0
            ],
            [
                - (dk * (L + 2*s0 - 2*sr) * (sqrt_kref * L * np.cos(sqrt_kref * L) + np.sin(sqrt_kref * L))) / (4 * np.sqrt(kref)),
                - (dk * (-sqrt_kref * L * np.cos(sqrt_kref * L) + (1 + kref * L * (L + 2*s0 - 2*sr)) * np.sin(sqrt_kref * L))) / (4 * kref**(3/2)),
                0, 0, 0, 0
            ],
            [
                0, 0,
                (dk * (-np.sqrt(kref) * L * np.cosh(np.sqrt(kref) * L)
                    + (1 + kref * L * (L + 2*s0 - 2*sr)) * np.sinh(np.sqrt(kref) * L))
                    ) / (4 * kref**(3/2)),
                (dk * (L + 2*s0 - 2*sr)
                    * (np.sqrt(kref) * L * np.cosh(np.sqrt(kref) * L)
                        - np.sinh(np.sqrt(kref) * L))
                    ) / (4 * kref**(3/2)),
                0, 0
            ],
            [
                0, 0,
                (dk * (L + 2*s0 - 2*sr)
                    * (np.sqrt(kref) * L * np.cosh(np.sqrt(kref) * L)
                        + np.sinh(np.sqrt(kref) * L))
                    ) / (4 * np.sqrt(kref)),
                (dk * (np.sqrt(kref) * L * np.cosh(np.sqrt(kref) * L)
                    + (-1 + kref * L * (L + 2*s0 - 2*sr))
                        * np.sinh(np.sqrt(kref) * L))
                    ) / (4 * kref**(3/2)),
                0, 0
            ],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
            ])

        # If defocusing quadrupole
        if defoc:
            Md = np.zeros([6,6])
            Md[0,0], Md[0,1] = -M[2,2], -M[2,3]
            Md[1,0], Md[1,1] = -M[3,2], -M[3,3]
            Md[2,2], Md[2,3] = -M[0,0], -M[0,1]
            Md[3,2], Md[3,3] = -M[1,0], -M[1,1]

            M = Md

        return M
    
    def compute_Mquad(self):
        kref, L = self.kref, self.L

        # Focusing quad
        if kref > 0 :
            sqrt_kref = np.sqrt(kref)
            M = np.array([
                [np.cos(sqrt_kref * L), np.sin(sqrt_kref * L) / sqrt_kref, 0, 0, 0, 0],
                [-sqrt_kref * np.sin(sqrt_kref * L), np.cos(sqrt_kref * L), 0, 0, 0, 0],
                [0, 0, np.cosh(sqrt_kref * L), np.sinh(sqrt_kref * L) / sqrt_kref, 0, 0],
                [0, 0, sqrt_kref * np.sinh(sqrt_kref * L), np.cosh(sqrt_kref * L), 0, 0],
                [0, 0, 0, 0, 1, L / (self.beta0**2 * self.gamma0**2)],
                [0, 0, 0, 0, 0, 1]
            ])

        # Defocusing quad
        else:
            sqrt_neg_kref = np.sqrt(-kref)
            M = np.array([
                [np.cosh(sqrt_neg_kref*L), np.sinh(sqrt_neg_kref*L)/sqrt_neg_kref, 0, 0, 0, 0],
                [sqrt_neg_kref*np.sinh(sqrt_neg_kref*L), np.cosh(sqrt_neg_kref*L), 0, 0, 0, 0],
                [0, 0, np.cos(sqrt_neg_kref*L), np.sin(sqrt_neg_kref*L)/sqrt_neg_kref, 0, 0],
                [0, 0, -sqrt_neg_kref*np.sin(sqrt_neg_kref*L), np.cos(sqrt_neg_kref*L), 0, 0],
                [0, 0, 0, 0, 1, L/(self.beta0**2*self.gamma0**2)],
                [0, 0, 0, 0, 0, 1]
            ])

        return M

def X_model(L, s0, sr, bprime, brho):
    return L**2 / 6 * (L + 3*s0 -3*sr) * bprime / brho

def X_matrix(L, s0, sr, bprime, brho, h):
    return bprime / brho * ( -h * (L+s0-sr) + h * (s0-sr) * np.cos(h*L) + np.sin(h*L)) / h**3

if __name__ == '__main__':

    path = Path(__file__).resolve().parent.parent
    file_path = path / 'lattices/lattice_consolidated_ESPPU/para_RCS1_gf_normal_qp__8_arc_v3.json'
    rcs = RingGeometry(file_path)
    t_acc = 0.34 * 1e-3

    L_dip = 15.225005150213795
    start_dip = 66.38883428663814
    start_arc_s = 0
    n_slice = 1000
    length_arc = rcs.circumference / rcs.nb_arc
    t_ref = 0
    href = rcs.ref_bending

    sr = length_arc / 2
    tshift = sr / sc.c / t_acc

    l_slice = L_dip / n_slice
    s_slice = start_arc_s + start_dip + np.linspace(0, L_dip, n_slice + 1) - l_slice / 2
    t_slice = s_slice / sc.c / t_acc - tshift

    B_slice = rcs.dipole_families['BNC']['B'](t_slice)
    energy_ref = rcs.injection_energy + (rcs.extraction_energy - rcs.injection_energy) * t_ref
    rho_ref = energy_ref / sc.c / rcs.dipole_families['BNC']['B'](t_ref)
    rho_slice = energy_ref / sc.c / B_slice

    bprime = (rcs.dipoles['BNC'][1] - rcs.dipoles['BNC'][0]) / t_acc / sc.c
    brho = rcs.inj_Brho
    Ldip_scan = np.linspace(0, L_dip, n_slice + 1)

    sliced_el = []
    normal_el = []
    dip_names= []

    for i in range(1, n_slice + 1):
        sliced_el.append(xt.Bend(
            k0=1 / rho_slice[i], 
            # h=1 / rho_ref,
            angle = 1 / rho_ref * l_slice,
            length=l_slice))
        dip_names.append(f'dip_slice_{i}')
        normal_el.append(xt.Bend(
            k0=1 / rho_ref, 
            angle = 1 / rho_ref * l_slice,
            # h=1 / rho_ref, 
            length=l_slice))

    # Line : Dipole with increasing k0(s)
    dip_sliced = xt.Line(
            elements = sliced_el, 
            element_names = dip_names)
    dip_sliced.particle_ref = xp.Particles(p0c=energy_ref, q0=1, mass0=xp.MUON_MASS_EV)

    # Line : Dipole with constant k0=h
    dip_normal = xt.Line(
            elements = normal_el,
            element_names = dip_names)
    dip_normal.particle_ref = dip_sliced.particle_ref


    # Build analytical model
    dip_model = EBModelDip(
            h = href, 
            L = L_dip, 
            s0 = start_dip + start_arc_s, 
            sr = sr, 
            bprime = bprime, 
            brho = brho, 
            gamma0 = rcs.inj_gamma
            )

    dip_model = EBModelDip(
        h = 0.0017194865213113746 ,
        L = 15.22543995282306 ,
        s0 = 66.38883428663814 ,
        sr = 374.375 ,
        bprime = 1.4114258061304369e-05,
        brho = 210.14508443439587 ,
        gamma0 = 596.2612969704611
        )

    Mdip = dip_model.Mdip
    deltaR = dip_model.dR
    dX = dip_model.dX
    dM = dip_model.dM_error
    
    dR_matrix_el = xt.FirstOrderTaylorMap(length=0, m0=dX, m1=dM)

    # Line : Dipole + Error Map
    dip_dR = xt.Line(
        elements = normal_el + [dR_matrix_el],
        element_names = dip_names + ['dR_matrix'])
    dip_dR.particle_ref = dip_sliced.particle_ref

    # Compute Twiss
    beta_init = 30
    dx_init = 0
    tw_sliced = dip_sliced.twiss(betx=beta_init, bety=beta_init, 
                    dx=dx_init)
    tw_normal = dip_normal.twiss(betx=beta_init, bety=beta_init, 
                    dx=dx_init)
    tw_dR = dip_dR.twiss(betx=beta_init, bety=beta_init, 
                    dx=dx_init)

    # Compute R matrices
    R_pert = tw_sliced.get_R_matrix(start='dip_slice_1', end='_end_point')
    R_normal = tw_normal.get_R_matrix(start='dip_slice_1', end='_end_point')
    R_model = tw_dR.get_R_matrix(start='dip_slice_1', end='_end_point')

    print('Residual')
    print(R_pert - R_normal - deltaR)
    print('Residual with dR')
    print(R_pert - R_model)

    #For quadrupole
    el_sqp = []
    el_normalqp = []
    name_qp = []

    gamma0 = rcs.inj_gamma
    kref = - 0.001
    gradient_inj = kref * rcs.func_Brho(0)
    gradient_ext = kref * rcs.func_Brho(1)
    Gprime = (gradient_ext-gradient_inj) / t_acc / sc.c
    L_quad = 5

    start_dip = 0
    start_arc_s = 0
    l_slice = L_quad / n_slice
    s_slice = start_arc_s + start_dip + np.linspace(0, L_quad, n_slice+1) - l_slice / 2
    t_slice = s_slice / sc.c / t_acc - tshift

    gradient_pulsed = gradient_inj + (gradient_ext - gradient_inj) * t_slice
    k_pulsed = gradient_pulsed / rcs.func_Brho(t_ref)

    EBQuad = EBModelQuad(
        kref = kref, 
        L = L_quad, 
        s0 = start_dip, 
        sr = sr, 
        Gprime = Gprime, 
        brho = brho, 
        gamma0 = gamma0)
    
    dX = EBQuad.dX
    dM = EBQuad.dM_error
    
    dR_matrix_el = xt.FirstOrderTaylorMap(length=0, m0=dX, m1=dM)

    for i in range(1, n_slice + 1):
        el_sqp.append(xt.Quadrupole(k1=k_pulsed[i], length=l_slice))
        el_normalqp.append(xt.Quadrupole(k1=gradient_inj/rcs.func_Brho(0), length=l_slice))
        name_qp.append(f'qd_slice_{i}')

    qp_sliced = xt.Line(
        elements = el_sqp,
        element_names = name_qp)
    qp_sliced.particle_ref = xp.Particles(p0c=energy_ref, q0=1, mass0=xp.MUON_MASS_EV)

    qp_normal = xt.Line(
        elements = el_normalqp,
        element_names = name_qp)
    qp_normal.particle_ref = xp.Particles(p0c=energy_ref, q0=1, mass0=xp.MUON_MASS_EV)

    qp_dR = xt.Line(
        elements = el_normalqp + [dR_matrix_el],
        element_names = name_qp + ['Rmatrix'])
    qp_dR.particle_ref = xp.Particles(p0c=energy_ref, q0=1, mass0=xp.MUON_MASS_EV)

    # Compute Twiss
    beta_init = 30
    dx_init = 0
    tw_qpsliced = qp_sliced.twiss(betx=beta_init, bety=beta_init, 
                    dx=dx_init)
    tw_qpnormal = qp_normal.twiss(betx=beta_init, bety=beta_init, 
                    dx=dx_init)
    tw_qpdR = qp_dR.twiss(betx=beta_init, bety=beta_init, 
                    dx=dx_init)

    # Compute R matrices
    R_pert = tw_qpsliced.get_R_matrix(start='qd_slice_1', end='_end_point')
    R_normal = tw_qpnormal.get_R_matrix(start='qd_slice_1', end='_end_point')
    R_dR = tw_qpdR.get_R_matrix(start='qd_slice_1', end='_end_point')
    
    # print(R_pert - R_normal)
    print(R_pert - R_normal - EBQuad.dR)
    print(R_pert - R_dR)

    #Plot Dx for the oversimple model and model
    plot_Dx = False
    if plot_Dx:
        Ddx_model_1 = Ddx_1(
            L = Ldip_scan, 
            s0 = start_dip + start_arc_s, 
            sr = sr, 
            bprime = bprime, 
            brho = brho
            )
                        
        print(tw_sliced.dx[-1] - tw_normal.dx[-1])

        plt.figure()
        plt.plot(tw_sliced.s, tw_sliced.dx - tw_normal.dx, label='Xsuite', color='tab:blue')
        plt.scatter(Ldip_scan[::50], Ddx_model_1[::50], label='Model 1', color='tab:red', marker='+')
        plt.xlabel('s [m]')
        plt.ylabel(r'$\Delta D_x$')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(tw_sliced.s, tw_sliced.dx - tw_normal.dx - Ddx_model_1, label='Xsuite - Model 1', color='tab:blue')
        plt.xlabel('s [m]')
        plt.ylabel(r'$\Delta D_x$')
        plt.legend()
        plt.show()

        Ddx_matrix_model = Ddx_matrix(
            L = Ldip_scan, 
            s0 = start_dip + start_arc_s, 
            sr = sr,  
            bprime = bprime, 
            brho = brho,
            h = href
            )

        plt.figure()
        plt.plot(tw_sliced.s, tw_sliced.dx-tw_normal.dx, label='Xsuite', color='tab:blue')
        plt.scatter(Ldip_scan[::50], Ddx_matrix_model[::50], label='Model matrix', color='tab:red', marker='+')
        plt.xlabel('s [m]')
        plt.ylabel(r'$\Delta D_x$')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(tw_sliced.s, tw_sliced.dx-tw_normal.dx-Ddx_matrix_model, label='Xsuite - Model matrix')
        plt.xlabel('s [m]')
        plt.ylabel(r'$\Delta D_x$')
        plt.show()