# -*- coding: utf-8 -*-
"""

@author: Lisa Soubirou
"""
from typing import Union, Tuple
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ring_geometry.ring_geometry import RingGeometry, load_file

base_input_parameters = {"Beam stay clear in sigma" : 6,
                         "Beam energy spread" : 5e-3,
                         "Normalized transverse emittance [m]" : 25e-6,
                         "Number of cavities" : 380,
                         "Length of a cavity [m]" : 1.2474,
                         "Phase advance [rad]" : np.pi/2,
                         "Minimum quadrupole aperture (diam.) [m]" : 30e-3,
                         "Radial aperture margin [m]" : 10e-3,
                         "Quadrupole pole field [T]" : 1.0,
                         "Number of bloc in the dipole pattern" : 1
                        }

class LatticeCell(RingGeometry):
    """ The Cell class aims at findind the optimum cell length of a rcs by
    minimizing the quad length. 
    """

    def __init__(self, file_input_cell: str = None,
                 file_input_geo: str = None) -> None:
        """Initialize the cell class. 

        Parameters:
            
            n_sigma : float : 
                beam stay clear in number of sigma
            sigma_delta : float 
                Max beam energy spread
            emitt_norm :float
                Transverse normalized emittance [m]
            n_cav : int
                Total number of cavities
            l_cav : float
                Length of a cavity [m]
            mu : float 
                Phase advance per cell [rad]
            apert_qp_min : float
                Minimum QP aperture to be chosen [m]
            dx : float
                Margin radius taken when computing the magnet aperture [m]
            Bpole : float
                Maximum field on the quadrupole pole [T]
            nb_bloc : int
                Number of bloc in the dipole pattern
        """
        super().__init__(file_input_geo)

        self.input_geo = load_file(file_input_geo)

        if file_input_cell is None:
            self.n_sigma = base_input_parameters["Beam stay clear in sigma"]
            self.sigma_delta = base_input_parameters["Beam energy spread"]
            self.emitt_norm = base_input_parameters["Normalized transverse emittance [m]"]
            self.n_cav = base_input_parameters["Number of cavities"]
            self.l_cav = base_input_parameters["Length of a cavity [m]"]
            self.mu = base_input_parameters["Phase advance [rad]"]
            self.apert_qp_min = base_input_parameters["Minimum quadrupole aperture (diam.) [m]"]
            self.dx = base_input_parameters["Radial aperture margin [m]"]
            self.Bpole = base_input_parameters["Quadrupole pole field [T]"]
            self.nb_bloc = base_input_parameters["Number of bloc in the dipole pattern"]
        else:
            input_cell = load_file(file_input_cell)
            for key, val in input_cell.items():
                setattr(self, key, val)

        self.check_inputs()
        l_cav_tot = self.n_cav * self.l_cav
        self.filling_ratio = 1 - l_cav_tot / self.circumference
        self.tr_emittance_inj = self.emitt_norm / self.inj_gamma  # Transverse emittance at injection
        l_cell_scan = np.linspace(10, self.arc_length, 30)
        self.lquad_compute(l_cell_scan)

    def reinit_cell(self) -> None:
        """Reinitialize both RingGeometry and Cell-specific attributes"""
        self.reinit_all()
        if self.type == 'hybrid':
            if self.nb_bloc == 1:
                compute_apert = self.compute_apert_1bloc
            elif self.nb_bloc == 2:
                compute_apert = self.compute_apert_2bloc
            self.width_nc_opt, self.width_sc_opt, self.aperture_nc_opt, self.aperture_sc_opt = compute_apert(self)

    def check_inputs(self) -> None:
        """
        Check if input parameters are physical. 

        Raises:
            ValueError:  If one of the limits of the parameters is not met.
        """
        if self.n_sigma <= 0.:
            raise ValueError('Number of sigma for beam size n_sigma must be positive')
        if self.sigma_delta <= 0.:
            raise ValueError(' Dispersion in energy of the beam sigma_delta must be positive')
        if self.emitt_norm <= 0.:
            raise ValueError('Normalized transverse emittance emitt_norm must be positive')
        if self.n_cav <= 0.:
            raise ValueError('Number of cavities n_cav must be positive')
        if self.l_cav <= 0.:
            raise ValueError('Length of cavities l_cav must be positive')
        if self.apert_qp_min <= 0.:
            raise ValueError('Minimal aperture of QP apert_qp_min must be positive')
        if self.dx <= 0.:
            raise ValueError('Extra aperture for magnets dx must be positive')
        if self.Bpole <= 0. or self.Bpole >= 1.2:
            raise ValueError('Field at the pole of QP Bpole must be positive and inferior to 1.2T')

    def compute_beam_size(self, lc: Union[np.ndarray, float]) -> np.ndarray:
        """
        Compute the beam size from the twiss function and input parameters for a given cell
        length.
        Args:
            lc (np.ndarray): cell length [m]
        Returns:
            np.ndarray: beam size radius [m]. Same dimension as lc. 
        """
        beta_max = lc * (1 + np.sin(self.mu / 2)) / np.sin(self.mu)
        d_max = 2 * np.pi * lc**2 / self.tot_arc_length * (1 + 1/2 * np.sin(self.mu / 2)) /(
            4 * (np.sin(self.mu / 2))**2)
        beam_size = self.sigma_delta * d_max + self.n_sigma * np.sqrt(self.tr_emittance_inj * beta_max)
        return beam_size

    def rotate_coord(self, rcs: RingGeometry, n_bloc: int, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """When a cell is made of multiple blocs of a given pattern, the trajectory has to be 
        corrected with reference to the axis of a single bloc.
        Args:
            rcs (RingGeometry): RingGeometry object
            n_bloc (int): number of bloc
            t (float): time [normalized]
        Returns:
            np.ndarray, np.ndarray: x [m] and y [m] coordinate after rotation in 1 bloc.
        """
        theta = rcs.cell_angle / 4 / self.nb_bloc
        coor_init = rcs.zn(t)[1:(1+n_bloc*(self.nb_sub + 1))]
        coor = coor_init - coor_init[0]
        x, y = np.real(coor), np.imag(coor)
        x_rotated = x * np.cos(theta) + y * np.sin(theta)
        y_rotated = -x * np.sin(theta) + y * np.cos(theta)
        return x_rotated, y_rotated

    def compute_apert_1bloc(self, rcs: RingGeometry) -> Tuple[float, float, float, float]:
        """Compute the excursion and aperture in the nc and sc dipoles for a 
            1 bloc pattern.
        Args:
            rcs (RingGeometry): RingGeometry object
        Returns:
            Tuple[float, float, float, float]: excursion in nc [m], excursion in sc [m], 
                aperture in nc [m], aperture in sc [m] 
        """
        beam_size = self.compute_beam_size(rcs.cell_length)
        mask_nc = [item == 'BNC' for item in rcs.pattern]
        mask_sc = [item == 'BSC' for item in rcs.pattern]
        max_width_dip = rcs.extrema(rcs.t_ext)[1] - rcs.extrema(rcs.t_inj)[0]
        width_sc = np.max(max_width_dip[mask_sc])
        width_nc = np.max(max_width_dip[mask_nc])
        aperture_sc = width_sc + (beam_size + self.dx) * 2
        aperture_nc = width_nc + (beam_size + self.dx) * 2

        return (width_nc, width_sc, aperture_nc, aperture_sc)

    def compute_apert_2bloc(self, rcs: RingGeometry) -> Tuple[float, float, float, float]:
        """Compute the excursion and aperture in the nc and sc dipoles for a 
            2 bloc pattern.
        Args:
            rcs (RingGeometry): RingGeometry object
        Returns:
            Tuple[float, float, float, float]: excursion in nc [m], excursion in sc [m], 
                aperture in nc [m], aperture in sc [m]
        """
        beam_size = self.compute_beam_size(rcs.cell_length)
        nd = len(rcs.pattern)
        n_bloc = len(rcs.pattern) // 2
        n_point = self.nb_sub + 1
        y_rot_inj = self.rotate_coord(rcs, n_bloc, 0)[1]
        y_rot_ext = self.rotate_coord(rcs, n_bloc, 1)[1]
        y_max = np.array([np.max(y_rot_ext[_id*n_point:(_id+1)*n_point]) for _id in range(nd//2)])
        y_min = np.array([np.min(y_rot_inj[_id*n_point:(_id+1)*n_point]) for _id in range(nd//2)])
        max_width_dip = y_max - y_min
        mask_nc = [item == 'BNC' for item in rcs.pattern[0:n_bloc]]
        mask_sc = [item == 'BSC' for item in rcs.pattern[0:n_bloc]]
        width_sc = np.max(max_width_dip[mask_sc])
        width_nc = np.max(max_width_dip[mask_nc])
        aperture_sc = width_sc + (beam_size + self.dx) * 2
        aperture_nc = width_nc + (beam_size + self.dx) * 2

        return (width_nc, width_sc, aperture_nc, aperture_sc)

    def plot_2bloc_traj(self) -> None:
        """In the case of a 2 bloc pattern, this function plots the trajectory
        in a single bloc of the pattern. All dipoles of the bloc are on the same 
        mechanical axis, which was rotated to be horizontal.
        """
        n_bloc = len(self.pattern) // 2
        plt.figure()
        for i in np.linspace(0, 1, 8):
            x_rot, y_rot = self.rotate_coord(self, n_bloc, i)
            plt.plot(x_rot, y_rot)
        for idx, (z1, z2) in enumerate(zip(self.z_begin(1)[0:n_bloc], self.z_end(1)[0:n_bloc])):
            x1, x2 = np.real(z1), np.real(z2)
            plt.axvline(x1, ls='--', color='k')
            plt.axvline(x2, ls='--', color='k')
            midpoint = (x1 + x2) / 2
            dipole_type = self.pattern[idx][1:]
            plt.text(midpoint, max(y_rot) * 1.07, dipole_type, ha='center', va='bottom', fontsize=12)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.show()

    def lquad_compute(self, lc: np.ndarray) -> None :
        """Method that computes the possible values of quad length and choses the optimum.

        Args:
            lc (np.ndarray): cell length [m]
        """
        size = self.compute_beam_size(lc) + self.dx
        size[size < self.apert_qp_min / 2] = self.apert_qp_min / 2
        lquad = 4 * np.sin(self.mu / 2) / lc * self.ext_Brho / self.Bpole * size
        self.lquad_max = (self.arc_length - self.dipole_length_tot / self.nb_arc - 2 * (self.nd + 1) *
                          self.dipole_spacing * np.floor(self.arc_length / lc)) / 2 / np.floor(self.arc_length / lc)
        mask = self.lquad_max - lquad > 0
        self.lquad_sol, self.size_beam_sol = lquad[mask], size[mask]
        self.lcell_sol = lc[mask]
        self.lquad_min = min(self.lquad_sol)
        self.nc_sol = np.floor(self.arc_length / lc).astype(int)[mask]
        nc_inter = np.floor(self.arc_length / self.lcell_sol[np.argmin(self.lquad_sol)]) + 1
        rcs_inter = RingGeometry(self.input_geo, filling_ratio=self.filling_ratio, length_short_insertion=self.lquad_min,
                             nb_cell_arc=nc_inter)
        self.lquad_opt = self.lquad_sol[np.argmin(np.absolute(self.lcell_sol - rcs_inter.cell_length))]
        self.rcs_opt = RingGeometry(self.input_geo, filling_ratio=self.filling_ratio, length_short_insertion=self.lquad_opt,
                                nb_cell_arc=nc_inter)
        self.nc_opt = self.rcs_opt.nb_cell_arc
        self.lcell_opt = self.rcs_opt.cell_length
        self.length_short_insertion = self.lquad_opt
        self.nb_cell_arc = self.rcs_opt.nb_cell_arc

        if self.type == 'hybrid':
            if self.nb_bloc == 1:
                print('Apertures computed for 1 bloc of dipole')
                compute_apert = self.compute_apert_1bloc
            elif self.nb_bloc == 2:
                print('Apertures computed for 2 blocs of dipole')
                compute_apert = self.compute_apert_2bloc
            self.width_nc_opt, self.width_sc_opt, self.aperture_nc_opt, self.aperture_sc_opt = compute_apert(self)
            cell_length_test = np.zeros(len(self.lquad_sol))
            nc_max = np.zeros(len(self.lquad_sol))
            width_dip = np.zeros([len(self.lquad_sol), 4])
            for ic, (lqp, nb_c) in enumerate(zip(self.lquad_sol, self.nc_sol)):
                rcs_scan = RingGeometry(self.input_geo, filling_ratio=self.filling_ratio,
                                    length_short_insertion=lqp, nb_cell_arc=nb_c)
                nc_max[ic] = rcs_scan.nb_cell_arc
                cell_length_test[ic] = float(rcs_scan.cell_length)
                width_dip[ic, 0], width_dip[ic, 1], width_dip[ic, 2], width_dip[ic, 3] = compute_apert(rcs_scan)
            self.lcell_geo = cell_length_test
            self.width_nc_geo, self.width_sc_geo = width_dip[:, 0], width_dip[:, 1]
            self.apert_nc_geo, self.apert_sc_geo = width_dip[:, 2], width_dip[:, 3]

    @property
    def beta_max(self) -> float:
        """Max of beta function for the optimized solution [m]"""
        return self.lcell_opt * (1 + np.sin(self.mu / 2)) / np.sin(self.mu)

    @property
    def disp_max(self) -> float:
        """Max of dispersion function for the optimized solution [m]"""
        return 2 * np.pi * self.lcell_opt**2 / self.tot_arc_length * (1 + 1/2 * np.sin(self.mu / 2))/(
            4 * (np.sin(self.mu / 2))**2)

    @property
    def beam_size(self) -> float:
        """Beam size for the optimized solution [m]"""
        return (self.n_sigma * np.sqrt(self.beta_max * self.tr_emittance_inj) + 
                self.sigma_delta * self.disp_max) *2

    @property
    def apert_diam(self) -> float:
        """Aperture diameter for the QP in the optimized solution [m]"""
        return self.beam_size + self.dx * 2

    @property
    def mcf(self) -> float:
        """Momentum compaction factor for the optimized solution"""
        return (self.cell_angle / 2)**2 *(1 / (np.sin(self.mu / 2))**2 - 1 / 12) * self.filling_ratio
