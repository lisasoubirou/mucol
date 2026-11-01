import numpy as np
import matplotlib.pyplot as plt
from class_geometry.ring_geometry.ring_geometry import RingGeometry
import scipy.constants as sc
import xtrack as xt
import xpart as xp
import copy
from scipy.interpolate import make_smoothing_spline
from EB_mismatch.EBModelOptics import EBModelDip, EBModelQuad
from pathlib import Path

class EB_error():

    def __init__(self, 
        line : xt.Line, 
        rcs : RingGeometry, 
        t_acc : float, 
        n_turn_rcs : int, 
        n_slice : int = 10,
        twiss_method : str = '6d',
        method : str = 'var_ref',
        time_traj_ref : float = 0.5,
        s_jitter : float = 0,
        jitter_diagonal : bool = False,
        RF_tuning : bool = False,
        ) -> None :
        """Class studying the effect of the mismatch between energy and bending field.

        Args:
            line (xt.Line): Line of the reference arc. 
            rcs (RingGeometry): RingGeometry object.
            t_acc (float): Acceleration time [s].
            n_turn_rcs (int): Number of turns in the RCS.
            n_slice (int, optional): Number of slices for pulsed elements. Defaults to 10.
            method (str, optional): Method to apply on dipoles: 'var_ref' or 'var_k'. Defaults to 'var_ref'.
        """

        self.line_arc = line    # Reference arc
        self.tw_arc = line.twiss(
            matrix_stability_tol=0.1, 
            compute_chromatic_properties=True,
            method=twiss_method)   # Twiss of the arc
        
        self.tab_arc = self.line_arc.get_table()    # Table of the line arc
        self.tw_init = self.tw_arc.get_twiss_init(0)    # Init twiss at beginning of the arc
        self.twiss_method = twiss_method
        self.length_arc = self.line_arc.get_length()    # Length of one arc [m]
        
        self.rcs = rcs  # RingGeometry object
        self.nb_arc = self.rcs.nb_arc   # Number of arcs
        self.t_acc = t_acc  # Acceleration time [s]
        self.n_turn_rcs = n_turn_rcs    # Number of turns 
        self.length_arc = rcs.circumference / rcs.nb_arc

        self.n_slice = n_slice  # Number of slice for pulsed elements
        self.method = method # Method to apply on dipole: var_ref or var_k
        self.time_traj_ref = time_traj_ref # Reference on trajectory when 'var_k' is used

        self.s_jitter = s_jitter    #
        self.jitter_diagonal = jitter_diagonal

        self.RF_tuning = RF_tuning

        if self.jitter_diagonal:
            self.sr = self.length_arc / 2   # Reference s for which B(s) = B* [m]
            # self.tshift = self.sr / sc.c / self.t_acc    # Time shift to center B on sr [s]
        else:    
            self.sr = self.length_arc / 2 + s_jitter   # Reference s for which B(s) = B* [m]
            # self.tshift = self.sr / sc.c / self.t_acc    # Time shift to center B on sr [s]

        self.bdot = (rcs.dipole_families['BNC']['B'](1) - rcs.dipole_families['BNC']['B'](0)) / self.t_acc  # Field ramp [T/s]
        self.bprime = self.bdot / sc.c   # Longitudinal gradient [T/m]

        self.t_beg_arc = self.length_arc * np.arange(0, self.n_turn_rcs*self.nb_arc) / sc.c / self.t_acc # Time at the beginning of each arc (normalized)

        self.ramping_option = {
            'ref' : self.build_arc_ref,
            'sliced' : self.build_arc_sliced, 
            'pert' : self.build_arc_pert, 
            'sliced_all' : self.build_arc_sliced_all,
            'meanquad' : self.build_arc_meanquad,
            'Rmatrix' : self.build_arc_RMatrix,
            'Rmatrix_all' : self.build_arc_RMatrix_all,
            } 
        
    def error_dx (self, s0 : float, L : float, t_ref : float) -> float:
        """ Computes the trajectory deviation created in a dipole due to energy - field mismatch.

        Args:
            s0 (float): Longitunial position of the start of the dipole [m].
            L (float): Length of the dipole [m].
            t_ref (float): Time reference of the energy. Normalized between [0,1].

        Returns:
            float
        """
        return 1/2 / self.rcs.func_Brho(t_ref) * L**2 * self.bprime * (s0 + L / 3 - self.sr)

    def error_dpx (self, s0 : float, L : float, t_ref : float) -> float:
        """Computes the x' created in a dipole due to energy - field mismatch.

        Args:
            s0 (float): Longitunial position of the start of the dipole [m].
            L (float): Length of the dipole [m].
            t_ref (float): Time reference of the energy. Normalized between [0,1].

        Returns:
            float
        """
        return L / self.rcs.func_Brho(t_ref) * self.bprime * (L / 2 + s0 - self.sr)


    def _build_arc_generic(self, t_ref : float, i_arc : int, handle_quad=None, handle_bend=None):
        """Generic method to build the arcs.

        Args:
            t_ref (float): Time reference of the energy. Normalized between [0,1].
            i_arc (int): Arc number.
            handle_quad (_type_, optional): Method to handle quad. Defaults to None (element is copied as such).
            handle_bend (_type_, optional): Method to handle bend. Defaults to None (element is copied as such).

        Returns:
            _type_: _description_
        """
        element_list = []
        name_list = []

        el_count = {etype: 0 for etype in ['Drift', 'Marker', 'ReferenceEnergyIncrease', 'Cavity', 'Quadrupole', 'Bend', 'LineSegmentMap', 'LongitudinalLimitRect', 'Multipole']}

        for el, name, el_type in zip(self.line_arc.elements, self.tab_arc.name, self.tab_arc.element_type):
            el_count[el_type] += 1

            if el_type == 'Quadrupole' and handle_quad:
                handle_quad(element_list, name_list, el, el_type, name, el_count[el_type], t_ref, i_arc)
            elif el_type == 'Bend' and 'BNC' in name and handle_bend:
                handle_bend(element_list, name_list, el, el_type, name, el_count[el_type], t_ref, i_arc)
            elif el_type == 'Bend': # and ('BSC' in name or 'BNC' in name):
                i_dip = int(name[3:4])
                if self.method == 'var_ref':
                    element_list.append(xt.Bend(
                        k0 = self.rcs.dip_bending(t_ref)[i_dip],
                        # h = self.rcs.dip_bending(t_ref)[i_dip],
                        angle = el.length * self.rcs.dip_bending(t_ref)[i_dip],
                        length = el.length, 
                        edge_entry_angle = self.rcs.epsilon(t_ref)[i_dip],
                        edge_exit_angle = - self.rcs.epsilon(t_ref)[i_dip+1]
                        ))

                elif self.method == 'var_k':
                    element_list.append(xt.Bend(
                        k0 = self.rcs.dip_bending(t_ref)[i_dip],
                        # h = self.rcs.dip_bending(self.time_traj_ref)[i_dip], 
                        angle = el.length * self.rcs.dip_bending(self.time_traj_ref)[i_dip],
                        length = el.length, 
                        edge_entry_angle = self.rcs.epsilon(self.time_traj_ref)[i_dip],
                        edge_exit_angle = - self.rcs.epsilon(self.time_traj_ref)[i_dip + 1]
                        ))
                else:
                    raise ValueError(f'Method {self.method} is not supported: try var_ref or var_k')

                name_list.append(f'{el_type}_{el_count[el_type]}_arc_{i_arc}')

            elif el_type == 'Cavity' and self.RF_tuning :
                dz = 2 * self.rcs.nb_cell_arc * (self.rcs.path_length(t_ref) - self.rcs.path_length(self.time_traj_ref))
                dz_element = xt.ZetaShift(dzeta = -dz / 2)
                element_list.append(dz_element)
                name_list.append(f'ZetaShift_{el_count[el_type]}_arc_{i_arc}')

                element_list.append(el.copy())
                name_list.append(f'{el_type}_{el_count[el_type]}_arc_{i_arc}')
            
            elif el_type in el_count:
                element_list.append(el.copy())
                name_list.append(f'{el_type}_{el_count[el_type]}_arc_{i_arc}')
            else:
                print(f'type of element {el_type} not supported!')

        return element_list, name_list
    
    def _handle_bend_sliced(self, el_list, name_list, el, el_type, name, count, t_ref, i_arc):
        """Dipoles are sliced and pulsed."""

        L_dip = el.length
        start_dip = self.tab_arc['s', name]
        start_arc_s = self.length_arc * i_arc

        tshift =  self.sr / sc.c / self.t_acc
        if self.jitter_diagonal:
            tshift += - (2 * t_ref - 1) * self.s_jitter / sc.c / self.t_acc

        l_slice = L_dip / self.n_slice
        s_slice = start_arc_s + start_dip + np.linspace(0, L_dip, self.n_slice+1) - l_slice / 2
        t_slice = s_slice / sc.c / self.t_acc - tshift

        B_slice = self.rcs.dipole_families['BNC']['B'](t_slice)
        energy_ref = self.rcs.injection_energy + (self.rcs.extraction_energy - self.rcs.injection_energy) * t_ref
        rho_ref = energy_ref / sc.c / self.rcs.dipole_families['BNC']['B'](t_ref)
        rho_slice = energy_ref / sc.c / B_slice

        i_dip = int(name[3:4])
        
        if self.method == 'var_ref':
            rho_ref = 1 / self.rcs.dip_bending(t_ref)[i_dip]
            for i in range(1, self.n_slice + 1):
                # el_list.append(xt.Bend(k0=1 / rho_slice[i], h=1 / rho_ref, length=l_slice))
                if i == 1:
                    el_list.append(xt.Bend(
                        k0 = 1 / rho_slice[i], #1 / rho_ref
                        # h = 1 / rho_ref,
                        angle = l_slice / rho_ref,
                        length = l_slice, 
                        edge_entry_angle = self.rcs.epsilon(t_slice[i])[i_dip] #self.rcs.epsilon(t_ref)[i_dip]
                        ))
                elif i == self.n_slice:
                    el_list.append(xt.Bend(
                        k0 = 1 / rho_slice[i], #1 / rho_ref
                        # h = 1 / rho_ref, 
                        angle = l_slice / rho_ref,
                        length = l_slice, 
                        edge_exit_angle = -self.rcs.epsilon(t_slice[i])[i_dip+1] #self.rcs.epsilon(t_ref)[i_dip]
                        ))
                else:
                    el_list.append(xt.Bend(
                        k0 = 1 / rho_slice[i], #1 / rho_ref
                        # h = 1 / rho_ref,
                        angle = l_slice / rho_ref,
                        length = l_slice
                        ))
                    # el_list.append(xt.Bend(
                    #                     k0 = 1 / rho_ref, 
                    #                     h = 1 / rho_ref, 
                    #                     length = l_slice))
                name_list.append(f'{el_type}_{count}_slice_{i}_arc_{i_arc}')
        
        elif self.method == 'var_k':

            rho_ref = 1 / self.rcs.dip_bending(self.time_traj_ref)[i_dip]
            epsilon_ref = self.rcs.epsilon(self.time_traj_ref)

            for i in range(1, self.n_slice + 1):
                # el_list.append(xt.Bend(k0=1 / rho_slice[i], h=1 / rho_ref, length=l_slice))
                if i == 1:
                    el_list.append(xt.Bend(
                        k0 = 1 / rho_slice[i], #1 / rho_ref
                        # h = 1 / rho_ref,
                        angle = l_slice / rho_ref,
                        length = l_slice,
                        edge_entry_angle = epsilon_ref[i_dip] #self.rcs.epsilon(t_ref)[i_dip]
                        ))
                elif i == self.n_slice:
                    el_list.append(xt.Bend(
                        k0 = 1 / rho_slice[i], #1 / rho_ref
                        # h = 1 / rho_ref,
                        angle = l_slice / rho_ref,
                        length = l_slice,
                        edge_exit_angle = - epsilon_ref[i_dip + 1] #self.rcs.epsilon(t_ref)[i_dip]
                        ))
                else:
                    el_list.append(xt.Bend(
                        k0 = 1 / rho_slice[i], #1 / rho_ref
                        # h = 1 / rho_ref,
                        angle = l_slice / rho_ref,
                        length = l_slice
                        ))
                # el_list.append(xt.Bend(
                #                     k0 = 1 / rho_ref, 
                #                     h = 1 / rho_ref, 
                #                     length = l_slice))
                name_list.append(f'{el_type}_{count}_slice_{i}_arc_{i_arc}')
        else:
            raise ValueError(f'Method {self.method} is not supported: try var_ref or var_k')
        
    
    def _handle_bend_traj(self, el_list, name_list, el, el_type, name, count, t_ref, i_arc):
        """Reference trajectory is perturbed after the dipoles."""

        L_dip = el.length
        start_dip = self.tab_arc['s', name]

        el_list.append(el.copy())
        name_list.append(f'{el_type}_{str(count)}_arc_{i_arc}')

        error_x = self.error_dx(s0=start_dip, L=L_dip, t_ref=t_ref)
        error_px = self.error_dpx(s0=start_dip, L=L_dip, t_ref=t_ref) * 180 / np.pi # In deg

        el_list.append(xt.XYShift(dx=error_x))
        el_list.append(xt.YRotation(angle=error_px))
        name_list.append(f'x_shift_{count}_arc_{i_arc}')
        name_list.append(f'px_shift_{count}_arc_{i_arc}')

    def _handle_bend_Rmatrix(self, el_list, name_list, el, el_type, name, count, t_ref, i_arc):
        """A matrix dR is added after the dipole to represent the error."""

        L_dip = el.length
        start_dip = self.tab_arc['s', name]

        iBNC = np.where(['BNC' in p for p in self.rcs.pattern])[0][0]
        h_ref = self.rcs.dip_bending(t_ref)[iBNC]
        brho_ref = self.rcs.func_Brho(t_ref)
        energy_ref = self.rcs.injection_energy + (self.rcs.extraction_energy - self.rcs.injection_energy) * t_ref
        gamma0_ref = energy_ref / self.rcs.muon_mass

        name_list.append(f'{el_type}_{str(count)}_arc_{i_arc}')
        i_dip = int(name[3:4])

        if self.method == 'var_ref':
            el_list.append(xt.Bend(
                k0 = self.rcs.dip_bending(t_ref)[i_dip],
                # h = self.rcs.dip_bending(t_ref)[i_dip],
                length = el.length,
                angle = el.length * self.rcs.dip_bending(t_ref)[i_dip],
                edge_entry_angle = self.rcs.epsilon(t_ref)[i_dip],
                edge_exit_angle = - self.rcs.epsilon(t_ref)[i_dip+1]
                ))

        elif self.method == 'var_k':
            el_list.append(xt.Bend(
                k0 = self.rcs.dip_bending(t_ref)[i_dip],
                # h = self.rcs.dip_bending(self.time_traj_ref)[i_dip], 
                length = el.length,
                angle =  el.length * self.rcs.dip_bending(self.time_traj_ref)[i_dip],
                edge_entry_angle = self.rcs.epsilon(self.time_traj_ref)[i_dip],
                edge_exit_angle = - self.rcs.epsilon(self.time_traj_ref)[i_dip + 1]
                ))
    
        else:
            raise ValueError(f'Method {self.method} is not supported: try var_ref or var_k')

        if self.jitter_diagonal:
            sjit_add = - (2 * t_ref - 1) * self.s_jitter
        else:
            sjit_add = 0

        dip_model = EBModelDip(
            h = h_ref,
            L = L_dip,
            s0 = start_dip,
            sr = self.sr + sjit_add,
            bprime = self.bprime,
            brho = brho_ref,
            gamma0 = gamma0_ref
            )
        
        dX = dip_model.dX
        dM = dip_model.dM_error
    
        dR_matrix_el = xt.FirstOrderTaylorMap(length=0, m0=dX, m1=dM)

        el_list.append(dR_matrix_el)
        name_list.append(f'dRMatrix_dip_{count}_arc_{i_arc}')

    def _handle_quad_RMatrix(self, el_list, name_list, el, el_type, name, count, t_ref, i_arc):
        """A matrix dR is added after the quadrupole to represent the error."""

        gradient_inj = el.k1 * self.rcs.func_Brho(0)
        gradient_ext = el.k1 * self.rcs.func_Brho(1)
        Gprime = (gradient_ext - gradient_inj) / self.t_acc / sc.c

        gradient_ref = gradient_inj + (gradient_ext - gradient_inj) * t_ref
        brho_ref = self.rcs.func_Brho(t_ref)
        kref = gradient_ref / brho_ref
        energy_ref = self.rcs.injection_energy + (self.rcs.extraction_energy - self.rcs.injection_energy) * t_ref
        gamma0_ref = energy_ref / self.rcs.muon_mass

        L_quad = el.length
        start = self.tab_arc['s', name]

        if self.jitter_diagonal:
            sjit_add = - (2 * t_ref - 1) * self.s_jitter
        else:
            sjit_add = 0

        quad_model = EBModelQuad(
            kref = kref,
            L = L_quad,
            s0 = start,
            sr = self.sr + sjit_add,
            Gprime = Gprime,
            brho = brho_ref,
            gamma0 = gamma0_ref)

        dX = quad_model.dX
        dM = quad_model.dM_error
        dR_matrix_el = xt.FirstOrderTaylorMap(length=0, m0=dX, m1=dM)

        el_list.append(el.copy())
        name_list.append(f'{el_type}_{str(count)}_arc_{i_arc}')
        el_list.append(dR_matrix_el)
        name_list.append(f'dRMatrix_quad_{count}_arc_{i_arc}')

    def _handle_quad_sliced(self, el_list, name_list, el, el_type, name, count, t_ref, i_arc):
        """Quads are sliced and pulsed."""

        gradient_inj = el.k1 * self.rcs.func_Brho(0)
        gradient_ext = el.k1 * self.rcs.func_Brho(1)

        tshift =  self.sr / sc.c / self.t_acc
        if self.jitter_diagonal:
            tshift += - (2 * t_ref - 1) * self.s_jitter / sc.c / self.t_acc

        L_quad = el.length
        start_dip = self.tab_arc['s', name]
        start_arc_s = self.length_arc * i_arc
        l_slice = L_quad / self.n_slice
        s_slice = start_arc_s + start_dip + np.linspace(0, L_quad, self.n_slice+1) - l_slice / 2
        t_slice = s_slice / sc.c / self.t_acc - tshift

        gradient_pulsed = gradient_inj + (gradient_ext - gradient_inj) * t_slice
        k_pulsed = gradient_pulsed / self.rcs.func_Brho(t_ref)

        for i in range(1, self.n_slice + 1):
            el_list.append(xt.Quadrupole(k1=k_pulsed[i], length=l_slice))
            name_list.append(f'Quadrupole_{count}_slice_{i}_arc_{i_arc}')

    def _handle_quad_meanquad(self, el_list, name_list, el, el_type, name, count, t_ref, i_arc):
        """Quads are averaged at the center field value"""
        gradient_inj = el.k1 * self.rcs.func_Brho(0)
        gradient_ext = el.k1 * self.rcs.func_Brho(1)

        tshift =  self.sr / sc.c / self.t_acc
        if self.jitter_diagonal:
            tshift += - (2 * t_ref - 1) * self.s_jitter / sc.c

        L_quad = el.length
        start_dip = self.tab_arc['s', name]
        start_arc_s = self.length_arc * i_arc
        s_mid = start_arc_s + start_dip + L_quad / 2
        t_mid = s_mid / sc.c / self.t_acc - tshift

        gradient_mid = gradient_inj + (gradient_ext - gradient_inj) * t_mid
        k_mid = gradient_mid / self.rcs.func_Brho(t_ref)

        el_list.append(xt.Quadrupole(k1=k_mid, length=L_quad))
        name_list.append(f'Quadrupole_{count}_arc_{i_arc}')

    def build_arc_ref(self, t_ref : float, i_arc : int):
        """Build reference arc with no errors. 
        Returns list of elements and element names.

        Args:
            t_ref (float): Time reference of the energy. Normalized between [0,1].
            i_arc (int): Arc number.

        Returns:
            _type_: list of elements, list of element_names
        """
        return self._build_arc_generic(
            t_ref, i_arc
            )

    def build_arc_sliced(self, t_ref : float, i_arc : int):
        """Build arc where dipoles are sliced and pulsed. 
        Returns list of elements and element names.

        Args:
            t_ref (float): Time reference of the energy. Normalized between [0,1].
            i_arc (int): Arc number.

        Returns:
            _type_: list of elements, list of element_names
        """
        return self._build_arc_generic(
            t_ref, i_arc, 
            handle_bend=self._handle_bend_sliced
            )

    def build_arc_pert(self, t_ref : float, i_arc : int):
        """Build arc where the error on the dipoles is added as a perturbation on the reference trajectory. 
        Returns list of elements and element names.

        Args:
            t_ref (float): Time reference of the energy. Normalized between [0,1].
            i_arc (int): Arc number.

        Returns:
            _type_: list of elements, list of element_names
        """
        return self._build_arc_generic(
            t_ref, i_arc, 
            handle_bend=self._handle_bend_traj
            )
    
    def build_arc_RMatrix(self, t_ref : float, i_arc : int):
        """Build arc where the error on the dipoles is added as a map placed after the dipole.  
        Returns list of elements and element names.

        Args:
            t_ref (float): Time reference of the energy. Normalized between [0,1].
            i_arc (int): Arc number.

        Returns:
            _type_: list of elements, list of element_names
        """
        return self._build_arc_generic(
            t_ref, i_arc, 
            handle_bend=self._handle_bend_Rmatrix
            )

    def build_arc_RMatrix_all(self, t_ref : float, i_arc : int):
        """Build arc where the error on the dipoles is added as a map placed after the dipole.  
        Returns list of elements and element names.

        Args:
            t_ref (float): Time reference of the energy. Normalized between [0,1].
            i_arc (int): Arc number.

        Returns:
            _type_: list of elements, list of element_names
        """
        return self._build_arc_generic(
            t_ref, i_arc, 
            handle_bend=self._handle_bend_Rmatrix,
            handle_quad=self._handle_quad_RMatrix
            )

    def build_arc_sliced_all(self, t_ref : float, i_arc : int):
        """Build arc where the error on the dipoles is added as a perturbation on the reference trajectory.
        The quadrupoles are sliced and sliced. 
        Returns list of elements and element names.

        Args:
            t_ref (float): Time reference of the energy. Normalized between [0,1].
            i_arc (int): Arc number.

        Returns:
            _type_: list of elements, list of element_names
        """
        return self._build_arc_generic(
            t_ref, i_arc,
            handle_quad=self._handle_quad_sliced,
            handle_bend=self._handle_bend_sliced
        )

    def build_arc_meanquad(self, t_ref : float, i_arc : int):
        """Build arc where the error on the dipoles is added as a perturbation on the reference trajectory.
        The strength of the quadrupoles is averaged to the value at the center of the element.
        Returns list of elements and element names.

        Args:
            t_ref (float): Time reference of the energy. Normalized between [0,1].
            i_arc (int): Arc number.

        Returns:
            _type_: list of elements, list of element_names
        """
        return self._build_arc_generic(
            t_ref, i_arc,
            handle_quad=self._handle_quad_meanquad,
            handle_bend=self._handle_bend_traj
        )
    
    def _build_rcs(self, build_arc_func, arc_attr_name, twiss_attr_name, t_range=None):
        """Build the RCS line by concatenating the pulsed arcs. Set the lines, twiss and masks. 

        Args:
            build_arc_func (_type_): Method to be applied to pulse the elements.
            arc_attr_name (_type_): Name of the arc attribute.
            twiss_attr_name (_type_): Name of the twiss attribute.
            t_range (_type_, optional): Number of arcs on which we build and compute twiss. Defaults to None.
        """
        element_list = []
        name_list = []

        t_range = t_range or range(len(self.t_beg_arc))
        for i_arc in t_range:
            t_ref = self.t_beg_arc[i_arc]
            el, name = build_arc_func(t_ref, i_arc)
            element_list += el
            name_list += name

        arc_line = xt.Line(elements=element_list, element_names=name_list)
        arc_line.particle_ref = xp.Particles(p0c=self.rcs.injection_energy, q0=1, mass0=xp.MUON_MASS_EV)
        self.tw_init.element_name = arc_line.element_names[0]

        twiss = arc_line.twiss(
            start=arc_line.element_names[0],
            end=arc_line.element_names[-1],
            init=self.tw_init,
            method = self.twiss_method,
            compute_chromatic_properties = True
        )

        setattr(self, arc_attr_name, arc_line)
        setattr(self, twiss_attr_name, twiss)

        el_type = arc_line.get_table().element_type
        mask = (el_type != 'XYShift') & (el_type != 'YRotation')
        setattr(self, f"mask_{arc_attr_name.split('_')[1]}", mask)

    def build_rcs_ref(self, t_range=None):
        """Build the rcs line anc compute twiss.
        Reference rcs without errors.
        """
        self._build_rcs(self.build_arc_ref, 'rcs_ref', 'tw_rcs_ref', t_range=t_range)

    def build_rcs_sliced(self, t_range=None):
        """Build the rcs line anc compute twiss. 
        Dipoles are sliced and pulsed.
        """
        self._build_rcs(self.build_arc_sliced, 'rcs_sliced', 'tw_rcs_sliced', t_range=t_range)

    def build_rcs_pert(self, t_range=None):
        """Build the rcs line anc compute twiss. 
        Reference trajectory is perturbed after the dipoles.
        """
        self._build_rcs(self.build_arc_pert, 'rcs_pert', 'tw_rcs_pert', t_range=t_range)

    def build_rcs_RMatrix(self, t_range=None):
        """Build the rcs line anc compute twiss. 
        R matrix is added after the dipoles.
        """
        self._build_rcs(self.build_arc_RMatrix, 'rcs_RMatrix', 'tw_rcs_RMatrix', t_range=t_range)
    
    def build_rcs_RMatrix_all(self, t_range=None):
        """Build the rcs line anc compute twiss. 
        R matrix is added after the dipoles and quadrupoles.
        """
        self._build_rcs(self.build_arc_RMatrix_all, 'rcs_RMatrix_all', 'tw_rcs_RMatrix_all', t_range=t_range)

    def build_rcs_sliced_all(self, t_range=None):
        """Build the rcs line anc compute twiss. 
        Quads are dipoles are sliced and pulsed. 
        """
        self._build_rcs(self.build_arc_sliced_all, 'rcs_sliced_all', 'tw_rcs_sliced_all', t_range=t_range)

    def build_rcs_meanquad(self, t_range=None):
        """Build the rcs line anc compute twiss. 
        Reference trajectory is perturbed after the dipoles. Quads are averaged at the value at the center. 
        """
        self._build_rcs(self.build_arc_meanquad, 'rcs_meanquad', 'tw_rcs_meanquad', t_range=t_range)
    
    def build_first_arc(self, ramp_option : str, t_ref : float):
        """Build the first arc of the RCS.

        Args:
            ramp_option (str): Ramp option. To be chosen between : 'sliced','pert','sliced_all','meanquad'.
            t_ref (float): Time reference of the energy. Normalized between [0,1].

        Raises:
            ValueError: Raise an error if ramp_option is not one of: 'sliced','pert','sliced_all','meanquad'.

        Returns:
            _type_: line_arc, tw.
        """
        if ramp_option not in self.ramping_option.keys():
            raise ValueError(f"ramp_option {ramp_option} is not supported. Valid options are 'sliced','pert','sliced_all','meanquad'")
        
        build_arc = self.ramping_option[ramp_option]
        elements, element_names = build_arc(t_ref=t_ref, i_arc=0)
        
        energy_ref = self.rcs.injection_energy + (self.rcs.extraction_energy - self.rcs.injection_energy) * t_ref
        arc = xt.Line(elements=elements, element_names=element_names)
        arc.particle_ref = xp.Particles(p0c = energy_ref, q0 = 1, mass0 = xp.MUON_MASS_EV)
        
        self.tw_init.element_name = arc.element_names[0]
        tw = arc.twiss( start = arc.element_names[0],
                        end = arc.element_names[-1],
                        init = self.tw_init,
                        method = self.twiss_method)
        
        return arc, tw

def last_occur_mask(s):
    """Mask that keep only the last occurence of s coordinates. 

    Args:
        s (list): List of s coordinates.

    Returns:
        _type_: Mask.
    """
    seen = set()
    mask = [False] * len(s)

    # Traverse in reverse to keep the last occurrence
    for i in reversed(range(len(s))):
        if s[i] not in seen:
            seen.add(s[i])
            mask[i] = True

    return mask

def smooth_twiss(line : xt.Line, tw_init : xt.TwissInit, n_point : int =100):
    """Smooth twiss functions over n_point. 

    Args:
        line (xt.Line): Line object.
        tw_init (xt.TwissInit): Init twiss object.
        n_point (int, optional): Number of point to smooth the functions over. Defaults to 100.

    Returns:
        _type_ : 
        results_smooth: Dict with functions smoothed (Keys: 's', 'betx', 'bety', 'dx'). 
        tw_sliced :Raw twiss with sliced elements.
    """

    line_sliced = copy.deepcopy(line)
    line_sliced.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(5))])
    tw_init.element_name = line_sliced.element_names[0]
    tw_sliced = line_sliced.twiss( start = line_sliced.element_names[0],
                        end = line_sliced.element_names[-1],
                        init = tw_init,
                        method = '6d')
    mask = last_occur_mask(tw_sliced.s)

    new_s = np.linspace(tw_sliced.s[0], tw_sliced.s[-1], n_point)

    spl_betx = make_smoothing_spline(tw_sliced.s[mask], tw_sliced.betx[mask])
    spl_bety = make_smoothing_spline(tw_sliced.s[mask], tw_sliced.bety[mask])
    spl_dx = make_smoothing_spline(tw_sliced.s[mask], tw_sliced.dx[mask])

    results_smooth = {
        's' : new_s,
        'betx' : spl_betx(new_s),
        'bety' : spl_bety(new_s),
        'dx' : spl_dx(new_s)
    }

    return results_smooth, tw_sliced

def adjust_qx_arc(line : xt.Line, qx : float, qy : float, dqx : float, dqy : float) -> xt.Line:
    """Adjust the tunes of the line by addind a LineSegmentMap object.

    Args:
        line (xt.Line): Line to adjust.
        qx (float): Horizontal tune goal.
        qy (float): Vertical tune goal.

    Returns:
        xt.Line: Line with adjusted tunes.
    """
    tw = line.twiss(matrix_stability_tol=0.1)
    qx_init = tw.qx
    qy_init = tw.qy
    segmap = xt.LineSegmentMap(
        length = 0,
        qx = qx - qx_init,
        qy = qy - qy_init,
        dqx = dqx - tw.dqx,
        dqy = dqy - tw.dqy,
        betx = tw.betx[0],
        bety = tw.bety[0],
        alfx = tw.alfx[0],
        alfy = tw.alfy[0],
        dx = tw.dx[0],
        dpx = tw.dpx[0],
        x_ref = tw.x[0],
        px_ref = tw.px[0],
        y_ref = tw.y[0],
        py_ref = tw.py[0],
        longitudinal_mode = 'frozen'
        )
    
    line_adj = xt.Line(elements = [segmap] + [el for el in line.elements], 
                       element_names=['segment_map'] + [name for name in line.element_names])
    line_adj.particle_ref = line.particle_ref
    
    return line_adj


if __name__ == '__main__':

    path_lattices = Path(__file__).resolve().parent.parent

    # Initial lattice
    file_line = path_lattices / 'lattices/lattice_08_25/line_RCS1_gf_normal_qp_8_arc.json'

    arc = xt.Line.from_json(file_line)
    tab = arc.get_table()
    tw_noramp = arc.twiss(matrix_stability_tol=0.1)

    # Geometric parameters
    file_geo = path_lattices / 'lattices/lattice_08_25/para_RCS1_gf_normal_qp__8_arc.json'
    rcs = RingGeometry(file_geo)

    t_acc = 0.34*1e-3
    n_turn_rcs = 17
    n_slice = 10

    EB = EB_error(line=arc, rcs=rcs, t_acc=t_acc, n_turn_rcs=n_turn_rcs, n_slice=n_slice)

    arc1_sliced, tw_sliced = EB.build_first_arc('sliced', 0)

    arc1_pert, tw_pert = EB.build_first_arc('pert', 0)
    el_type = arc1_pert.get_table().element_type
    mask_pert = (el_type != 'XYShift') & (el_type != 'YRotation')

    arc1_sliced_all, tw_sliced_all = EB.build_first_arc('sliced_all', 0)
    el_type_sliced_all = arc1_sliced_all.get_table().element_type
    mask_wquad = (el_type_sliced_all != 'XYShift') & (el_type_sliced_all != 'YRotation')

    arc1_meanquad, tw_meanquad = EB.build_first_arc('meanquad', 0)

    plt.figure(figsize=(6.4,4.8))
    plt.plot(tw_sliced.s, tw_sliced.x * 1e3, label='Dip sliced', color='tab:blue')
    plt.scatter(tw_pert.s[mask_pert], tw_pert.x[mask_pert] * 1e3, label=r'Kick $ \Delta x + \Delta p_x$', marker='+', color='tab:red')
    plt.xlabel('s [m]')
    plt.ylabel('x [mm]')
    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6.4,4.8))
    plt.plot(tw_sliced.s, tw_sliced.px, label='Dip sliced', color='tab:blue')
    plt.scatter(tw_pert.s[mask_pert], tw_pert.px[mask_pert], label=r'Kick $ \Delta x + \Delta p_x$', marker='+', color='tab:red')
    plt.xlabel('s [m]')
    plt.ylabel("x'")
    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show()


    color_dip = 'tab:blue'
    color_quad = 'tab:orange'

    plt.figure(figsize=(6.4,4.8))
    plt.plot(tw_sliced.s, tw_sliced.x * 1e3, label='Dip pulsed', color=color_dip)
    plt.plot(tw_sliced_all.s[mask_wquad], tw_sliced_all.x[mask_wquad] * 1e3, label='Dip + Quad pulsed', color=color_quad)
    plt.xlabel('s [m]')
    plt.ylabel('x [mm]')
    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6.4,4.8))
    plt.plot(tw_sliced.s, tw_sliced.px, label='Dip pulsed', color=color_dip)
    plt.plot(tw_sliced_all.s[mask_wquad], tw_sliced_all.px[mask_wquad], label='Dip + Quad pulsed', color=color_quad)
    plt.xlabel('s [m]')
    plt.ylabel("x'")
    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show()

    tw_list = [tw_noramp, tw_sliced, tw_sliced_all]
    labels = ['No pulsed magnets', 'Dip pulsed', 'Dip + Quad pulsed']

    fig = plt.figure()
    fig.set_size_inches(10, 6)  # Adjust the width and height of the figure
    betx = plt.subplot(3, 1, 1)
    bety = plt.subplot(3, 1, 2, sharex=betx)
    disp = plt.subplot(3, 1, 3, sharex=betx)
    betx.plot(tw_noramp.s, tw_noramp.betx, linestyle='dashed', color='tab:blue', label=labels[0])
    betx.plot(tw_sliced.s, tw_sliced.betx, label=labels[1], color='tab:orange')
    betx.plot(tw_sliced_all.s, tw_sliced_all.betx, label=labels[2], color='tab:green')
    bety.plot(tw_noramp.s, tw_noramp.bety, color='tab:blue', linestyle='dashed')
    bety.plot(tw_sliced.s, tw_sliced.bety, color='tab:orange')
    bety.plot(tw_sliced_all.s, tw_sliced_all.bety, color='tab:green')
    disp.plot(tw_noramp.s, tw_noramp.dx, color='tab:blue', linestyle='dashed')
    disp.plot(tw_sliced.s, tw_sliced.dx, color='tab:orange')
    disp.plot(tw_sliced_all.s, tw_sliced_all.dx, color='tab:green')
    betx.set_ylabel(r'$\Delta \beta_{x} \ / \ \beta_{x}$ ')
    bety.set_ylabel(r'$\Delta \beta_{y} \ / \ \beta_{y}$')
    disp.set_ylabel(r'$\Delta D_{x}$ [m]')
    disp.set_xlabel(r"s [m]")
    fig.legend(loc='upper center', bbox_to_anchor=(0.48, 0.88))
    plt.subplots_adjust(right=0.8)
    plt.show()

    n_point_smooth = 100
    tw_smooth, tw_sliced = smooth_twiss(line=arc, tw_init=EB.tw_init, n_point=n_point_smooth)
    tw_sliced_smooth, tw_sliced_sliced = smooth_twiss(line=arc1_sliced, tw_init=EB.tw_init, n_point=n_point_smooth)
    tw_sliced_all_smooth, tw_wquad_sliced = smooth_twiss(line=arc1_sliced_all, tw_init=EB.tw_init, n_point=n_point_smooth)
    tw_meanquad_smooth, tw_meanquad_sliced = smooth_twiss(line=arc1_meanquad, tw_init=EB.tw_init, n_point=n_point_smooth)

    dbetax_sliced = (tw_sliced_smooth['betx'] - tw_smooth['betx']) / tw_smooth['betx']
    dbetay_sliced = (tw_sliced_smooth['bety'] - tw_smooth['bety']) / tw_smooth['bety']
    ddx_sliced = (tw_sliced_smooth['dx'] - tw_smooth['dx']) / np.sqrt(tw_smooth['betx'])

    dbetax_sliced_all = (tw_sliced_all_smooth['betx'] - tw_smooth['betx']) / tw_smooth['betx']
    dbetay_sliced_all = (tw_sliced_all_smooth['bety'] - tw_smooth['bety']) / tw_smooth['bety']
    ddx_sliced_all = (tw_sliced_all_smooth['dx'] - tw_smooth['dx']) / np.sqrt(tw_smooth['betx'])

    dbetax_meanquad = (tw_meanquad_smooth['betx'] - tw_smooth['betx']) / tw_smooth['betx']
    dbetay_meanquad = (tw_meanquad_smooth['bety'] - tw_smooth['bety']) / tw_smooth['bety']
    ddx_meanquad = (tw_meanquad_smooth['dx'] - tw_smooth['dx']) / np.sqrt(tw_smooth['betx'])

    fig = plt.figure()
    fig.suptitle('Dipole pulsed vs Dipole + Quadrupole pulsed', y=0.92)
    fig.set_size_inches(10, 8)  # Adjust the width and height of the figure
    betx = plt.subplot(4, 1, 2)
    x_traj = plt.subplot(4, 1, 1, sharex=betx)
    bety = plt.subplot(4, 1, 3, sharex=betx)
    disp = plt.subplot(4, 1, 4, sharex=betx)
    x_traj.plot(tw_sliced['s'], tw_sliced['x'], color='tab:red', label='Dip')
    x_traj.plot(tw_sliced_all['s'], tw_sliced_all['x'], color='tab:green', label='Dip+Quad')
    betx.plot(tw_smooth['s'], dbetax_sliced, color='tab:red', label='Dip')
    betx.plot(tw_smooth['s'], dbetax_sliced_all, color='tab:green', label='Dip+Quad')
    bety.plot(tw_smooth['s'], dbetay_sliced, color='tab:red', label='Dip')
    bety.plot(tw_smooth['s'], dbetay_sliced_all, color='tab:green', label='Dip+Quad')
    disp.plot(tw_smooth['s'], ddx_sliced, color='tab:red', label='Dip ')
    disp.plot(tw_smooth['s'], ddx_sliced_all, color='tab:green', label='Dip+Quad')
    x_traj.set_ylabel(r'x [m] ')
    betx.set_ylabel(r'$\Delta \beta_{x} \ / \ \beta_{x}$ ')
    bety.set_ylabel(r'$\Delta \beta_{y} \ / \ \beta_{y}$')
    disp.set_ylabel(r'$\Delta D_{x} \ / \ \sqrt{\beta_{x}} \ [m^{1/2}]$')
    disp.set_xlabel(r"s [m]")
    x_traj.legend(loc='upper right')
    plt.subplots_adjust(right=0.8)
    plt.show()

    fig = plt.figure()
    fig.suptitle('Sliced quadrupoles vs Averaged k1 quadrupoles', y=0.92)
    fig.set_size_inches(10, 8)  # Adjust the width and height of the figure
    betx = plt.subplot(4, 1, 2)
    x_traj = plt.subplot(4, 1, 1, sharex=betx)
    bety = plt.subplot(4, 1, 3, sharex=betx)
    disp = plt.subplot(4, 1, 4, sharex=betx)
    x_traj.plot(tw_meanquad['s'], tw_meanquad['x'], color='tab:red', label='Quad aver.')
    x_traj.scatter(tw_sliced_all['s'], tw_sliced_all['x'], color='tab:green', label='Quad sliced', marker='+')
    betx.scatter(tw_sliced_all_smooth['s'], dbetax_meanquad-dbetax_sliced_all, color='tab:red', marker='+')
    bety.scatter(tw_sliced_all_smooth['s'], dbetay_meanquad-dbetay_sliced_all, color='tab:red', marker='+')
    disp.scatter(tw_sliced_all_smooth['s'], ddx_meanquad-ddx_sliced_all, color='tab:red', marker='+')
    betx.set_ylabel(r'$\Delta \beta_{x} \ / \ \beta_{x}$ ')
    bety.set_ylabel(r'$\Delta \beta_{y} \ / \ \beta_{y}$')
    disp.set_ylabel(r'$\Delta D_{x} \ / \ \sqrt{\beta_{x}} \ [m^{1/2}]$')
    disp.set_xlabel(r"s [m]")
    x_traj.set_ylabel(r'x [m] ')
    x_traj.legend()
    plt.subplots_adjust(right=0.8)
    plt.show()
