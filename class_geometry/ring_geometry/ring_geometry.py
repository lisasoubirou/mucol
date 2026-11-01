# -*- coding: utf-8 -*-
"""
@author: Lisa Soubirou
"""
import json
from pathlib import Path
import warnings
from typing import Tuple, Any, Optional, Dict, List
import copy
import yaml  # type: ignore
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.figure import Figure
import scipy.constants as cons  # type: ignore
from scipy.optimize import minimize  # type: ignore

from rcsparameters.acceleration import particle_parameters

plt.rc("axes", labelsize=12)    # fontsize of the x and y labels
plt.rc("xtick", labelsize=12)    # fontsize of the tick labels
plt.rc("ytick", labelsize=12)    # fontsize of the tick labels
plt.rc("legend", fontsize=12)    # legend fontsize

MUON_LIFETIME = particle_parameters.PosMuon().lifetime_at_rest  # 2.1969811e-6
MUON_MASS = particle_parameters.PosMuon().E_rest_in_eV

# Default parameters for RCS 2 Greenfield
base_input_parameters: Dict[str, Any] = {"Type of RCS" : "hybrid",
                         "Circumference [m]" : 5990.0,
                         "Injection energy [eV]" : 313.830e9,
                         "Extraction energy [eV]" : 750e9,
                         "Length short insertion [m]" : 2.5,
                         "Dipole spacing [m]" : 0.3,
                         "Number of cells per arc" : 8,
                         "Number of RF sections" : 24,
                         "Number of arcs" : 26,
                         "Dipole types and field [T]" : {"BSC": 10., 
                                                        "BNC": (1.8, "linear")},
                         "Pattern of a half-cell of dipoles" : ["BSC", "BNC", "BSC"],
                         "Filling ratio of arcs" : 0.85,
                         "Filling ratio of RF cavities in RF section" : 0.4
                        }


def load_file(file_input: str) -> dict:
    """Load file for different format"""
    if file_input is None:
        kw_file={}
    else:
        if isinstance(file_input, dict):
            kw_file = file_input
        elif Path(file_input).exists():
            try:
                with open(file_input, "r", encoding="utf-8") as f:
                    kw_file = json.load(f)
            except json.JSONDecodeError:
                try:
                    with open(file_input, "r", encoding="utf-8") as f:
                        kw_file = yaml.safe_load(f)
                except yaml.parser.ParserError as e:
                    raise yaml.parser.ParserError(f"{file_input} is an invalid file") from e
        else:
            raise FileNotFoundError(f"{file_input} does not exist")
    return kw_file


class RingGeometry:
    """ This class aims at computing geometric parameters of the RCS from basic parameters of the 
    acceleration chain."""

    def __init__(self, file_input: Optional[str] = None, **kw_user: Any) -> None:
        """ 
        Parameters
        ----------
        circumference : float
            Circumference of the ring [m]
        injection_energy : float
            Injection energy [eV]
        extraction_energy : float
            Extraction energy [eV]
        length_short_insertion : float
            Length of the straight section in between the half-cells of dipoles [m]
        dipole_spacing : float 
            Length in between dipoles [m]
        nb_arc : int
            Number of arcs in the RCS
        nb_RF_section : int 
            Number of RF insertion in the RCS
        nb_cell_arc : int
            Number of cells per arc. If non realistic, it will be decreased.
        dipoles: dict
            Describe the dipole families and their magnetic properties (field are in [T])
        pattern : list
            Dipole pattern of a half-cell (example: ["BSC","BNC","BSC"])
        filling_ratio : float
            Filling ratio of the arcs (remaining fraction is for RF insertion) 
        filling_ratio_RF : float
            Filling ratio of the RF in the RF insertion

        Those parameters can be modified, the geometry will be recomputed accordingly.
        """
        self._type: str = ""
        self._circumference: float = 0.0
        self._injection_energy: float = 0.0
        self._extraction_energy: float = 0.0
        self._length_short_insertion: float = 0.0
        self._dipole_spacing: float = 0.0
        self._nb_cell_arc: int = 0
        self._nb_RF_section: int = 0
        self._nb_arc: int = 0
        self._dipoles: Dict[str, Any] = {}
        self._pattern: List[str] = []
        self._filling_ratio: float = 0.0
        self._filling_ratio_RF: float = 0.0

        if file_input is None:
            self._type = base_input_parameters["Type of RCS"]
            self._circumference = base_input_parameters["Circumference [m]"]
            self._injection_energy = base_input_parameters["Injection energy [eV]"]
            self._extraction_energy = base_input_parameters["Extraction energy [eV]"]
            self._length_short_insertion = base_input_parameters["Length short insertion [m]"]
            self._dipole_spacing = base_input_parameters["Dipole spacing [m]"]
            self._nb_cell_arc = base_input_parameters["Number of cells per arc"]
            self._nb_RF_section = base_input_parameters["Number of RF sections"]
            self._nb_arc = base_input_parameters["Number of arcs"]
            self._dipoles = copy.deepcopy(base_input_parameters["Dipole types and field [T]"])
            self._pattern = copy.deepcopy(base_input_parameters["Pattern of a half-cell of dipoles"])
            self._filling_ratio = base_input_parameters["Filling ratio of arcs"]
            self._filling_ratio_RF = base_input_parameters["Filling ratio of RF cavities in RF section"]
        else:
            kw_file = load_file(file_input)
            kw_file.update(kw_user)
            for attr, val in kw_file.items():
                private_attr = f"_{attr}"
                if not hasattr(self, private_attr):
                    raise ValueError(f"Unknown parameter: {attr}")
                setattr(self, private_attr, val)

        self._nb_sub = 10 # Number of coordinates computed in each dipoles
        self.muon_mass = MUON_MASS
        self.muon_lifetime = MUON_LIFETIME
        self.t_inj = 0
        self.t_ext = 1
        self.dipole_families: dict = {}
        self.nb_dipole_families = 0

        self.reinit_all()

    def check_parameter_feasibility(self) -> None:
        """
        Check if input parameters are physical. 

        Raises:
            ValueError:  If one of the limits of the parameters is not met.
        """
        if self._circumference <= 0:
            raise ValueError("Circumference must be > 0")
        if self._injection_energy >= self.extraction_energy or self.injection_energy <= 0:
            raise ValueError("Injection energy must be smaller than extraction energy or > 0")
        if self._extraction_energy <= self.injection_energy or self.extraction_energy <= 0:
            raise ValueError("Extraction energy must be larger than injection energy or > 0")
        if self._nb_cell_arc <= 0:
            raise ValueError("Number of cells per arc must be at least 1")
        if self._nb_arc <= 0:
            raise ValueError("Number of arcs must be at least 1")
        if self._length_short_insertion <= 0:
            raise ValueError("length_short_insertion length must > 0")
        if self._dipole_spacing <= 0:
            raise ValueError("Dipole spacing must be > 0")
        if self._filling_ratio <= 0 or self._filling_ratio >= 1:
            raise ValueError("Filling ratio of the arc must be in the range ]0,1[")
        if self._filling_ratio_RF <= 0 or self._filling_ratio_RF >= 1:
            raise ValueError("Filling ratio of RF must be in the range ]0,1[")
        if self._type not in ["hybrid", "normal"]:
            raise ValueError("Type unknown")

    def reinit_all(self) -> None:
        """Re-initialize parameters and re-compute the geometry"""
        self.check_parameter_feasibility()
        self._dipole_parameters()
        self._calc_geo()
        self._z_dipole()

    def _assign_magnetic_field(self, key: str) -> None:
        """Attribute a magnetic field to each type of dipole from the given pattern. 
        Magnetic properties are given in the dict dipoles. 
        Ex : self.dipoles = {"B1": 10., "B2": [1.5, "linear"]} 
        A float is given for "B1": B = constant = 10 T
        A list is given for "B2":
            "linear":
                len(val) == 2: (example "B2": (1.5, "linear")) 
                Linear function from -1.5 to 1.5 T 
                len(val) == 3 : (example "B2": (-1.5, 1, "linear")) 
                Linear function from -1.5 to 1 T
            else: (example "BNC": [0.0, 1.8, -0.38197])
                B(t) defined with harmonics
        If a string is given: use of analytical expression
        """
        val = self._dipoles[key]
        if isinstance(val, (float, int)):  # Case B = constant (for steady/SC dipoles)
            self.dipole_families[key]["B"] = lambda x: val
        elif isinstance(val, (tuple, list, np.ndarray)):  # Case B(t) (for pulsed/NC dipoles)
            if val[-1] == "linear":
                if len(val) == 2:  # Linear, symmetric from -Bmax to Bmax
                    Bmax = val[0]
                    self.dipole_families[key]["B"] = lambda x: Bmax * (2 * x - 1)
                elif len(val) == 3: # Linear, from Bmin to Bmax,
                    self.dipole_families[key]["B"] = lambda x: val[0] + (val[1] - val[0]) * x
                else:
                    raise ValueError(f"The dipole family {key} is not valid !")
            else:  # With harmonics
                tab = np.array(val)
                self.dipole_families[key]["B"] = lambda x: np.sum([v * np.sin( i *(- np.pi / 2 + np.pi * x))
                                                                   for i, v in enumerate(tab)])
        else:
            raise ValueError(f"The dipole family {key} is not valid !")

    def _dipole_parameters(self) -> None:
        """ Associate an expression of B(t) to each type of dipole. 
        Create a dict dipole_families containing a dict dipole_families[key] for each type of magnet.
        Attributes of dipole_families[key] are detailed, with the functions that computes it :
        dipole_families[key] attributes computed in _assign_magnetic_field:
            B : Args : x(float) : time
                Returns: Magnetic field
        dipole_families[key] attributes computed in _calc_geo:
            length_half_cell : Dipole length in a half cell
            tot_length : Total dipole length in RCS
            nb_half_cell : nb dipole in half-cell
            nb_tot :  total nb of dipoles in RCS
            length : length of 1 dipole
            BL : Args : x(float) : time
                 Returns: B(x)*length on 1 half-cell
        """
        list_key = sorted(list(self._dipoles.keys()))
        self.dipole_families = dict()
        for key in list_key:
            self.dipole_families[key] = dict()
            self._assign_magnetic_field(key)
        self.nb_dipole_families = len(self.dipole_families)
        # Check all dipoles in pattern are defined
        for key in self._pattern:
            if key not in self.dipole_families:
                raise ValueError(f"{key} is not defined")

    def _calc_geo(self) -> None:
        """Calculates sequences list_a and list_b (see A. CHANCE note) that allows calculation
        of geometric parameters.        
        e0: initial angle of half-cell
        L0: Dipole length in a half cell (all dipoles included)
        dipole_length_tot : Total dipole length in RCS (all dipoles included)
        func_Brho : Args: x (float): time
                    Returns: Magnetic rigidity
        nc_best : optimised # of cells per arc, if nb_cell_arc > nc_best : nb_cell_arc = nc_best
        """
        self.e0 = np.sin(self.cell_angle/4)
        # Isomagnetic case, only 1 family of dipole (normal conducting RCS)
        if self.nb_dipole_families == 1:
            key = list(self.dipole_families.keys())[0]
            dip = self.dipole_families[key]
            Bext = dip["B"](self.t_ext)
            self.L0 = self.ext_Brho/Bext*2*self.e0  # Dipole length in a half cell
            self.dipole_length_tot = 2*self.nb_cell_rcs*self.L0  # Total dipole length in RCS
            dip["length_half_cell"] = self.L0
            dip["tot_length"] = self.dipole_length_tot
            dip["nb_half_cell"] = len(self.pattern)
            dip["nb_tot"] = dip["nb_half_cell"]*2*self.nb_cell_rcs
            dip["length"] = self.dipole_length_tot/dip["nb_tot"]
            dip["BL"] = lambda x: dip["length_half_cell"]*dip["B"](x)
            self.list_a = np.ones(dip["nb_half_cell"])/dip["nb_half_cell"]
            self.list_b = lambda x: np.ones(len(self.pattern))
            self.func_Brho = lambda x: self.L0*dip["B"](x)/2./self.e0  # Magnetic rigidity
        # More than 1 dipole family (hybrid with 2 families or more)
        else:
            A = np.ones((2, self.nb_dipole_families))
            list_key = sorted(self.dipole_families.keys())
            list_dip = [self.dipole_families[key] for key in list_key]
            A[0] = [dip["B"](self.t_inj)/2./self.e0 for dip in list_dip]
            A[1] = [dip["B"](self.t_ext)/2./self.e0 for dip in list_dip]
            B = np.ones(2)
            B[0] = self.inj_Brho
            B[1] = self.ext_Brho
            if np.abs(np.linalg.det(A[:, :2])) > 1e-6 and self.nb_dipole_families == 2:
                invA = np.linalg.inv(A[:, :2])
                Lsol = np.dot(invA, B)
            else:
                list_phi = np.linspace(self.t_inj, self.t_ext, 101)[1:-1]
                C = np.ones((len(list_phi), self.nb_dipole_families))
                for i, phi in enumerate(list_phi):
                    C[i] = [dip["B"](phi)/2./self.e0 for dip in list_dip]
                D = self.inj_Brho+(self.ext_Brho-self.inj_Brho)*(list_phi-self.t_inj
                                                            )/(self.t_ext-self.t_inj)
                constraint = ({"type": "eq",
                              "fun": lambda x:  np.sum((np.dot(A, x) - B)**2)},)
                bnds = [(0, None)]*self.nb_dipole_families
                res = minimize(lambda x: np.sum((np.dot(C, x) - D)**2),
                               np.ones(self.nb_dipole_families)/self.nb_dipole_families,
                               method="SLSQP", bounds=bnds,
                               constraints=constraint)
                if not res.success:
                    raise ValueError("No solution for this set of dipole families")
                Lsol = res.x
            self.L0 = np.sum(Lsol)  # Dipole length in a half cell
            self.dipole_length_tot = 2*self.nb_cell_rcs*self.L0  # Total dipole length in RCS
            pattern = list(self.pattern)
            for i, dip in enumerate(list_dip):
                dip["length_half_cell"] = Lsol[i]
                dip["tot_length"] = 2*self.nb_cell_rcs*Lsol[i]
                dip["nb_half_cell"] = pattern.count(list_key[i])
                dip["length"] = Lsol[i]/dip["nb_half_cell"]
                dip["nb_tot"] = dip["nb_half_cell"]*2*self.nb_cell_rcs
                dip["BL"] = lambda x, dip=dip: dip["length_half_cell"] * dip["B"](x)
            self.func_Brho = lambda x: np.sum([l*dip["B"](x) for l, dip in
                                        zip(Lsol, list_dip)])/2./self.e0  # Magnetic rigidity
            self.list_a = np.ones(len(self.pattern))
            for i, key in enumerate(self.pattern):
                dip = self.dipole_families[key]
                self.list_a[i] = dip["length_half_cell"]/self.L0/dip["nb_half_cell"]
            self.list_ba = lambda x: np.array([
                self.L0*self.dipole_families[key]["B"](x)/self.func_Brho(x)/2/self.e0
                                                    for key in self.pattern])  # type: ignore
            self.list_b = lambda x: np.where(np.abs(self.list_ba(x)) < 1e-8,  # type: ignore
                                             1e-8, self.list_ba(x))  # To find a solution when Bnc=0
        self.nc_best = int((self.arc_length-self.dipole_length_tot/self._nb_arc)/(
            (len(self.pattern)+1)*self._dipole_spacing+self._length_short_insertion)/2)  # Optimised nb_cell_arc

        # Number of cells per arc may have to be optimised, or could get unphysical geometry
        if self.nb_cell_arc > self.nc_best:
            if self.nc_best <= 0:
                raise ValueError("Optimum number of cells is negative: unphysical solution. Check input parameters.")
            warnings.warn(f"Number of cells per arc was optimised from {self._nb_cell_arc} to {self.nc_best}", UserWarning)
            self._nb_cell_arc = self.nc_best
            self._dipole_parameters()
            self._calc_geo()

    @property
    def type(self) -> str:
        """Type of the RCS: hybrid or normal"""
        return self._type

    @property
    def circumference(self) -> float:
        """Circumference [m]"""
        return self._circumference

    @circumference.setter
    def circumference(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Circumference must be > 0")
        self._circumference = value
        self.reinit_all()

    @property
    def injection_energy(self) -> float:
        """Injection energy [eV]"""
        return self._injection_energy

    @injection_energy.setter
    def injection_energy(self, value: float) -> None:
        if value >= self.extraction_energy or value <= 0:
            raise ValueError("Injection energy must be smaller than extraction energy or > 0")
        self._injection_energy = value
        self.reinit_all()

    @property
    def extraction_energy(self) -> float:
        """Extraction energy [eV]"""
        return self._extraction_energy

    @extraction_energy.setter
    def extraction_energy(self, value: float) -> None:
        if value <= self.injection_energy or value <= 0:
            raise ValueError("Extraction energy must be larger than injection energy or > 0")
        self._extraction_energy = value
        self.reinit_all()

    @property
    def nb_cell_arc(self) -> int:
        """Number of cells in 1 arc"""
        return self._nb_cell_arc

    @nb_cell_arc.setter
    def nb_cell_arc(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Number of cells per arc must be at least 1")
        self._nb_cell_arc = value
        self.reinit_all()

    @property
    def nb_arc(self) -> int:
        """Number of arc in RCS"""
        return self._nb_arc

    @nb_arc.setter
    def nb_arc(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Number of arcs must be at least 1")
        self._nb_arc = value
        self.reinit_all()

    @property
    def nb_RF_section(self) -> int:
        """Number of RF insertion in RCS"""
        return self._nb_RF_section

    @nb_RF_section.setter
    def nb_RF_section(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Number of RF sections must be at least 1")
        self._nb_RF_section = value
        self.reinit_all()

    @property
    def nb_sub(self) -> int:
        """Number of plot subdivision"""
        return self._nb_sub

    @nb_sub.setter
    def nb_sub(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Number of subs must be at least 1")
        self._nb_sub = value

    @property
    def length_short_insertion(self) -> float:
        """Length of the straight section in between the cells, for qp/sxt"""
        return self._length_short_insertion

    @length_short_insertion.setter
    def length_short_insertion(self, value: float) -> None:
        if value <= 0:
            raise ValueError("length_short_insertion length must > 0")
        self._length_short_insertion = value
        self.reinit_all()

    @property
    def dipole_spacing(self) -> float:
        """Length between the dipoles in a cell"""
        return self._dipole_spacing

    @dipole_spacing.setter
    def dipole_spacing(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Dipole spacing must be > 0")
        self._dipole_spacing = value
        self.reinit_all()

    @property
    def filling_ratio(self) -> float:
        """Filling ratio of the arc length wrt C"""
        return self._filling_ratio

    @filling_ratio.setter
    def filling_ratio(self, value: float) -> None:
        if value <= 0 or value >= 1:
            raise ValueError("Filling ratio must be in the range [0,1]")
        self._filling_ratio = value
        self.reinit_all()

    @property
    def filling_ratio_RF(self) -> float:
        """Filling ratio of the RF in the RF section"""
        return self._filling_ratio_RF

    @filling_ratio_RF.setter
    def filling_ratio_RF(self, value: float) -> None:
        if value <= 0 or value >= 1:
            raise ValueError("Filling ratio of RF must be in the range [0,1]")
        self._filling_ratio_RF = value
        self.reinit_all()

    @property
    def dipoles(self) -> dict:
        """Describe the dipole families and their magnetic properties"""
        return self._dipoles

    @dipoles.setter
    def dipoles(self, value: dict) -> None:
        self._dipoles = value
        self.reinit_all()

    @property
    def pattern(self) -> list:
        """Dipole pattern of a half-cell (example: ["BSC","BNC","BSC"])"""
        return self._pattern

    @pattern.setter
    def pattern(self, value: list) -> None:
        self._pattern = value
        self.reinit_all()

    @property
    def L_extra_rcs(self) -> float:
        """Extra length left for the rcs after computing the cell length"""
        return (self.tot_arc_length - self.dipole_length_tot - 2*self.nb_cell_rcs*(self.nd+1)*self.dipole_spacing -
                self.length_short_insertion*2*self.nb_cell_rcs)

    @property
    def L_extra_arc(self) -> float:
        """Extra length left for one arc after computing the cell length"""
        return self.L_extra_rcs/self.nb_arc

    @property
    def inj_gamma(self) -> float:
        """Injection gamma"""
        return self.injection_energy/self.muon_mass

    @property
    def ext_gamma(self) -> float:
        """Extraction gamma"""
        return self.extraction_energy/self.muon_mass

    @property
    def inj_Brho(self) -> float:
        """Magnetic rigidity at injection [eV/c]"""
        return np.sqrt(self.inj_gamma**2-1)*self.muon_mass/cons.c

    @property
    def ext_Brho(self) -> float:
        """Magnetic rigidity at extraction [eV/c]"""
        return np.sqrt(self.ext_gamma**2-1)*self.muon_mass/cons.c

    @property
    def tot_arc_length(self) -> float:
        """Total length of arcs in RCS [m]"""
        return self._filling_ratio*self._circumference

    @property
    def tot_insertion_length(self) -> float:
        """Total length of insertion in RCS [m]"""
        return self._circumference-self.tot_arc_length

    @property
    def arc_length(self) -> float:
        """Length of 1 arc [m]"""
        return self.tot_arc_length/self._nb_arc

    @property
    def insertion_length(self) -> float:
        """Length of 1 insertion [m]"""
        return self.tot_insertion_length/self._nb_arc

    @property
    def RF_length_tot(self) -> float:
        """Total length dedicated to RF in RF insertion [m]"""
        return self._filling_ratio_RF*self.insertion_length*self._nb_RF_section

    @property
    def nb_cell_rcs(self) -> float:
        """Total number of cells in RCS"""
        return self._nb_cell_arc*self._nb_arc

    @property
    def cell_length(self) -> float:
        """Length of 1 cell [m]"""
        return self.arc_length/self._nb_cell_arc

    @property
    def cell_angle(self) -> float:
        """Angle of 1 cell"""
        return 2*np.pi/self.nb_cell_rcs

    @property
    def filling_ratio_dipole(self) -> float:
        """Filling ratio of dipole in the RCS: Ltot/C"""
        return self.dipole_length_tot/self._circumference

    @property
    def nd(self) -> int:
        """Number of dipoles per half_cell"""
        return len(self.list_a)

    @property
    def ref_bending(self) -> float:
        """Reference bending [1/m]"""
        return 2*self.e0/self.L0

    @property
    def rho(self) -> float:
        """Reference radius [m]"""
        return 1/self.ref_bending

    @property
    def l_dipoles(self) -> np.ndarray:
        """Array of the length of all dipoles [m]"""
        return self.L0*self.list_a

    def theta(self, x: float) -> np.ndarray:
        """ Bending angle of dipoles [rad]
        Args: x (float): time
        Returns: list 
        """
        eps=self.epsilon(x)
        return -eps[1:]+eps[:-1]

    def L_dd_path(self, x: float) -> np.ndarray:
        """ Path length in the drift between the dipoles [m]
        Args: x (float): time
        Returns: list
        """
        return self._dipole_spacing/np.cos(self.epsilon(x)[0: len(self._pattern)])

    def L_dip(self, x: float) -> np.ndarray:
        """ Chord length of trajectory in dipoles [m]
        Args: x (float): time
        Returns: list
        """
        ye = self.y_end(x)
        yb = self.y_begin(x)
        return np.array([np.sqrt((self.dipole_families[key]["length"])**2 +
            (ye[ii]-yb[ii])**2) for ii, key in enumerate(self._pattern)])

    def L_dip_path(self, x: float) -> np.ndarray:
        """ Path length in dipoles [m]
        Args: x (float): time
        Returns: list
        """
        ee = self.epsilon(x)
        e1 = ee[1:]
        e0 = ee[:-1]
        ang = e0-e1
        return ang/self.dip_bending(x)

    def L_qp_dip_path(self, x: float) -> np.ndarray:
        """ Path length in the drift between a quad and a dipole
        Args: x (float): time
        Returns: float
        """
        ee = self.epsilon(x)
        return self._dipole_spacing/np.cos(ee[0])

    def _dz(self, x: float) -> np.ndarray:
        """ Deviation of position in dipole. Function defined to compute _z_dipole (see below). 
        Args: x (float): time
        Returns: list
        """
        ex = self.epsilon(x)
        bb = self.list_b(x)
        return self.rho * np.append([0], np.cumsum((np.cos(ex[1:]) - np.cos(ex[:-1])) / bb))

    def _dld(self, x: float) -> np.ndarray:
        """ Deviation of position in drift. Function defined to compute _z_dipole (see below). 
        Args: x (float): time
        Returns: list
        """
        ex = self.epsilon(x)
        return self._dipole_spacing * np.append([0], np.cumsum(np.tan(ex[1:-1])))

    def _z_dipole(self) -> None:
        """ Calculates coordinates of points in complex plane:
            z_begin : at entry of dipoles 
            z_end : at exit of dipoles 
        """
        cuml = np.append([0], np.cumsum(self.l_dipoles))
        cumld = np.arange(self.nd)*self._dipole_spacing
        dz = self._dz
        dld = self._dld
        self.z_begin = lambda x: [c1 + c2 + 1j*(z + d) for c1, c2, z, d in zip(
            cuml[:-1], cumld, dz(x)[:-1], dld(x))]
        self.z_end = lambda x: [c1 + c2 + 1j*(z+d) for c1, c2, z, d in zip(
            cuml[1:], cumld, dz(x)[1:], dld(x))]

    def dip_bending(self, x: float) -> np.ndarray:
        """Bending of dipoles [1/m]
        Args: x (float): time
        Returns: list
        """
        return self.list_b(x) * self.ref_bending

    def sn(self, x: float) -> np.ndarray:
        """ Intermediate sequence sn_i to compute sequence phi_i
        Args: x (float): time
        Returns: list 
        """
        return np.append([0], np.cumsum(self.list_a*self.list_b(x)))

    def phi(self, x: float) -> np.ndarray:
        """ Sequence of phi_i=sin(epsilon_i) (see function epsilon below)
        Args: x (float): time
        Returns: list
        """
        return self.e0*(1-2*self.sn(x))

    def epsilon(self, x: float) -> np.ndarray:
        """ Edge angles at the entrance of dipoles [rad]
        Args: x (float): time 
        Returns: list
        """
        return np.arcsin(self.phi(x))

    def zn(self, x: float) -> np.ndarray:
        """ Coordinates of intermediary points inside the dipoles. 
        The number of points is defined by nb_sub.
        Args: x (float): time
        Returns: list
        """
        eps = self.epsilon(x)
        z_b = self.z_begin(x)
        bb = self.list_b(x)
        z = np.zeros((self.nb_sub + 1) * self.nd + 2, dtype=complex)
        z[1] = (self._dipole_spacing+ self.L_extra_arc/self.nb_cell_arc/4)/np.cos(eps[0])*np.exp(1j*eps[0])
        dc = np.linspace(0., 1., self.nb_sub+1)
        for n in np.arange(self.nd):
            e0 = eps[n]
            e1 = eps[n+1]
            e = e0 + dc*(e1-e0)
            cc = (np.cos(e)-np.cos(e0))/bb[n]
            i0 = 1 + n*(self.nb_sub+1)
            z0 = z[1]+z_b[n]
            z[i0:i0+self.nb_sub+1] = z0+self.l_dipoles[n]*dc+1j*self.rho*cc
        z[-1] = z[-2] + (self._dipole_spacing + self.L_extra_arc/self.nb_cell_arc/4)/np.cos(eps[-1])*np.exp(1j*eps[-1])
        return z

    def y_begin(self, x: float) -> np.ndarray:
        """ y coordinate at entry of each dipole [m]
        Args: x (float): time
        Returns: list
        """
        return np.imag(self.z_begin(x))

    def y_end(self, x: float) -> np.ndarray:
        """ y coordinate at exit of each dipole [m]
        Args: x (float): time
        Returns: list
        """
        return np.imag(self.z_end(x))

    def extrema(self, x: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.float64, np.float64]:
        """Calculates the extrema of trajectories y_min, y_max and width 
        covered by the trajectory in each dipole.
        Args: x (float): time
        Returns: 
            y_min (list) : min y coordinate for the trajectory in each dipole
            y_max (list) : max y coordinate for the trajectory in each dipole
            width (list) : width = y_max-y_min of trajectory in each dipole
            max_apert_noshift (float) : np.max(y_max)-np.min(y_min) : max aperture if 
              no shift are made between the dipoles, "they are all aligned" 
            max_apert (float): np.max(width[1:-1]), max aperture if the dipoles are positioned to
            minimize overall width. 
        """
        y_min = np.zeros(self.nd)
        y_max = np.zeros(self.nd)
        list_e = self.epsilon(x)
        list_z_beg = self.y_begin(x)
        list_z_end = self.y_end(x)
        lb=self.list_b(x)
        for n in np.arange(self.nd):
            e0 = list_e[n]
            e1 = list_e[n+1]
            y0 = list_z_beg[n]
            y1 = list_z_end[n]
            bb = lb[n]
            if e0 > 0:
                if e1 > 0:
                    y_min[n] = y0
                    y_max[n] = y1
                else:
                    y_min[n] = np.min([y0, y1])
                    y_max[n] = y0+self.rho*(1-np.cos(e0))/bb
            elif e1 > 0:
                y_min[n] = y0+self.rho*(1-np.cos(e0))/bb
                y_max[n] = np.max([y0, y1])
            else:
                y_min[n] = y1
                y_max[n] = y0
        width = y_max - y_min
        max_apert_noshift = np.max(y_max)-np.min(y_min)
        max_apert = np.max(width[1:-1])
        return y_min, y_max, width, max_apert_noshift, max_apert

    def path_length(self, x: float) -> float:
        """Path length in ONE HALF-CELL, which pattern is given as input. 
        Length insertion for quadrupoles is included.
        Args: x (float): time
        Returns: float
        """
        ee = self.epsilon(x)
        e1 = ee[1:]
        e0 = ee[:-1]
        ang = e0-e1
        path_length = np.sum(ang/self.dip_bending(x))+self._dipole_spacing*np.sum(1/np.cos(e1[:-1]))+(
            self._length_short_insertion+2*self._dipole_spacing)/np.cos(e0[0])
        return path_length

    def path_length_tot(self, x: float) -> float:
        """ Total path length in the nb_arc of the RCS. 
        Path length in RF insertion is not included, as it undergoes no variation during ramping. 
        Args: x (float): time
        Returns: float
        """
        return 2*self._nb_cell_arc*self._nb_arc*self.path_length(x)

    def plot_traj(self, t_traj: np.ndarray = np.linspace(0, 1, 8)) -> Figure:
        """ Plots the trajectories for different times.
        Args:
            t_traj (list): time for each trajectory, between [0,1]
        """
        fig = plt.figure(figsize=(6.4, 4.8))
        for i in t_traj:
            plt.plot(np.real(self.zn(i)), np.imag(self.zn(i)))

        zloc = self._dipole_spacing + self.L_extra_arc/self.nb_cell_arc/4
        iz = 1

        ax = plt.gca()
        for idx, (z1, z2) in enumerate(zip(self.z_begin(1), self.z_end(1))):
            x1, x2 = np.real(z1), np.real(z2)

            plt.axvline(zloc + x1, ls="--", color="k", alpha=0.5)
            plt.axvline(zloc + x2, ls="--", color="k", alpha=0.5)

            dipole_type = self.pattern[idx][1:]

            plt.text(
                zloc + 0.5*(x1+x2),
                1.02,
                dipole_type,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center")

            iz += 1
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.tight_layout()
        plt.show()

        return fig

    def plot_path_length_diff(self, t_traj: np.ndarray = np.linspace(0, 1, 20)) -> Figure:
        """Plot the total path length difference.

        Args:
            t_traj (np.ndarray, optional): time for each trajectory, between [0,1]. Defaults to np.linspace(0, 1, 20).
        """

        path_t = [self.path_length_tot(t) for t in t_traj]
        path_min = np.min(path_t)

        fig = plt.figure(figsize=(6.4, 4.8))
        plt.plot(t_traj, (path_t-path_min)*1e3)
        plt.xlabel(r"$t \ / \ t_{acc}$")
        plt.ylabel("Path length variation [mm]")
        plt.tight_layout()
        plt.show()

        return fig
