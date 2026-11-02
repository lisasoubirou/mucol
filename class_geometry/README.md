# Class Geometry

This folder contains the tools developed for the geometry definition of the hybrid synchrotrons and the initial FODO cell optimisation.

It is organised into three subfolders:

### `parameter_files/`
Contains the parameter files defining the RCS configurations for the two siting options:

- **Greenfield siting:**
  - `RCS1`: Low Energy (LE)
  - `RCS2`: Medium Energy (ME)
  - `RCS3`: High Energy (HE)
  - `RCS4`: Final Energy (FE)
- **CERN siting:**
  - `CERN_RCS1`, `CERN_RCS2`, etc.

These files provide the input parameters used to define the geometry of the different RCS configurations.

### `ring_geometry/`
Contains the functions used to describe the geometry of hybrid synchrotrons.

An example Jupyter notebook is provided to illustrate how to use these functions.

### `lattice_cell/`
Contains an initial function for FODO cell optimisation, aiming to minimise the total quadrupole length while providing an estimate of the momentum compaction factor.

An example Jupyter notebook is provided.