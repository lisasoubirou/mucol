# Muon Collider – RCS Chain

This repository contains the work carried out on the **Rapid Cycling Synchrotron (RCS) chain for the Muon Collider project**. It includes tools and functions developed for the geometry definition, lattice optimisation, optics design, and beam dynamics studies of the RCSs.

The repository is organised into the following main directories:

### `class_geometry/`
Contains the work on the geometry of hybrid synchrotrons, along with the initial functions developed for FODO cell optimisation.

### `optics/`
Contains functions for the optimisation of FODO, hybrid FODO, and combined-function cells. It also includes functions for building dispersion suppressors for FODO and hybrid FODO lattices.

### `lattices/`
Contains the results of cell optimisation studies, as well as the resulting arc lattice configurations and their associated parameter tables.

### `EB_mismatch/`
Contains the main functions developed to study the effects of mismatches between the energy increase and the linearly ramped magnetic fields of the magnets. It also includes functions to build the lattice from injection to extraction.

### `tracking/`
Contains the methods used to track particles through the lattices, along with Jupyter notebooks investigating the beam dynamics of different RCS configurations.

### `tools/`
Contains additional utility functions used across the different parts of the project.