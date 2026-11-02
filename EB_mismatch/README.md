# Energy–Field Mismatch Studies

This folder contains the tools and studies investigating the effects of a mismatch between the energy increase of the beam and the magnetic field ramp of the magnets during acceleration in the RCS.

### Main functions

- **`EB_error.py`**: Main functions to build the start-to-end lattice including the energy–field mismatch. Can be used to build individual arc lines or complete RCS lines from injection to extraction. Supports the construction of a reference line (without mismatch) and lines including the mismatch, using either sliced magnets or perturbation matrices.
- **`EBPlotter.py`**: Functions to plot the results of the mismatch studies, including trajectory and optics perturbations.
- **`EBModelOptics.py`**: Contains the matrices used in the analytical model to describe dipoles and quadrupoles, including their associated offsets.
- **`EB_scaling.py`**: Studies the scaling laws associated with the mismatch, in particular its dependence on the number of RF stations and the magnetic field ramp rate.

### Parameter and beam dynamics studies

- **`EB_jitter.py`**: Studies the impact of current jitter on the fast magnetic field ramp.
- **`mismatch_chroma_study.py`**: Investigates the effect of chromaticity on trajectory and optics perturbations.
- **`mismatch_no_acc.py`**: Studies the effect of the energy–field mismatch without acceleration, using an 8-arc lattice.
- **`mismatch_optics_study.py`**: Detailed study of optics perturbations as a function of the magnetic field ramp rate and number of arcs. Also includes convergence studies on the number of slices used to model the magnets.

### Illustrative plots

- **`mismatch_plot_1.py`**: Illustrates the mismatch between the energy increase and magnetic field ramp (E vs. B).
- **`mismatch_plot_2.py`**: Illustrates the errors on the energy deviation (`delta E`) and magnetic rigidity (`rho`).