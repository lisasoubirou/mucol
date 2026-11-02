# Tracking

This folder contains the functions, notebooks and scripts developed for beam tracking studies through the RCS lattices. It includes the analysis of particle losses, emittance evolution and the impact of different acceleration parameters and magnetic field errors.

### Tracking functions

- **`track_function.py`**: Older tracking implementation, including Xsuite tracking, particle loss calculations and emittance computation.
- **`tracking_chain_functions.py`**: Older functions for tracking particles through the full RCS chain, including cumulative loss calculations and transverse beam adaptation.
- **`TrackingRCS.py`**: Current implementation of the tracking class for tracking particles through an RCS lattice. Includes methods to plot the RF bucket and beam distribution, compute emittances and plot their evolution during tracking.

### Notebooks

Contains Jupyter notebooks used to investigate and compare different tracking configurations:

- **`tracking_compare`**: Test studies comparing different longitudinal emittances and injection energies.
- **`tracking_rcs1`**: Tracking studies for RCS1.
- **`tracking_rcs234`**: Tracking studies for RCS2, RCS3 and RCS4.

### Tracking scripts

Contains the scripts used to perform the main tracking studies, including:

- Impact of the longitudinal emittance.
- Impact of the injection energy.
- Comparison of tracking results with and without energy–field mismatch in RCS1.
- Impact of current jitter during the RCS1 fast ramp.