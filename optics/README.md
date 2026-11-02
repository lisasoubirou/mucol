# Optics

This folder contains the functions developed for the optics design and optimisation of the RCS lattices. It covers FODO, hybrid FODO and combined-function cells, as well as arc construction, dispersion suppressor matching and some optics analysis.

### Cell optimisation

- **`cf_cell.py`**: Functions for the optimisation of combined-function cells.
- **`hybrid_qp.py`**: Optimisation functions for conventional FODO and hybrid FODO cells.
- **`FODO_cells_energy_spread.json`**: Parameter file used for the RCS cell optimisation, including the momentum spread (`sigma_delta`) and alpha parameter for HFODO (`alpha`).

### Arc and dispersion suppressor construction

- **`generate_fodo.py`**: Previous implementation of the FODO arc construction. Deprecated; use `generate_fodo_normal.py` instead.
- **`generate_fodo_normal.py`**: Routines to build FODO arcs and match the corresponding dispersion suppressor.
- **`generate_fodo_hybrid.py`**: Routines to build hybrid FODO arcs and match the corresponding dispersion suppressor.

### Beam distribution generation

- **`generate_particles.py`**: Functions to generate transverse and longitudinal beam distributions matched to the lattice. Initially developed for lattice optimisation. The functions were subsequently updated and integrated into TrackingRCS in the tracking/ folder, which contains the more recent implementation.

### Optics analysis and plotting

- **`dp_accep.py`**: Provides an estimate of the momentum acceptance of the RCS.
- **`fodo_hybrid_time_var.py`**: Studies the evolution of the optics with energy in hybrid FODO lattices.
- **`plot_optics.py`**: Utility functions to plot Twiss parameters (beta functions and dispersion), as well as Montague functions and second-order dispersion.
- **`plot_twiss_strength.py`**: Functions to plot Twiss parameters together with magnet strengths and beam envelopes.

### Notebooks

- **`notebook_optics/`**: Notebooks documenting the initial optics design studies.
- **`notebook_optics_final/`**: Notebooks used for the optics design presented in the final HEC report.

### Deprecated code

- **`sc_field_error/`**: Old code developed to study the effects of systematic field errors in superconducting dipoles. Deprecated.