# timelags

This repository contains the Python scripts associated to the paper:

Ducellier A., Creager K.C. Depth and thickness of tectonic tremor in the northeastern Olympic Peninsula, _to be submitted_.

The cross-correlation of the seismic recordings are done with src/stack_ccorr_tremor.py.

The computation of the corresponding time lags is done with src/get_timelag.py.

The files containing the time lags for all the tremor-array locations are the XX_timelag.pkl, where XX is the two-letter code for the array.

The corresponding depths are computed using src/plot_envelopes.py. The corresponding thicknesses are computed using src/plot_thickness.py

The input files for the maps are then created using src/write_files_depth.py and src/write_files_thickness.py.

The GMT scripts to plot the map are in src/map_depth and src/map_thick.
