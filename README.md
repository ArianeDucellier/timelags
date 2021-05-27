# timelags

This repository contains the Python scripts associated to the paper:

Ducellier A., Creager K.C. Depth and thickness of tectonic tremor in the northeastern Olympic Peninsula, _to be submitted_.

The tremor catalogs used in the paper are the mat files in data.

Figure 1 of the paper is done using the files in figures.

The cross-correlation of the seismic recordings are done with src/stack_ccorr_tremor.py. The corresponding output files should be stored in src/cc but are not on the Github repository due to their large volume. They are available on request.

The clusterings of the time windows are done with src/cluster_select.py. This Python script has also been used to make Figures 2 and 3.

The computation of the corresponding time lags is done with src/get_timelag.py.

The files containing the time lags for all the tremor-array locations are the src/XX_timelag.pkl, where XX is the two-letter code for the array.

The scripts to compute the depth given a timelag and a 1D velocity model are in misc.py. The corresponding velocity models are stored in the src/ttgrid_*.pkl.

The corresponding depths of the tremor are computed using src/plot_envelopes.py. Figure 4 and the figures in the supplement are also made with plot_envelopes.py.

The difference in timelags and depths for the EW and NS components are also computed with src/plot_envelopes.py, and then plotted using plot_diff_EW_NS.py.

The variations of the depth with the Poisson's ratio is plotted using src/plot_variations.py.

The corresponding thicknesses of the tremor zone are computed using src/plot_thickness.py.

The input files for the maps are then created using src/write_files_depth.py and src/write_files_thickness.py.

The GMT scripts to plot the maps are in src/map_depth and src/map_thick.

The linear regression to fit the plane is done with src/fit_plane.py.

The two notebooks in notebooks explain in more detail how we have downloaded and preprocessed the seismic data, and how we have computed and stacked the cross-correlations.
