#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt

from kuibit import SimDir
from kuibit.visualize_matplotlib import setup_matplotlib, plot_color

# PLOT A GRID FUNCTION AND A VECTOR FIELD ON A PLANE APPLYING A MASK BASED ON
# THE GRID DATA (e.g., remove the atmosphere)

# Options

# What iteration do we want to plot?
ITERATION = 21120

# In what region?
# X0 is the lower left corner, X1 the upper right
X0, X1 = (-30, -30), (30, 30)

# On which plane
# (Note tinyBNS only contains xy data)
PLANE = "xy"

# What resolution to use for the plot?
RES = 300

# Name of the vector variable
VEC_VAR_NAME = "v"
# Name of the scalar variable
SCA_VAR_NAME = "rho_b"

# Plot with a logarithmic scale?
LOGSCALE = True

# Create a mask for values smaller than this for the scalar variable.
# Ignore points that are masked in the three variables
MASK_LESS_VALUE = 1e-8

# Where to find the data
DIRNAME = "tinyBNS"

# Save a pickle file here. Pickle files reduce the cost of analyzing the same
# data multiple times
PICKLE_NAME = "sim.pickle"

# We use a context manager here. The goal of this context manager is to save
# part of the progresses of the analysis to the pickle file. Alternatively, one
# could simply call sim = SimDir(DIRNAME)
with SimDir(DIRNAME, pickle_file=PICKLE_NAME) as sim:

    # Setup

    # Prepare the readers. Each of the ..._all variables is a reader to read the
    # corresponding variable on the given plane. It is a dictionary-like object
    # which keys are the iterations.

    # We can see what iterations are in the data with something like
    # sca_all.available_iterations, which returns a list

    vx_all = sim.gridfunctions[PLANE][f"{VEC_VAR_NAME}{PLANE[0]}"]
    vy_all = sim.gridfunctions[PLANE][f"{VEC_VAR_NAME}{PLANE[1]}"]
    sca_all = sim.gridfunctions[PLANE][f"{SCA_VAR_NAME}"]

    # This step is not necessary. It is here only because Gabriele Bozzola
    # doesn't like the default values of matplotlib. Passing a dictionary to
    # setup_matplotlib() will apply those changes to the default values of
    # matplotlib.
    setup_matplotlib()
    # {"figure.figsize": (20, 10)}

    # Reading current iteration
    sca_it, vx_it, vy_it = sca_all[ITERATION], vx_all[ITERATION], vy_all[ITERATION]

    # The ..._it variables are HierachicalGridData. They contain all the
    # information for all the refinement levels. They cannot be plotted
    # directly. Instead, we must resample to a UniformGridData.

    print(f"Read variables at iteration {ITERATION} (time = {sca_it.time:.3e})")

    # Here we perform the resampling. RES is the resolution along the two axis,
    # x0 is the coordinate on the lower left corner, and x1 of the top right.

    vx_it_res = vx_it.to_UniformGridData([RES, RES], x0=X0, x1=X1)
    vy_it_res = vy_it.to_UniformGridData([RES, RES], x0=X0, x1=X1)
    sca_res = sca_it.to_UniformGridData([RES, RES], x0=X0, x1=X1)

    # Now that we have the resampled data, we can apply the mask. First, we
    # apply the mask to the scalar data.
    sca_res.mask_less(MASK_LESS_VALUE)
    # Then, we apply the same mask to the vector data.
    vx_it_res.mask_apply(sca_res.mask)
    vy_it_res.mask_apply(sca_res.mask)

    print(f"Max {VEC_VAR_NAME}{PLANE[0]}: {vx_it_res.abs_max():.3e}")
    print(f"Max {VEC_VAR_NAME}{PLANE[1]}: {vy_it_res.abs_max():.3e}")

    # Get the coordinates as 2D NumPy arrays
    X, Y = vx_it_res.coordinates_from_grid()

    # "Technical" step. The inferno colormap looks better if we paint all the
    # masked color as black (instead of the default: white)
    cmap = matplotlib.cm.get_cmap("inferno").copy()
    cmap.set_bad("black", alpha=1.)

    plt.clf()

    # plot_color is a powerful convenience function for plotting
    # UniformGridData. As the zeroth level, it is a wrapper around imshow for
    # UniformGridData (and HierarchicalGridData, and even OneGridFunction). It
    # does its best to plot whatever it is given.

    plot_color(sca_res, xlabel=f"{PLANE[0]}", ylabel=f"{PLANE[1]}",
               colorbar=True, label=SCA_VAR_NAME, logscale=LOGSCALE,
               cmap=cmap)

    # We can use standard matplotlib. The data in UniformGridData is stored in
    # the .data attribute. Due to the difference in conventions (Fortran vs C),
    # if we want to plot it we should use the .data_xyz attribute instead.
    plt.streamplot(X, Y, vx_it_res.data_xyz, vy_it_res.data_xyz,
                   color="white")
    plt.savefig("/tmp/test.pdf")
