#!/usr/bin/env python3

# Copyright (C) 2020-2022 Gabriele Bozzola
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/>.

"""Visualize 2D grid data, optionally adding drawing the horizons.
"""

import matplotlib
from matplotlib.pyplot import clf as clear_figure
from matplotlib.pyplot import streamplot

from kuibit import argparse_helper as kah
from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import (
    add_text_to_corner,
    plot_color,
    plot_horizon_on_plane_at_iteration,
    save,
    setup_matplotlib,
)


def mopi_add_custom_options(parser):
    # These are the custom options that mopi will see.

    parser.add_argument("--datadir", default=".", help="Data directory.")
    kah.add_horizon_to_parser(parser)
    parser.add_argument(
        "--ignore-symlinks",
        action="store_true",
        help="Ignore symlinks in the data directory",
    )
    parser.add_argument("--pickle-file", help="Read SimDir from this file.")
    kah.add_grid_to_parser(parser, dimensions=2)
    kah.add_figure_to_parser(parser)
    parser.add_argument(
        "--variable", type=str, required=True, help="Variable to plot."
    )
    parser.add_argument(
        "--vector-variable", type=str, required=True, help="Variable to plot the field lines."
    )
    parser.add_argument(
        "--multilinear-interpolate",
        action="store_true",
        help="Whether to interpolate to smooth data with multilinear"
        " interpolation before plotting.",
    )
    parser.add_argument(
        "--interpolation-method",
        type=str,
        default="none",
        help="Interpolation method for the plot. See docs of np.imshow."
        " (default: %(default)s)",
    )
    parser.add_argument(
        "--colorbar",
        action="store_true",
        help="Whether to draw the color bar.",
    )
    parser.add_argument(
        "--logscale",
        action="store_true",
        help="Whether to use log scale.",
    )
    parser.add(
        "--mask-value",
        help=(
            "Mask values smaller than this one."
        ),
        type=float,
    )
    parser.add(
        "--vmin",
        help=(
            "Minimum value of the variable. "
            "If logscale is True, this has to be the log. "
        ),
        type=float,
    )
    parser.add(
        "--vmax",
        help=(
            "Maximum value of the variable. "
            "If logscale is True, this has to be the log."
        ),
        type=float,
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Whether to take the absolute value.",
    )


class MOPIMovie:
    def __init__(self, args):
        # Here we initialize all the objects that we need for all the frames.
        # All the expensive stuff has to be done here.

        # TODO: Automatically compute min and max if they are not specified.

        self.sim = SimDir(
            args.datadir,
            ignore_symlinks=args.ignore_symlinks,
            pickle_file=args.pickle_file,
        )
        self.x0, self.x1, self.res = args.origin, args.corner, args.resolution
        self.shape = [self.res, self.res]
        self.reader = self.sim.gridfunctions[args.plane]
        self.var = self.reader[args.variable]
        self.vec_var0 = self.reader[f"{args.vector_variable}{args.plane[0]}"]
        self.vec_var1 = self.reader[f"{args.vector_variable}{args.plane[1]}"]

        self.iterations = self.var.available_iterations

        # Apparent horizons
        self.ahs = {}

        if args.ah_show:
            for ah in self.sim.horizons.available_apparent_horizons:
                self.ahs[ah] = self.sim.horizons.get_apparent_horizon(ah)

        self.args = args

    def get_frames(self):
        # Here we define what is a "frame" (it is one iteration). This function
        # has to return an iterable on what we want to make frames of.
        return self.iterations

    def make_frame(self, path, iteration):
        # Here we plot a frame. This function has to take the output path and
        # the identifier of what is a frame (in this case, an iteration).

        setup_matplotlib()
        clear_figure()

        if self.args.absolute:
            data = abs(self.var[iteration])
            variable = f"abs({self.args.variable})"
        else:
            data = self.var[iteration]
            variable = self.args.variable

        vec_data0 = self.vec_var0[iteration]
        vec_data1 = self.vec_var1[iteration]

        # Resample
        data_res = data.to_UniformGridData(x0=self.x0,
                                           x1=self.x1,
                                           shape=self.shape)
        vec_data0_res = vec_data0.to_UniformGridData(x0=self.x0,
                                                     x1=self.x1,
                                                     shape=self.shape)
        vec_data1_res = vec_data1.to_UniformGridData(x0=self.x0,
                                                     x1=self.x1,
                                                     shape=self.shape)

        if self.args.mask_value is not None:
            data_res.mask_less(self.args.mask_value)
            vec_data0_res.mask_apply(data_res.mask)
            vec_data1_res.mask_apply(data_res.mask)

        if self.args.logscale:
            label = f"log10({variable})"
        else:
            label = variable

        cmap = matplotlib.cm.get_cmap("inferno").copy()
        cmap.set_bad("black", alpha=1.)

        plot_color(
            data_res,
            xlabel=self.args.plane[0],
            ylabel=self.args.plane[1],
            resample=self.args.multilinear_interpolate,
            colorbar=self.args.colorbar,
            logscale=self.args.logscale,
            vmin=self.args.vmin,
            vmax=self.args.vmax,
            label=label,
            interpolation=self.args.interpolation_method,
            cmap=cmap
        )

        X, Y = data_res.coordinates_from_grid()

        streamplot(X, Y, vec_data0_res.data_xyz, vec_data1_res.data_xyz,
                   color="white")

        if self.args.ah_show:
            for hor in self.ahs.values():
                # Check if we have the shape at the current iteration
                if iteration in hor.shape_iterations:
                    plot_horizon_on_plane_at_iteration(
                        hor,
                        iteration,
                        self.args.plane,
                        color=self.args.ah_color,
                        edgecolor=self.args.ah_edge_color,
                        alpha=self.args.ah_alpha,
                    )

        time = self.var.time_at_iteration(iteration)
        add_text_to_corner(rf"$t = {time:.3f}$")

        save(path)

        self.var.clear_cache()



