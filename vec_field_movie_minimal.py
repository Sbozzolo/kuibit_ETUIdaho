#!/usr/bin/env python3

import matplotlib
from matplotlib.pyplot import clf as clear_figure
from matplotlib.pyplot import streamplot

from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import (
    plot_color,
    save,
)

class MOPIMovie:
    def __init__(self, args):
        self.sim = SimDir("tinyBNS", pickle_file="sim.pickle")
        self.x0, self.x1, self.res = (-30, -30), (30, 30), 300
        self.shape = [self.res, self.res]
        self.reader = self.sim.gridfunctions.xy
        self.var = self.reader["rho_b"]
        self.vec_var0 = self.reader["vx"]
        self.vec_var1 = self.reader["vy"]

        self.iterations = self.var.available_iterations

        self.args = args

    def get_frames(self):
        return self.iterations

    def make_frame(self, path, iteration):

        clear_figure()

        data = self.var[iteration]
        label = "rho_b"

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

        data_res.mask_less(1e-8)
        vec_data0_res.mask_apply(data_res.mask)
        vec_data1_res.mask_apply(data_res.mask)

        plot_color(
            data_res,
            xlabel="x",
            ylabel="y",
            colorbar=True,
            logscale=True,
            vmin=-10,
            vmax=-3,
            label=label
        )

        X, Y = data_res.coordinates_from_grid()

        streamplot(X, Y, vec_data0_res.data_xyz, vec_data1_res.data_xyz)

        save(path)
