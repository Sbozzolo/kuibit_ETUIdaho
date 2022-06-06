#!/usr/bin/env python3

import matplotlib.pyplot as plt

from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import setup_matplotlib, plot_color

ITERATION = 0
X0, X1 = (-30, -30), (30, 30)
RES = 300

sim = SimDir("tinyBNS")

vx_all = sim.gridfunctions.xy["vx"]
vy_all = sim.gridfunctions.xy["vy"]
rho_all = sim.gridfunctions.xy["rho_b"]

rho_it, vx_it, vy_it = rho_all[ITERATION], vx_all[ITERATION], vy_all[ITERATION]
vx_it_res = vx_it.to_UniformGridData([RES, RES], x0=X0, x1=X1)
vy_it_res = vy_it.to_UniformGridData([RES, RES], x0=X0, x1=X1)
rho_res = rho_it.to_UniformGridData([RES, RES], x0=X0, x1=X1)

rho_res.mask_less(1e-8)
vx_it_res.mask_apply(rho_res.mask)
vy_it_res.mask_apply(rho_res.mask)

X, Y = vx_it_res.coordinates_from_grid()

plt.clf()

plot_color(rho_res, xlabel="x", ylabel="y", colorbar=True,
           label="rho_b", logscale=True)

plt.streamplot(X, Y, vx_it_res.data_xyz, vy_it_res.data_xyz)
plt.savefig("/tmp/test.pdf")
