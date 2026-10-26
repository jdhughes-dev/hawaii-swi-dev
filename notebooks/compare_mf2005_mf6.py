# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MODFLOW-2005 SWI2 and MODFLOW 6 SWI
#
# Converts a MODFLOW-2005 SWI2 model of Rotzoll and Izuka (2023) to a MODFLOW 6
# model with the SWI Package, runs it, and compares the simulated interface and
# heads with the MODFLOW-2005 output in the data release.
#
# The MODFLOW-2005 models are steady-state flow models in which SWI2 moves the
# interface toward equilibrium over a long stress period. The MODFLOW 6 model
# is a steady-state, single-fluid SWI simulation, in which the interface is in
# equilibrium with the freshwater head after one time step.
#
# Run `pixi run get-data <island>` first. Model files are written to
# `notebooks/models/<island>/` and output to its `output/` subdirectory.

# %%
import sys
from pathlib import Path

import flopy
import matplotlib.pyplot as plt
import numpy as np

root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pixi.toml").exists())
sys.path.insert(0, str(root / "scripts"))
from mf2005_to_mf6 import convert, mf2005_heads_zeta

# %% [markdown]
# ## Island and scenario
#
# Scenario 0 is the Current scenario (2010 conditions).

# %%
island = "oahu"
scenario = 0

ws = root / "notebooks" / "models" / island
if scenario != 0:
    ws = ws.with_name(f"{island}.{scenario}")

# %% [markdown]
# ## Convert and run the MODFLOW 6 model

# %%
sim = convert(island, scenario, ws)
sim.write_simulation(silent=True)
success, _ = sim.run_simulation(silent=True)
assert success, "MODFLOW 6 did not converge"
gwf = sim.get_model()

# %% [markdown]
# ## Load the results
#
# The MODFLOW-2005 interface and heads are averages of the last two saved time
# steps, as in Izuka and Rotzoll (2023), which damps the interface oscillation
# at sharp hydraulic-conductivity contrasts.

# %%
head05, zeta05 = mf2005_heads_zeta(island, scenario)

zeta6 = flopy.utils.HeadFile(ws / "output" / f"{island}.zta", text="zeta").get_data()
head6 = gwf.output.head().get_data()

active = gwf.dis.idomain.array > 0
top = gwf.dis.top.array
botm = gwf.dis.botm.array

# %% [markdown]
# ## Compare the interface
#
# Cells where the MODFLOW-2005 interface is at the aquifer bottom (all
# freshwater) or at the top of the aquifer (all saltwater) are reported
# separately from cells where the interface is within the aquifer.

# %%
dz = zeta6 - zeta05
dh = head6 - head05
layer_top = np.vstack((top[np.newaxis], botm[:-1]))
within = active & (zeta05 > botm + 1.0e-3) & (zeta05 < layer_top - 1.0e-3)


def summarize(label, diff, mask):
    d = diff[mask]
    print(
        f"{label:28s} n={d.size:7d}  mean={d.mean():9.3f}  "
        f"mean |d|={np.abs(d).mean():8.3f}  max |d|={np.abs(d).max():9.3f}"
    )


summarize("zeta, all active cells", dz, active)
summarize("zeta, interface in aquifer", dz, within)
summarize("head, all active cells", dh, active & (head05 > -1.0e29))

# %% [markdown]
# ## Plot the results

# %%
for k in range(gwf.dis.nlay.array):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    zmin = float(np.nanmin(np.where(active[k], zeta05[k], np.nan)))
    for ax, z, title in (
        (axes[0, 0], zeta05, "MODFLOW-2005 SWI2"),
        (axes[0, 1], zeta6, "MODFLOW 6 SWI"),
    ):
        pmv = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=k)
        im = pmv.plot_array(np.ma.masked_where(~active, z), vmin=zmin, vmax=0.0)
        ax.set_title(f"{title} interface, layer {k + 1}")
        ax.set_aspect("equal")
    fig.colorbar(im, ax=axes[0, :], label="Interface elevation, in feet", shrink=0.8)

    lim = max(1.0, float(np.percentile(np.abs(dz[k][within[k]]), 99)))
    pmv = flopy.plot.PlotMapView(model=gwf, ax=axes[1, 0], layer=k)
    im = pmv.plot_array(
        np.ma.masked_where(~active, dz), cmap="RdBu_r", vmin=-lim, vmax=lim
    )
    axes[1, 0].set_title("MODFLOW 6 minus MODFLOW-2005")
    axes[1, 0].set_aspect("equal")
    fig.colorbar(im, ax=axes[1, 0], label="Interface difference, in feet", shrink=0.8)

    ax = axes[1, 1]
    ax.plot(zeta05[k][within[k]], zeta6[k][within[k]], ".", ms=1, alpha=0.3)
    ax.plot([zmin, 0], [zmin, 0], "k-", lw=0.8)
    ax.set_xlabel("MODFLOW-2005 interface elevation, in feet")
    ax.set_ylabel("MODFLOW 6 interface elevation, in feet")
    ax.set_title("Cells with the interface in the aquifer")
plt.show()

# %% [markdown]
# ## Reference
#
# Rotzoll, K., and Izuka, S.K., 2023, MODFLOW-2005 and SWI2 models for assessing
# groundwater availability scenarios in volcanic aquifers on Kauai, Oahu, and
# Maui, Hawaii: U.S. Geological Survey data release,
# https://doi.org/10.5066/P97CPK5C.
