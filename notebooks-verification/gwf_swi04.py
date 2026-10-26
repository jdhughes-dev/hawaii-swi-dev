# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Coastal aquifer with a sloping sea floor
#
# A one-layer, steady-state cross section 10,000 m long. The aquifer top slopes
# from -20 m at the seaward edge up to -1 m at the coastline (x = 3,000 m) and
# steps up to 10 m onshore. The sea is represented by GHB cells over the
# submarine part, and recharge is applied everywhere. The SWI Package uses a
# fixed saltwater head (sea level) of 0 m and 1 m in the two cases.
#
# Ported from `autotest/test_gwf_swi04.py` on the `feat-swi` branch of
# [christianlangevin/modflow6](https://github.com/christianlangevin/modflow6/tree/feat-swi).

# %%
from pathlib import Path

import flopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% [markdown]
# ## Model parameters
#
# Model files are written to `models/<case>/` beside the notebooks, and output
# to its `output/` subdirectory.

# %%
root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pixi.toml").exists())
model_root = root / "notebooks-verification" / "models"

# saltwater head (sea level), in meters
cases = {"swi04a": 0.0, "swi04b": 1.0}

dmax = -20.0  # deepest sea floor elevation, in meters
dmin = -1.0  # shallowest sea floor elevation, in meters
Lx_coast = 3000.0  # coastline position, in meters
Lx_edge = 10000.0  # domain length, in meters
delr, delc = 100.0, 1.0
nlay, nrow = 1, 1
ncol = int(Lx_edge / delr)
top_island = 10.0
botm_aquifer = -200.0
recharge = 0.001  # meters per day
hydraulic_conductivity = 100.0  # meters per day
ghb_cond_fact = 0.001

dx = Lx_edge / ncol
x = np.linspace(dx / 2.0, Lx_edge - dx / 2, ncol)
slope = (dmin - dmax) / Lx_coast
top = dmax + x * slope
top[x > Lx_coast] = top_island
top = top.reshape((nrow, ncol))

# %% [markdown]
# ## Build and run the models


# %%
def build_model(name):
    ws = model_root / name
    (ws / "output").mkdir(parents=True, exist_ok=True)
    h0 = cases[name]
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(
        sim,
        no_ptcrecord=True,
        outer_maximum=500,
        linear_acceleration="bicgstab",
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True, newtonoptions=True)
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm_aquifer,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    flopy.mf6.ModflowGwfnpf(
        gwf, save_specific_discharge=True, icelltype=1, k=hydraulic_conductivity
    )
    flopy.mf6.ModflowGwfswi(
        gwf, zeta_filerecord=f"output/{name}.zta", saltwater_head=h0
    )
    # equivalent freshwater head of the sea at the sea floor
    cghb = hydraulic_conductivity * ghb_cond_fact * delr * delc / 1
    ghb_list = []
    for j in np.where(x < Lx_coast)[0]:
        freshwater_head = h0 + (h0 - top[0, j]) * 0.025
        ghb_list.append([0, 0, j, freshwater_head, cghb])
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=ghb_list)
    flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"output/{name}.cbc",
        budgetcsv_filerecord=f"output/{name}.bud.csv",
        head_filerecord=f"output/{name}.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return sim


sims = {}
for name in cases:
    sim = build_model(name)
    sim.write_simulation(silent=True)
    success, _ = sim.run_simulation(silent=True)
    assert success, f"{name} did not run"
    sims[name] = sim

# %% [markdown]
# ## Check the results
#
# The interface must satisfy the Ghyben-Herzberg relation
# zeta = -40 h + 41 h_s, limited to the cell top and bottom, and the budget
# must close.

# %%
topf = top.flatten()
for name, h0 in cases.items():
    gwf = sims[name].get_model()
    ws = Path(sims[name].sim_path)
    head = gwf.output.head().get_data().flatten()
    zeta = gwf.swi.output.zeta().get_data().flatten()

    zeta_expected = np.clip(-40.0 * head + 41.0 * h0, botm_aquifer, topf)
    assert np.allclose(zeta, zeta_expected, atol=1.0e-4)
    assert np.all(zeta >= botm_aquifer - 1.0e-6), "zeta below aquifer bottom"
    assert np.all(zeta <= topf + 1.0e-6), "zeta above aquifer top"

    # absolute imbalance relative to the peak flow
    df = pd.read_csv(ws / f"output/{name}.bud.csv")
    imbalance = (df["TOTAL_IN"] - df["TOTAL_OUT"]).abs().max()
    peak = max(df["TOTAL_IN"].max(), df["TOTAL_OUT"].max())
    tol = 1.0e-6 + 1.0e-4 * peak
    assert imbalance < tol
    print(
        f"{name}: maximum zeta difference "
        f"{np.abs(zeta - zeta_expected).max():.2e} m, "
        f"maximum budget imbalance {imbalance:.2e} m3/d"
    )

# %% [markdown]
# ## Plot the results

# %%
fig, axes = plt.subplots(
    len(cases), 1, figsize=(8, 6), sharex=True, constrained_layout=True
)
for (name, h0), ax in zip(cases.items(), axes):
    gwf = sims[name].get_model()
    head = gwf.output.head().get_data().flatten()
    zeta = gwf.swi.output.zeta().get_data().flatten()
    pxs = flopy.plot.PlotCrossSection(gwf, line={"row": 0}, ax=ax)
    pxs.plot_fill_between(zeta, head=head, colors=["cyan", "red"], edgecolors="none")
    sea = x < Lx_coast
    ax.fill_between(x[sea], top[0, sea], h0, color="steelblue", alpha=0.4, label="sea")
    ax.plot(x, head, "b-", label="head")
    ax.plot(x, top[0], "k-", lw=1, label="aquifer top")
    ax.axvline(Lx_coast, color="k", ls=":", lw=1, label="coastline")
    ax.set_ylim(botm_aquifer, top_island)
    ax.set_title(f"{name} -- saltwater head {h0:.1f} m")
    ax.set_ylabel("Elevation, in meters")
axes[0].legend(loc="lower right", ncol=2)
axes[-1].set_xlabel("Distance, in meters")
plt.show()
