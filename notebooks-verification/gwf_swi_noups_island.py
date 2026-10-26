# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Conductance weighting for a strip-island lens
#
# A one-row, 100-column, unconfined, steady-state strip island with uniform
# recharge and a constant freshwater head at the two end cells. The interface
# stays well above the aquifer bottom, so the freshwater thickness is
# $(1 + \alpha_f) h$ and the exact Dupuit solution is
# $h^2 = h_e^2 + N (L^2 - x^2) / (K (1 + \alpha_f))$. Because $h^2$ is
# quadratic in $x$, the arithmetic mean of the cell thicknesses (AMT-LMK)
# reproduces it exactly, whereas upstream weighting of the thickness biases the
# heads low. The harmonic mean is shown for comparison.
#
# Ported from `autotest/test_gwf_swi_noups_island.py` on the `feat-swi` branch of
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

cases = {
    "island-upstream": "Upstream weighting (default)",
    "island-amtlmk": "No upstream weighting, AMT-LMK",
    "island-hmean": "No upstream weighting, harmonic mean",
}
no_upstream = {"island-upstream": False, "island-amtlmk": True, "island-hmean": True}
cell_averaging = {
    "island-upstream": None,
    "island-amtlmk": "AMT-LMK",
    "island-hmean": None,
}

Lx = 10000.0  # island width, in meters
delr, delc = 100.0, 1.0
ncol = int(Lx / delr)
top, botm = 20.0, -400.0
k = 10.0  # meters per day
recharge = 0.001  # meters per day
h_edge = 0.5  # freshwater head at the end cells, in meters
alphaf = 40.0

# %% [markdown]
# ## Build and run the models


# %%
def build_model(name):
    ws = model_root / name
    (ws / "output").mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(
        sim,
        no_ptcrecord=True,
        outer_maximum=500,
        inner_maximum=300,
        outer_dvclose=1.0e-9,
        inner_dvclose=1.0e-10,
        linear_acceleration="bicgstab",
    )
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=name, save_flows=True, newtonoptions="newton"
    )
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=1, ncol=ncol, delr=delr, delc=delc, top=top, botm=botm
    )
    flopy.mf6.ModflowGwfic(gwf, strt=h_edge)
    npf_kwargs = {}
    if cell_averaging[name] is not None:
        npf_kwargs["alternative_cell_averaging"] = cell_averaging[name]
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=k, **npf_kwargs)
    swi_kwargs = {}
    if no_upstream[name]:
        swi_kwargs["dev_no_upstream_weighting"] = True
    flopy.mf6.ModflowGwfswi(
        gwf, zeta_filerecord=f"output/{name}.zta", saltwater_head=0.0, **swi_kwargs
    )
    flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data=[[0, 0, 0, h_edge], [0, 0, ncol - 1, h_edge]]
    )
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
# ## Compare with the exact solution
#
# Every case must conserve mass and satisfy Ghyben-Herzberg. AMT-LMK must
# match the exact solution, and upstream weighting must be measurably worse and
# biased low.

# %%
x = (np.arange(ncol) + 0.5) * delr - 0.5 * Lx
L = 0.5 * Lx - 0.5 * delr
h_exact = np.sqrt(h_edge**2 + recharge * (L**2 - x**2) / (k * (1.0 + alphaf)))

results = {}
for name in cases:
    gwf = sims[name].get_model()
    ws = Path(sims[name].sim_path)
    head = gwf.output.head().get_data().flatten()
    zeta = flopy.utils.HeadFile(ws / f"output/{name}.zta", text="zeta")
    zeta = zeta.get_data().flatten()
    err = np.abs(head - h_exact).max()
    results[name] = head, zeta

    # Ghyben-Herzberg with a saltwater head of zero
    assert np.allclose(zeta, np.clip(-alphaf * head, botm, top), atol=1.0e-4)
    assert zeta.min() > botm + 1.0

    df = pd.read_csv(ws / f"output/{name}.bud.csv")
    imbalance = (df["TOTAL_IN"] - df["TOTAL_OUT"]).abs().max()
    peak = max(df["TOTAL_IN"].max(), df["TOTAL_OUT"].max())
    assert imbalance < 1.0e-6 + 1.0e-5 * peak

    if name == "island-amtlmk":
        assert err < 1.0e-3
    elif name == "island-upstream":
        assert err > 1.0e-2
        assert (head - h_exact).max() < 1.0e-3
    print(f"{name}: max |h - h_exact| = {err:.4e} m, budget imbalance {imbalance:.2e}")

# %% [markdown]
# ## Plot the results

# %%
fig, axes = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)
ax_h, ax_e, ax_z = axes
styles = {"island-upstream": "C3", "island-amtlmk": "C0", "island-hmean": "C2"}
xd = x + 0.5 * Lx
ax_h.plot(xd, h_exact, "k-", lw=2, label="Exact (Dupuit)")
ax_z.plot(xd, -alphaf * h_exact, "k-", lw=2, label="Exact (Dupuit)")
for name, title in cases.items():
    head, zeta = results[name]
    c = styles[name]
    ax_h.plot(xd, head, "o", color=c, mfc="none", ms=3, label=f"{name} -- {title}")
    ax_e.plot(xd, head - h_exact, "-", color=c, label=f"{name} -- {title}")
    ax_z.plot(xd, zeta, "o", color=c, mfc="none", ms=3, label=f"{name} -- {title}")
ax_h.set_title("Freshwater head")
ax_h.set_ylabel("Head, in meters")
ax_h.legend(loc="lower center", fontsize=8)
ax_e.axhline(0.0, color="k", lw=0.5, zorder=0)
ax_e.set_title("Head error")
ax_e.set_ylabel("Simulated minus exact head,\nin meters")
ax_z.set_title("Interface")
ax_z.set_ylabel("Elevation, in meters")
ax_z.set_xlabel("Distance, in meters")
plt.show()
