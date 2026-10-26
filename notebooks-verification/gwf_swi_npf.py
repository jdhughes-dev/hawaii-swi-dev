# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SWI with no saltwater reproduces NPF
#
# A confined, three-layer cross section with heterogeneous horizontal and
# vertical hydraulic conductivity, recharge, constant heads at the top corners,
# and an injection well in the bottom layer, so that vertical flow is both
# upward and downward. With the default saltwater head of zero the interface
# ($\zeta = -\alpha_f h_f$) is far below the model bottom, so the SWI Package
# must reproduce the heads of the same model run without it.
#
# Ported from `autotest/test_gwf_swi_npf.py` on the `feat-swi` branch of
# [christianlangevin/modflow6](https://github.com/christianlangevin/modflow6/tree/feat-swi).

# %%
from pathlib import Path

import flopy
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## Model parameters
#
# Model files are written to `models/<case>/` beside the notebooks, and output
# to its `output/` subdirectory.

# %%
root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pixi.toml").exists())
model_root = root / "notebooks-verification" / "models"

cases = {"swi-npf": "With SWI", "swi-npf-ref": "Without SWI (NPF only)"}

nlay, nrow, ncol = 3, 1, 10
delr, delc = 100.0, 1.0
top = 10.0
botm = [5.0, 0.0, -5.0]

# heterogeneous, reproducible hydraulic conductivity, in meters per day
rng = np.random.default_rng(2024)
k = rng.uniform(1.0, 100.0, size=(nlay, nrow, ncol))
k33 = rng.uniform(0.01, 5.0, size=(nlay, nrow, ncol))

recharge = 0.02  # meters per day
qwell = 100.0  # injection rate, in cubic meters per day
hchd = 8.0  # constant head, in meters

# %% [markdown]
# ## Build and run the models


# %%
def build_model(name, with_swi):
    ws = model_root / name
    (ws / "output").mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(
        sim,
        linear_acceleration="bicgstab",
        outer_maximum=100,
        inner_maximum=100,
    )
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=name, save_flows=True, newtonoptions="newton"
    )
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=8.0)
    flopy.mf6.ModflowGwfnpf(
        gwf, icelltype=0, k=k, k33=k33, save_specific_discharge=True
    )
    flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[[(nlay - 1, 0, ncol // 2), qwell]])
    flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data=[[(0, 0, 0), hchd], [(0, 0, ncol - 1), hchd]]
    )
    if with_swi:
        flopy.mf6.ModflowGwfswi(gwf, zeta_filerecord=f"output/{name}.zta")
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"output/{name}.hds",
        budget_filerecord=f"output/{name}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return sim


sims = {}
for name in cases:
    sim = build_model(name, with_swi=name == "swi-npf")
    sim.write_simulation(silent=True)
    success, _ = sim.run_simulation(silent=True)
    assert success, f"{name} did not run"
    sims[name] = sim

# %% [markdown]
# ## Compare the heads
#
# The flow field must have both upward and downward vertical flow for the
# comparison to exercise the vertical conductance.

# %%
gwf_swi = sims["swi-npf"].get_model()
gwf_npf = sims["swi-npf-ref"].get_model()
spdis = gwf_swi.output.budget().get_data(text="DATA-SPDIS")[0]
qz = spdis["qz"]
print(f"qz range: [{qz.min():.3e}, {qz.max():.3e}]")
assert qz.min() < 0.0 and qz.max() > 0.0

h_swi = gwf_swi.output.head().get_data()
h_npf = gwf_npf.output.head().get_data()
dmax = np.abs(h_swi - h_npf).max()
print(f"max |h_swi - h_npf| = {dmax:.3e}")
assert np.allclose(h_swi, h_npf, atol=1.0e-9, rtol=0.0)

# %% [markdown]
# ## Plot the results

# %%
fig, (ax_xs, ax_h) = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
pxs = flopy.plot.PlotCrossSection(gwf_swi, line={"row": 0}, ax=ax_xs)
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf_swi)
qzmax = np.abs(qz).max()
pc = pxs.plot_array(qz, cmap="RdBu_r", vmin=-qzmax, vmax=qzmax)
pxs.plot_grid(color="0.5", lw=0.5)
fig.colorbar(pc, ax=ax_xs, label="Vertical specific discharge,\nin meters per day")
ax_xs.set_title("With SWI -- vertical flow (positive upward)")
ax_xs.set_ylabel("Elevation, in meters")

x = gwf_swi.modelgrid.xcellcenters.flatten()
colors = ["C0", "C1", "C2"]
for ilay in range(nlay):
    ax_h.plot(x, h_npf[ilay, 0], "-", color=colors[ilay], label=f"Layer {ilay + 1}")
    ax_h.plot(x, h_swi[ilay, 0], "o", color=colors[ilay], mfc="none")
ax_h.plot([], [], "k-", label=cases["swi-npf-ref"])
ax_h.plot([], [], "ko", mfc="none", label=cases["swi-npf"])
ax_h.set_title("Heads with and without SWI")
ax_h.set_xlabel("Distance, in meters")
ax_h.set_ylabel("Head, in meters")
ax_h.legend()
plt.show()
