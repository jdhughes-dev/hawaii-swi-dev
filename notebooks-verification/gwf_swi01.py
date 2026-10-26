# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Steady-state freshwater lens
#
# A one-layer, steady-state cross section of an island 10,000 m wide with
# uniform recharge and the sea at both ends. The interface simulated by the
# SWI Package is compared with the analytical solution of Fetter (1972) for
# confined, unconfined, and unconfined Newton-Raphson formulations.
#
# Ported from `autotest/test_gwf_swi01.py` on the `feat-swi` branch of
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

cases = {
    "swi01a": "Confined",
    "swi01b": "Unconfined",
    "swi01c": "Unconfined Newton",
}
top = {"swi01a": 0.0, "swi01b": 50.0, "swi01c": 50.0}
icelltype = {"swi01a": 0, "swi01b": 1, "swi01c": 1}
newtonoptions = {"swi01a": False, "swi01b": False, "swi01c": "newton"}

Lx = 10000.0  # island width, in meters
delr, delc = 100.0, 1.0
ncol = int(Lx / delr) + 1
nlay, nrow = 1, 1
botm = -400.0
recharge = 0.001  # meters per day
k = 10.0  # meters per day
h0 = 0.0  # sea level, in meters

# %% [markdown]
# ## Build and run the models


# %%
def build_model(name):
    ws = model_root / name
    (ws / "output").mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(sim, outer_maximum=500, linear_acceleration="bicgstab")
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=name, save_flows=True, newtonoptions=newtonoptions[name]
    )
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top[name],
        botm=botm,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_specific_discharge=True,
        alternative_cell_averaging="amt-hmk",
        icelltype=icelltype[name],
        k=k,
    )
    flopy.mf6.ModflowGwfswi(gwf, zeta_filerecord=f"output/{name}.zta")
    cghb = 10.0 * delr * delc / 10.0
    flopy.mf6.ModflowGwfghb(
        gwf,
        stress_period_data=[[0, 0, 0, h0, cghb], [0, 0, ncol - 1, h0, cghb]],
    )
    flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"output/{name}.cbc",
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
# ## Compare with the analytical solution
#
# Fetter (1972) gives the head in the freshwater lens for a relative density
# difference of 1/40.

# %%
alphaf = 40.0


def analytical_head(x, a, confined):
    if confined:
        return np.sqrt(recharge * (a**2 - x**2) / k / alphaf)
    return np.sqrt(recharge * (a**2 - x**2) / k / (1 + alphaf))


def head_comparison(name):
    gwf = sims[name].get_model()
    xc = gwf.modelgrid.xcellcenters.flatten()
    jstart = (ncol - 1) // 2
    length = xc[-1] - xc[0]
    x = xc[jstart:] - (length / 2 + delr / 2)
    h_analytical = analytical_head(x, length / 2.0, name == "swi01a")
    h_model = gwf.output.head().get_data().flatten()[jstart:]
    return x, h_analytical, h_model


for name in cases:
    x, h_analytical, h_model = head_comparison(name)
    diff = h_model - h_analytical
    if name == "swi01c":
        # Newton-Raphson heads are lower than, but close to, the analytical heads
        assert np.all(diff[:-1] < 0.0) and np.all(diff[:-1] > -0.5)
    else:
        assert np.allclose(diff[:-1], 0.0, atol=0.003)
    print(f"{name}: maximum head difference {np.abs(diff[:-1]).max():.4f} m")

# %% [markdown]
# ## Plot the results

# %%
fig, axes = plt.subplots(len(cases), 2, figsize=(10, 8), constrained_layout=True)
for (name, title), (ax_xs, ax_h) in zip(cases.items(), axes):
    gwf = sims[name].get_model()
    ws = Path(sims[name].sim_path)
    x = gwf.modelgrid.xcellcenters.flatten()
    head = gwf.output.head().get_data().flatten()
    zeta = flopy.utils.HeadFile(ws / f"output/{name}.zta", text="zeta")
    zeta = zeta.get_data().flatten()
    tp = np.zeros(head.shape) if name == "swi01a" else head

    pxs = flopy.plot.PlotCrossSection(gwf, line={"row": 0}, ax=ax_xs)
    ax_xs.fill_between(x, tp, zeta, color="cyan", label="freshwater")
    ax_xs.fill_between(x, zeta, botm, color="red", label="saltwater")
    ax_xs.plot(x, head, "k-", label="head")
    ax_xs.plot(x, zeta, "k--", label="interface")
    ax_xs.set_ylim(-400, 50)
    ax_xs.set_title(f"{name} -- {title}")
    ax_xs.set_ylabel("Elevation, in meters")

    xa, h_analytical, h_model = head_comparison(name)
    ax_h.plot(xa, h_analytical, "b-", label="analytical")
    ax_h.plot(xa, h_model, "bo", mfc="none", label="simulated")
    ax_h.set_title(f"{name} -- {title}")
    ax_h.set_ylabel("Head, in meters")
axes[0, 0].legend(loc="lower left")
axes[0, 1].legend()
axes[-1, 0].set_xlabel("Distance, in meters")
axes[-1, 1].set_xlabel("Distance from the island center, in meters")
plt.show()

# %% [markdown]
# ## Reference
#
# Fetter, C.W., 1972, Position of the saline water interface beneath oceanic
# islands: Water Resources Research, v. 8, no. 5, p. 1307--1315.
