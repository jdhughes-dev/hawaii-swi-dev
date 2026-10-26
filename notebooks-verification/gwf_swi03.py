# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Coupled freshwater and saltwater models
#
# The island cross section of `gwf_swi02` simulated with two coupled GWF
# models, one for the freshwater and one for the saltwater, joined by the
# SWI-SWI exchange and solved together in one solution. Recharge is applied
# to the freshwater model only. The freshwater interface must satisfy the
# Ghyben-Herzberg relation, match the saltwater interface, and the saltwater
# head must remain at sea level.
#
# Ported from `autotest/test_gwf_swi03.py` on the `feat-swi` branch of
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

name = "swi03"
models = {"freshwater": "Freshwater model", "saltwater": "Saltwater model"}

Lx = 10000.0  # island width, in meters
delr, delc = 100.0, 1.0
ncol = int(Lx / delr)
nlay, nrow = 1, 1
top = 0.0
botm = -400.0
recharge = 0.001  # meters per day
k = 10.0  # meters per day
h0 = 0.0  # sea level, in meters
perlen, nstp = 100000.0, 1  # days

# %% [markdown]
# ## Build and run the simulation


# %%
def build_gwf_model(sim, mname):
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=mname, save_flows=True, newtonoptions=True
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
    flopy.mf6.ModflowGwfic(gwf)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_specific_discharge=True,
        alternative_cell_averaging=None,
        icelltype=1,
        k=k,
    )
    flopy.mf6.ModflowGwfswi(gwf, zeta_filerecord=f"output/{mname}.zta")
    cghb = 1.0 * delr * delc / 10.0
    flopy.mf6.ModflowGwfghb(
        gwf,
        stress_period_data=[[0, 0, 0, h0, cghb], [0, 0, ncol - 1, h0, cghb]],
    )
    if mname == "freshwater":
        flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"output/{mname}.cbc",
        budgetcsv_filerecord=f"output/{mname}.bud.csv",
        head_filerecord=f"output/{mname}.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return gwf


def build_simulation():
    ws = model_root / name
    (ws / "output").mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim, perioddata=[(perlen, nstp, 1.0)])
    ims = flopy.mf6.ModflowIms(
        sim,
        no_ptcrecord=True,
        outer_maximum=50,
        linear_acceleration="bicgstab",
    )
    gwfs = [build_gwf_model(sim, mname) for mname in models]
    flopy.mf6.ModflowSwiswi(
        sim,
        exgtype="SWI6-SWI6",
        exgmnamea="freshwater",
        exgmnameb="saltwater",
    )
    sim.register_ims_package(ims, [gwf.name for gwf in gwfs])
    return sim


sim = build_simulation()
sim.write_simulation(silent=True)
success, _ = sim.run_simulation(silent=True)
assert success, f"{name} did not run"

# %% [markdown]
# ## Check the results
#
# The freshwater interface must lie 40 times the freshwater head below sea
# level and coincide with the saltwater interface, the saltwater head must be
# zero, and both model budgets must close.

# %%
ws = Path(sim.sim_path)
head, zeta = {}, {}
for mname in models:
    gwf = sim.get_model(mname)
    head[mname] = gwf.output.head().get_data().flatten()
    zobj = flopy.utils.HeadFile(ws / f"output/{mname}.zta", text="zeta")
    zeta[mname] = zobj.get_data().flatten()

zeta_answer = -40.0 * head["freshwater"]
assert np.allclose(zeta["freshwater"], zeta_answer)
assert np.allclose(zeta["freshwater"], zeta["saltwater"])
assert np.allclose(head["saltwater"], 0.0)
print(
    "maximum zeta difference "
    f"{np.abs(zeta['freshwater'] - zeta_answer).max():.2e} m (Ghyben-Herzberg), "
    f"{np.abs(zeta['freshwater'] - zeta['saltwater']).max():.2e} m "
    "(freshwater - saltwater)"
)
print(f"maximum saltwater head {np.abs(head['saltwater']).max():.2e} m")

# absolute imbalance relative to the peak flow
for mname in models:
    df = pd.read_csv(ws / f"output/{mname}.bud.csv")
    imbalance = (df["TOTAL_IN"] - df["TOTAL_OUT"]).abs().max()
    peak = max(df["TOTAL_IN"].max(), df["TOTAL_OUT"].max())
    tol = 1.0e-6 + 1.0e-4 * peak
    assert imbalance < tol
    print(f"{mname}: maximum budget imbalance {imbalance:.2e} m3/d")

# %% [markdown]
# ## Plot the results

# %%
fig, axes = plt.subplots(
    len(models), 1, figsize=(8, 6), sharex=True, constrained_layout=True
)
for (mname, title), ax in zip(models.items(), axes):
    gwf = sim.get_model(mname)
    x = gwf.modelgrid.xcellcenters.flatten()
    tp = np.zeros(x.shape)
    flopy.plot.PlotCrossSection(gwf, line={"row": 0}, ax=ax)
    ax.fill_between(x, tp, zeta[mname], color="cyan", label="freshwater")
    ax.fill_between(x, zeta[mname], botm, color="red", label="saltwater")
    ax.plot(x, head[mname], "b-", label="head")
    ax.plot(x, zeta[mname], "k--", label="interface")
    ax.set_ylim(-400, 50)
    ax.set_title(title)
    ax.set_ylabel("Elevation, in meters")
axes[0].legend(loc="lower center", ncol=4)
axes[-1].set_xlabel("Distance, in meters")
plt.show()
