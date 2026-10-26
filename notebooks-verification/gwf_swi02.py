# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Transient freshwater lens
#
# The one-layer island cross section of `gwf_swi01`, run transient for
# 100,000 days from an initial head of zero until the freshwater lens reaches
# steady state. The interface simulated by the SWI Package is checked against
# the Ghyben-Herzberg relation for confined, unconfined, and unconfined
# Newton-Raphson formulations.
#
# Ported from `autotest/test_gwf_swi02.py` on the `feat-swi` branch of
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
    "swi02a": "Confined",
    "swi02b": "Unconfined",
    "swi02c": "Unconfined Newton",
}
top = {"swi02a": 0.0, "swi02b": 50.0, "swi02c": 50.0}
icelltype = {"swi02a": 0, "swi02b": 1, "swi02c": 1}
iconvert = {"swi02a": 0, "swi02b": 1, "swi02c": 1}
newtonoptions = {"swi02a": False, "swi02b": False, "swi02c": "newton"}

Lx = 10000.0  # island width, in meters
delr, delc = 100.0, 1.0
ncol = int(Lx / delr)
nlay, nrow = 1, 1
botm = -400.0
recharge = 0.001  # meters per day
k = 10.0  # meters per day
ss, sy = 1.0e-5, 0.2
h0 = 0.0  # sea level, in meters
perlen, nstp = 100000.0, 10  # days

# %% [markdown]
# ## Build and run the models


# %%
def build_model(name):
    ws = model_root / name
    (ws / "output").mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim, perioddata=[(perlen, nstp, 1.0)])
    flopy.mf6.ModflowIms(sim, outer_maximum=50, linear_acceleration="bicgstab")
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
    flopy.mf6.ModflowGwfic(gwf)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_specific_discharge=True,
        alternative_cell_averaging=None,
        icelltype=icelltype[name],
        k=k,
    )
    flopy.mf6.ModflowGwfsto(gwf, iconvert=iconvert[name], ss=ss, sy=sy)
    flopy.mf6.ModflowGwfswi(gwf, zeta_filerecord=f"output/{name}.zta")
    cghb = 1.0 * delr * delc / 10.0
    flopy.mf6.ModflowGwfghb(
        gwf,
        stress_period_data=[[0, 0, 0, h0, cghb], [0, 0, ncol - 1, h0, cghb]],
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
# ## Check the results
#
# At the end of the simulation the interface must lie 40 times the head below
# sea level, and the budget must close in every time step.


# %%
def read_zeta(name):
    ws = Path(sims[name].sim_path)
    return flopy.utils.HeadFile(ws / f"output/{name}.zta", text="zeta")


for name in cases:
    gwf = sims[name].get_model()
    ws = Path(sims[name].sim_path)
    head = gwf.output.head().get_data().flatten()
    zeta = read_zeta(name).get_data().flatten()
    zeta_answer = -40.0 * head
    assert np.allclose(zeta, zeta_answer)

    # absolute imbalance relative to the peak flow
    df = pd.read_csv(ws / f"output/{name}.bud.csv")
    imbalance = (df["TOTAL_IN"] - df["TOTAL_OUT"]).abs().max()
    peak = max(df["TOTAL_IN"].max(), df["TOTAL_OUT"].max())
    tol = 1.0e-6 + 1.0e-4 * peak
    assert imbalance < tol
    print(
        f"{name}: maximum zeta difference "
        f"{np.abs(zeta - zeta_answer).max():.2e} m, "
        f"maximum budget imbalance {imbalance:.2e} m3/d"
    )

# %% [markdown]
# ## Plot the results

# %%
fig, axes = plt.subplots(
    len(cases), 1, figsize=(8, 8), sharex=True, constrained_layout=True
)
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(0.0, perlen)
for (name, title), ax in zip(cases.items(), axes):
    gwf = sims[name].get_model()
    x = gwf.modelgrid.xcellcenters.flatten()
    head = gwf.output.head().get_data().flatten()
    zobj = read_zeta(name)
    flopy.plot.PlotCrossSection(gwf, line={"row": 0}, ax=ax)
    for t in zobj.times:
        zeta = zobj.get_data(totim=t).flatten()
        ax.plot(x, zeta, color=cmap(norm(t)), lw=1)
    ax.plot(x, head, "b-", label="final head")
    ax.set_ylim(-400, 50)
    ax.set_title(f"{name} -- {title}")
    ax.set_ylabel("Elevation, in meters")
axes[0].legend(loc="lower center")
axes[-1].set_xlabel("Distance, in meters")
fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes,
    label="Time of interface, in days",
)
plt.show()
