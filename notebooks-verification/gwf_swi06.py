# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Strip island with a rising sea level
#
# The strip island of `gwf_swi05`, run for 10 stress periods of 3,650 days
# (without storage) while sea level rises in equal steps from 0 m to 40 m. The
# saltwater head is supplied to the SWI Package through its time-varying (TVG)
# input, and the GHB heads at both ends follow the same sea level.
#
# Ported from `autotest/test_gwf_swi06.py` on the `feat-swi` branch of
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

name = "swi06a"

Lx = 10000.0  # island width, in meters
delr, delc = 100.0, 1.0
nlay, nrow = 1, 1
ncol = int(Lx / delr)
top_island = 50.0
botm_aquifer = -50.0
recharge = 0.0001  # meters per day
hydraulic_conductivity = 100.0  # meters per day
ghb_cond_fact = 1.0
nper = 10
perlen = 10 * 365.0  # days
sealevel_start, sealevel_stop = 0.0, 40.0  # meters

# sea level in each stress period
sl = np.linspace(sealevel_start, sealevel_stop, nper)

# %% [markdown]
# ## Build and run the model


# %%
def build_model(name):
    ws = model_root / name
    (ws / "output").mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=nper * [(perlen, 1, 1.0)])
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
        top=top_island,
        botm=botm_aquifer,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    flopy.mf6.ModflowGwfnpf(
        gwf, save_specific_discharge=True, icelltype=1, k=hydraulic_conductivity
    )
    swi = flopy.mf6.ModflowGwfswi(
        gwf,
        zeta_filerecord=f"output/{name}.zta",
        tvg6_filerecord=f"{name}.swi.tvg",
    )
    # saltwater head and an unused auxiliary array for each stress period
    aux = {}
    for kper in range(nper):
        aux[kper] = [sl[kper], np.ones((nlay, nrow, ncol))]
    swi.tvg.initialize(auxiliary=["saltwater_head", "myauxvar"], aux=aux)

    cghb = hydraulic_conductivity * ghb_cond_fact * delr * delc / 1
    ghbspd = {}
    for kper in range(nper):
        ghbspd[kper] = [[0, 0, j, sl[kper], cghb] for j in (0, ncol - 1)]
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=ghbspd)
    flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"output/{name}.cbc",
        budgetcsv_filerecord=f"output/{name}.bud.csv",
        head_filerecord=f"output/{name}.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return sim


sim = build_model(name)
sim.write_simulation(silent=True)
success, _ = sim.run_simulation(silent=True)
assert success, f"{name} did not run"

# %% [markdown]
# ## Check the results
#
# At every output time the interface must satisfy the Ghyben-Herzberg relation
# zeta = -40 h + 41 h_s for that period's sea level, limited to the cell top
# and bottom, and the budget must close in every period.

# %%
gwf = sim.get_model()
ws = Path(sim.sim_path)
hobj = gwf.output.head()
zobj = gwf.swi.output.zeta()
times = hobj.times
assert len(times) == nper, f"expected {nper} output times, got {len(times)}"

for i, t in enumerate(times):
    head = hobj.get_data(totim=t).flatten()
    zeta = zobj.get_data(totim=t).flatten()
    zeta_expected = np.clip(-40.0 * head + 41.0 * sl[i], botm_aquifer, top_island)
    adiff = np.abs(zeta - zeta_expected).max()
    assert np.allclose(zeta, zeta_expected, atol=1.0e-4), f"period {i + 1}"
    assert np.all(zeta >= botm_aquifer - 1.0e-6), "zeta below aquifer bottom"
    assert np.all(zeta <= top_island + 1.0e-6), "zeta above aquifer top"
    print(
        f"period {i + 1}: sea level {sl[i]:.2f} m, maximum zeta difference {adiff:.2e} m"
    )

# absolute imbalance relative to the peak flow
df = pd.read_csv(ws / f"output/{name}.bud.csv")
imbalance = (df["TOTAL_IN"] - df["TOTAL_OUT"]).abs().max()
peak = max(df["TOTAL_IN"].max(), df["TOTAL_OUT"].max())
tol = 1.0e-6 + 1.0e-4 * peak
assert imbalance < tol
print(f"{name}: maximum budget imbalance {imbalance:.2e} m3/d")

# %% [markdown]
# ## Plot the results

# %%
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
x = gwf.modelgrid.xcellcenters.flatten()
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(0.0, times[-1] / 365.0)
flopy.plot.PlotCrossSection(gwf, line={"row": 0}, ax=ax)
for t in times:
    color = cmap(norm(t / 365.0))
    ax.plot(x, zobj.get_data(totim=t).flatten(), color=color, lw=1.5)
    ax.plot(x, hobj.get_data(totim=t).flatten(), color=color, lw=1, ls="--")
ax.plot([], [], "k-", label="interface")
ax.plot([], [], "k--", label="head")
ax.legend(loc="lower center", ncol=2)
ax.set_ylim(botm_aquifer, top_island)
ax.set_title(f"{name} -- rising sea level")
ax.set_xlabel("Distance, in meters")
ax.set_ylabel("Elevation, in meters")
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Time, in years")
plt.show()
