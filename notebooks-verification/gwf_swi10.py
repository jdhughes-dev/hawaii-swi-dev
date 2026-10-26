# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Short growing and shrinking freshwater bubble with coupled models
#
# A one-layer, 21-column cross section with uniform 100-m columns simulated with separate freshwater and
# saltwater models coupled by the SWI-SWI exchange. Recharge in the first
# stress period forms a freshwater bubble that shrinks in the second stress
# period without recharge. The interface is checked against the Ghyben-Herzberg
# relation for the freshwater and saltwater heads for confined and unconfined
# conditions. The problem is a shorter version of `gwf_swi08`, set up so the
# coefficient matrix of the coupled models is small enough to be extracted.
#
# Ported from `autotest/test_gwf_swi10.py` on the `feat-swi` branch of
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
    "swi10a-conf": "Confined",
    "swi10b-unconf": "Unconfined",
}
top = {"swi10a-conf": 0.0, "swi10b-unconf": 10.0}
icelltype = {"swi10a-conf": 0, "swi10b-unconf": 1}
iconvert = {"swi10a-conf": 0, "swi10b-unconf": 1}

nlay, nrow, ncol = 1, 1, 21
delr = 100.0
delc = 1.0
botm = -80.0
recharge = {0: 0.0075, 1: 0.0}  # meters per day
k_fw = 10.0  # meters per day
k_sw = 10.0  # meters per day
h0 = 0.0  # sea level, in meters
ss = 0.0
sy = 0.2

# period length, in days, number of time steps, and multiplier
perioddata = [(10000.0, 10, 1.0), (10000.0, 10, 1.0)]

# %% [markdown]
# ## Build and run the models


# %%
def build_gwf_model(name, sim, is_saltwater):
    mname = "saltwater" if is_saltwater else "freshwater"
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=mname, save_flows=True, newtonoptions="newton"
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
    flopy.mf6.ModflowGwfic(gwf, strt=0.0 if is_saltwater else 0.001)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_specific_discharge=True,
        save_saturation=True,
        icelltype=icelltype[name],
        k=k_sw if is_saltwater else k_fw,
    )
    flopy.mf6.ModflowGwfsto(gwf, iconvert=iconvert[name], ss=ss, sy=sy)
    flopy.mf6.ModflowGwfswi(gwf, zeta_filerecord=f"output/{mname}.zta")
    flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data=[[0, 0, 0, h0], [0, 0, ncol - 1, h0]]
    )
    if not is_saltwater:
        flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"output/{mname}.cbc",
        budgetcsv_filerecord=f"output/{mname}.bud.csv",
        head_filerecord=f"output/{mname}.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return gwf


def build_model(name):
    ws = model_root / name
    (ws / "output").mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim, nper=len(perioddata), perioddata=perioddata)
    ims = flopy.mf6.ModflowIms(
        sim,
        no_ptcrecord=True,
        under_relaxation="dbd",
        under_relaxation_gamma=0.1,
        under_relaxation_theta=0.7,
        under_relaxation_kappa=0.07,
        under_relaxation_momentum=0.0,
        outer_maximum=500,
        inner_maximum=600,
        outer_dvclose=1e-7,
        inner_dvclose=1e-8,
        linear_acceleration="bicgstab",
    )
    gwf_fresh = build_gwf_model(name, sim, False)
    gwf_salt = build_gwf_model(name, sim, True)
    flopy.mf6.ModflowSwiswi(
        sim,
        exgtype="SWI6-SWI6",
        exgmnamea="freshwater",
        exgmnameb="saltwater",
    )
    sim.register_ims_package(ims, [gwf_fresh.name, gwf_salt.name])
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
# The interface in both models at the end of the simulation must equal
# $\zeta = -\alpha_f h_f + \alpha_s h_s$, with $\alpha_f = 40$ and
# $\alpha_s = 41$ for freshwater and saltwater densities of 1,000 and
# 1,025 kg/m$^3$. Both model budgets must also conserve mass.

# %%
alphaf = 1000.0 / (1025.0 - 1000.0)
alphas = 1025.0 / (1025.0 - 1000.0)


def zeta_file(name, mname):
    ws = Path(sims[name].sim_path)
    return flopy.utils.HeadFile(ws / f"output/{mname}.zta", text="zeta")


for name in cases:
    sim = sims[name]
    ws = Path(sim.sim_path)
    head_f = sim.get_model("freshwater").output.head().get_data().flatten()
    head_s = sim.get_model("saltwater").output.head().get_data().flatten()
    zeta = zeta_file(name, "freshwater").get_data().flatten()
    zeta_s = zeta_file(name, "saltwater").get_data().flatten()
    zeta_answer = -alphaf * head_f + alphas * head_s
    assert np.allclose(zeta, zeta_answer), f"{name}: zeta is not right"
    assert np.allclose(zeta, zeta_s), f"{name}: saltwater and freshwater zeta differ"
    print(f"{name}: maximum zeta error {np.abs(zeta - zeta_answer).max():.2e} m")

    # absolute imbalance relative to the peak flow
    for mname in ("freshwater", "saltwater"):
        df = pd.read_csv(ws / f"output/{mname}.bud.csv")
        imbalance = (df["TOTAL_IN"] - df["TOTAL_OUT"]).abs().max()
        peak = max(df["TOTAL_IN"].max(), df["TOTAL_OUT"].max())
        tol = 1.0e-6 + 1.0e-4 * peak
        assert imbalance < tol, f"{name} {mname} budget does not conserve mass"
        print(f"  {mname}: maximum imbalance {imbalance:.2e} (tolerance {tol:.2e})")

# %% [markdown]
# ## Plot the results
#
# The interface and freshwater head at selected times in the first stress
# period (recharge, solid lines) and the second (no recharge, dashed lines).

# %%
fig, axes = plt.subplots(len(cases), 2, figsize=(10, 7), constrained_layout=True)
for (name, title), (ax_z, ax_h) in zip(cases.items(), axes):
    sim = sims[name]
    gwf = sim.get_model("freshwater")
    x = gwf.modelgrid.xcellcenters.flatten()
    zobj = zeta_file(name, "freshwater")
    hobj = gwf.output.head()
    times = zobj.times
    nstp1 = perioddata[0][1]
    plot_times = [times[i] for i in (0, 1, 2, 4, 9, 10, 11, 14, 19)]
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_times)))

    flopy.plot.PlotCrossSection(gwf, line={"row": 0}, ax=ax_z)
    for t, c in zip(plot_times, colors):
        ls = "-" if t <= times[nstp1 - 1] else "--"
        label = f"{t:,.0f} days"
        ax_z.plot(x, zobj.get_data(totim=t).flatten(), ls=ls, color=c, label=label)
        ax_h.plot(x, hobj.get_data(totim=t).flatten(), ls=ls, color=c, label=label)
    ax_z.set_ylim(-100.0, 10.0)
    ax_z.set_title(f"{name} -- {title}, interface")
    ax_z.set_ylabel("Elevation, in meters")
    ax_h.set_title(f"{name} -- {title}, freshwater head")
    ax_h.set_ylabel("Head, in meters")
axes[0, 1].legend(fontsize=8, ncol=2)
axes[-1, 0].set_xlabel("Distance, in meters")
axes[-1, 1].set_xlabel("Distance, in meters")
plt.show()
