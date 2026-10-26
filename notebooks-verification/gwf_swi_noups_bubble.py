# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Conductance weighting for a growing and shrinking bubble
#
# A one-layer, 21-column, unconfined strip island that is initially all
# saltwater. Recharge in the first stress period grows a freshwater bubble
# until the interface reaches the aquifer bottom and the lens is steady; the
# recharge is removed in the second period and the bubble shrinks. Freshwater
# and saltwater are separate GWF models coupled by a SWI-SWI exchange, solved
# with and without upstream weighting of the fluid-zone conductance.
#
# Ported from `autotest/test_gwf_swi_noups_bubble.py` on the `feat-swi` branch of
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
    "bubble-upstream": "Upstream weighting (default)",
    "bubble-amtlmk": "No upstream weighting, AMT-LMK",
}
no_upstream = {"bubble-upstream": False, "bubble-amtlmk": True, "bubble-hmean": True}
cell_averaging = {
    "bubble-upstream": None,
    "bubble-amtlmk": "AMT-LMK",
    "bubble-hmean": None,
}

ncol = 21
delr, delc = 100.0, 1.0
top, botm = 10.0, -80.0
k = 10.0  # meters per day
recharge = {0: 0.0075, 1: 0.0}  # meters per day
sy = 0.2
alphaf, alphas = 40.0, 41.0
perioddata = [(80000.0, 100, 1.0), (10000.0, 10, 1.0)]
nstp1 = perioddata[0][1]

# %% [markdown]
# ## Build and run the models
#
# All cases use DBD under-relaxation with backtracking; without backtracking
# the AMT-LMK case diverges when the saltwater zone pinches out at the bottom.


# %%
def build_gwf(sim, is_salt, noups, avg):
    name = "saltwater" if is_salt else "freshwater"
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=name, save_flows=True, newtonoptions="newton"
    )
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=1, ncol=ncol, delr=delr, delc=delc, top=top, botm=botm
    )
    flopy.mf6.ModflowGwfic(gwf, strt=0.0 if is_salt else 0.001)
    npf_kwargs = {}
    if avg is not None:
        npf_kwargs["alternative_cell_averaging"] = avg
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=k, **npf_kwargs)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=1, ss=0.0, sy=sy)
    swi_kwargs = {}
    if noups:
        swi_kwargs["dev_no_upstream_weighting"] = True
    flopy.mf6.ModflowGwfswi(gwf, zeta_filerecord=f"output/{name}.zta", **swi_kwargs)
    flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data=[[0, 0, 0, 0.0], [0, 0, ncol - 1, 0.0]]
    )
    if not is_salt:
        flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"output/{name}.cbc",
        budgetcsv_filerecord=f"output/{name}.bud.csv",
        head_filerecord=f"output/{name}.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return gwf


def build_model(name):
    ws = model_root / name
    (ws / "output").mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim, nper=2, perioddata=perioddata)
    ims = flopy.mf6.ModflowIms(
        sim,
        csv_outer_output_filerecord="output/outer.csv",
        no_ptcrecord=True,
        under_relaxation="DBD",
        under_relaxation_gamma=0.1,
        under_relaxation_theta=0.7,
        under_relaxation_kappa=0.07,
        under_relaxation_momentum=0.0,
        outer_maximum=500,
        inner_maximum=600,
        outer_dvclose=1e-7,
        inner_dvclose=1e-8,
        backtracking_number=20,
        backtracking_tolerance=1.05,
        backtracking_reduction_factor=0.1,
        backtracking_residual_limit=0.002,
        linear_acceleration="bicgstab",
    )
    fresh = build_gwf(sim, False, no_upstream[name], cell_averaging[name])
    salt = build_gwf(sim, True, no_upstream[name], cell_averaging[name])
    flopy.mf6.ModflowSwiswi(
        sim, exgtype="SWI6-SWI6", exgmnamea=fresh.name, exgmnameb=salt.name
    )
    sim.register_ims_package(ims, [fresh.name, salt.name])
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
# Both cases must satisfy Ghyben-Herzberg, give the same interface in both
# models, and conserve mass for both fluids. At the end of period 1 the lens is
# steady with the interface on the aquifer bottom over the interior. In the
# near-shore cell the AMT-LMK interface must be deeper than the upstream one,
# that is, closer to the Dupuit position: upstream weighting takes the thick
# interior cell's freshwater fraction for the shoreward face, which overstates
# the conductance and pulls the interface up.


# %%
def read_results(ws):
    ws = Path(ws) / "output"
    zobj = flopy.utils.HeadFile(ws / "freshwater.zta", text="zeta")
    zeta = np.array([zobj.get_data(totim=t).flatten() for t in zobj.times])
    hf = flopy.utils.HeadFile(ws / "freshwater.hds").get_data().flatten()
    hs = flopy.utils.HeadFile(ws / "saltwater.hds").get_data().flatten()
    zs = flopy.utils.HeadFile(ws / "saltwater.zta", text="zeta").get_data()
    nouter = len(pd.read_csv(ws / "outer.csv"))
    return np.array(zobj.times), zeta, hf, hs, zs.flatten(), nouter


results = {}
for name in cases:
    ws = Path(sims[name].sim_path)
    times, zeta, hf, hs, zs, nouter = read_results(ws)
    results[name] = times, zeta
    assert np.allclose(zeta[-1], -alphaf * hf + alphas * hs)
    assert np.allclose(zeta[-1], zs)
    assert zeta.min() >= botm - 1.0e-6 and zeta.max() <= top + 1.0e-6
    for nam in ("freshwater", "saltwater"):
        df = pd.read_csv(ws / f"output/{nam}.bud.csv")
        imbalance = (df["TOTAL_IN"] - df["TOTAL_OUT"]).abs().max()
        peak = max(df["TOTAL_IN"].max(), df["TOTAL_OUT"].max())
        tol = 1.0e-6 + 1.0e-4 * peak
        assert imbalance < tol, f"{name} {nam} budget imbalance {imbalance:.3e}"
    z_end1 = zeta[nstp1 - 1]
    assert z_end1[ncol // 2] < botm + 1.0e-3, "lens did not reach the bottom"
    assert z_end1[1] > botm + 1.0, "no toe near the shore"
    print(f"{name}: total outer iterations = {nouter}")

zeta = results["bubble-amtlmk"][1]
zeta_ref = results["bubble-upstream"][1]
dz = np.abs(zeta - zeta_ref)
print(f"max |zeta - zeta_upstream| at end of period 1 = {dz[nstp1 - 1].max():.3f} m")
print(f"max |zeta - zeta_upstream| at end of period 2 = {dz[-1].max():.3f} m")
print(
    f"near-shore zeta at end of period 1: AMT-LMK {zeta[nstp1 - 1][1]:.2f} m, "
    f"upstream {zeta_ref[nstp1 - 1][1]:.2f} m"
)
assert zeta[nstp1 - 1][1] < zeta_ref[nstp1 - 1][1] - 1.0

# %% [markdown]
# ## Expected failure of the harmonic mean
#
# The harmonic mean cannot grow a fluid zone from zero thickness, because the
# conductance into an all-saltwater cell is zero, so this case fails at the
# first time step.

# %%
name = "bubble-hmean"
sim = build_model(name)
sim.write_simulation(silent=True)
success, _ = sim.run_simulation(silent=True)
assert not success, f"{name} should fail"
print(f"{name}: failed as expected")

# %% [markdown]
# ## Plot the results

# %%
x = sims["bubble-upstream"].get_model("freshwater").modelgrid.xcellcenters.flatten()
times = results["bubble-upstream"][0]
periods = {
    "Period 1 -- recharge, bubble grows": [0, 1, 2, 4, nstp1 - 1],
    "Period 2 -- no recharge, bubble shrinks": [nstp1 + 1, nstp1 + 4, len(times) - 1],
}
styles = {"bubble-upstream": "--", "bubble-amtlmk": "-"}
fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
for ax, (title, steps) in zip(axes, periods.items()):
    for i, istp in enumerate(steps):
        color = plt.cm.viridis(i / max(len(steps) - 1, 1))
        for name in cases:
            ax.plot(x, results[name][1][istp], styles[name], color=color)
        ax.plot([], [], "-", color=color, label=f"{times[istp]:,.0f} days")
    for name, label in cases.items():
        ax.plot([], [], styles[name], color="k", label=label)
    ax.set_ylim(botm, top)
    ax.set_title(title)
    ax.set_ylabel("Elevation, in meters")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
axes[-1].set_xlabel("Distance, in meters")
plt.show()
