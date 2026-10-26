# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Descending freshwater front without premature breakthrough
#
# A 10-layer vertical column simulated with coupled freshwater and saltwater
# models. Recharge on top of the freshwater model pushes a sharp freshwater
# front down through the column while saltwater leaves through the bottom.
# Freshwater must not enter a cell before the interface in the overlying cell
# reaches that cell's bottom; the vertical buoyancy restriction in the SWI
# Package prevents this premature breakthrough.
#
# Ported from `autotest/test_gwf_swi_breakthrough.py` on the `feat-swi` branch of
# [christianlangevin/modflow6](https://github.com/christianlangevin/modflow6/tree/feat-swi).

# %%
from pathlib import Path

import flopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# %% [markdown]
# ## Model parameters
#
# Model files are written to `models/<case>/` beside the notebooks, and output
# to its `output/` subdirectory.

# %%
root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pixi.toml").exists())
model_root = root / "notebooks-verification" / "models"

name = "swi-breakthrough"

nlay, nrow, ncol = 10, 1, 1
delr = delc = 1.0
top = 0.0
botm = [-float(k + 1) for k in range(nlay)]  # 1-m cells
sy = 0.2
recharge = 0.02  # meters per day
alphaf = 1000.0 / (1025.0 - 1000.0)
alphas = 1025.0 / (1025.0 - 1000.0)
zeta0 = -0.5  # initial interface in the top cell, in meters

nstp = 20
perlen = 40.0  # days

cell_top = np.array([top] + botm[:-1])
cell_bot = np.array(botm)

# %% [markdown]
# ## Build and run the model


# %%
def build_gwf_model(sim, is_saltwater):
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
        top=top,
        botm=botm,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=0.0 if is_saltwater else -zeta0 / alphaf)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=1.0)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.0, sy=sy, transient=True)
    flopy.mf6.ModflowGwfswi(gwf, zeta_filerecord=f"output/{mname}.zta")
    if is_saltwater:
        # saltwater leaves through the bottom as the interface descends
        flopy.mf6.ModflowGwfghb(gwf, stress_period_data=[[nlay - 1, 0, 0, 0.0, 10.0]])
    else:
        flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"output/{mname}.hds",
        budgetcsv_filerecord=f"output/{mname}.bud.csv",
        saverecord=[("HEAD", "ALL")],
    )
    return gwf


ws = model_root / name
(ws / "output").mkdir(parents=True, exist_ok=True)
sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(perlen, nstp, 1.0)])
# the freshwater model has no head boundary and is anchored only through the
# exchange to the saltwater model
flopy.mf6.ModflowIms(
    sim,
    complexity="complex",
    outer_maximum=200,
    inner_maximum=200,
    outer_dvclose=1.0e-7,
    inner_dvclose=1.0e-8,
    linear_acceleration="bicgstab",
)
build_gwf_model(sim, False)
build_gwf_model(sim, True)
flopy.mf6.ModflowSwiswi(
    sim, exgtype="SWI6-SWI6", exgmnamea="freshwater", exgmnameb="saltwater"
)
sim.write_simulation(silent=True)
success, _ = sim.run_simulation(silent=True)
assert success, f"{name} did not run"

# %% [markdown]
# ## Check the results
#
# The freshwater saturation of each cell is computed from the interface
# $\zeta = -\alpha_f h_f + \alpha_s h_s$. A cell may hold freshwater (saturation
# above 0.02) only when the overlying cell is full (saturation above 0.95). The
# front must also descend monotonically to within 1 m of the mass-balance
# depth, $\zeta_0 - R t / S_y = -4.5$ m, and both budgets must conserve mass.

# %%
hf = flopy.utils.HeadFile(ws / "output/freshwater.hds")
hs = flopy.utils.HeadFile(ws / "output/saltwater.hds")
times = np.array(hf.times)
sf = []
for t in times:
    zeta = (
        -alphaf * hf.get_data(totim=t).flatten()
        + alphas * hs.get_data(totim=t).flatten()
    )
    sf.append(np.clip((cell_top - zeta) / (cell_top - cell_bot), 0.0, 1.0))
sf = np.array(sf)  # (ntime, nlay)

# no premature breakthrough
wet, full = 0.02, 0.95
worst = 0.0
for it, t in enumerate(times):
    for k in range(1, nlay):
        if sf[it, k] > wet and sf[it, k - 1] < full:
            worst = max(worst, sf[it, k])
            print(
                f"premature at t={t:.1f} layer {k + 1}: Sf={sf[it, k]:.3f} "
                f"while overlying Sf={sf[it, k - 1]:.3f}"
            )
assert worst == 0.0, f"premature breakthrough (maximum Sf={worst:.3f})"
print("no premature breakthrough")

# front is the bottom of the deepest cell more than half full
front = np.array(
    [cell_bot[np.where(s > 0.5)[0][-1]] if np.any(s > 0.5) else top for s in sf]
)
mb_front = zeta0 - recharge * perlen / sy
assert np.all(np.diff(front) <= 1.0e-6), "front should descend monotonically"
assert abs(front[-1] - mb_front) <= 1.0, "front should track the mass balance"
print(f"final front {front[-1]:.1f} m, mass-balance front {mb_front:.1f} m")

# absolute imbalance relative to the peak flow
for mname in ("freshwater", "saltwater"):
    df = pd.read_csv(ws / f"output/{mname}.bud.csv")
    imbalance = (df["TOTAL_IN"] - df["TOTAL_OUT"]).abs().max()
    peak = max(df["TOTAL_IN"].max(), df["TOTAL_OUT"].max())
    tol = 1.0e-6 + 1.0e-4 * peak
    assert imbalance < tol, f"{mname} budget does not conserve mass"
    print(f"{mname}: maximum imbalance {imbalance:.2e} (tolerance {tol:.2e})")

# %% [markdown]
# ## Plot the results
#
# Freshwater (blue) and saltwater (red) in each cell at nine times, and the
# interface elevation from the freshwater volume in the column compared with
# the mass-balance interface.

# %%
snap = np.linspace(0, len(times) - 1, 9).round().astype(int)
fig = plt.figure(figsize=(10, 7), constrained_layout=True)
subfigs = fig.subfigures(2, 1, height_ratios=[1.3, 1.0])
axes = subfigs[0].subplots(1, len(snap), sharey=True)
for ax, it in zip(axes, snap):
    for k in range(nlay):
        zc = cell_top[k] - sf[it, k] * (cell_top[k] - cell_bot[k])
        ax.add_patch(
            Rectangle(
                (0, cell_bot[k]),
                1.0,
                zc - cell_bot[k],
                facecolor="red",
                edgecolor="k",
                lw=0.4,
            )
        )
        ax.add_patch(
            Rectangle(
                (0, zc),
                1.0,
                cell_top[k] - zc,
                facecolor="blue",
                edgecolor="k",
                lw=0.4,
            )
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(botm[-1], top)
    ax.set_xticks([])
    ax.set_title(f"{times[it]:.0f} days", fontsize=9)
axes[0].set_ylabel("Elevation, in meters")
subfigs[0].suptitle("Freshwater (blue) and saltwater (red) in the column")

ax = subfigs[1].subplots()
zeta_volume = top - (sf * (cell_top - cell_bot)).sum(axis=1)
t = np.concatenate(([0.0], times))
ax.plot(t, zeta0 - recharge * t / sy, "k-", label="mass balance")
ax.plot(times, zeta_volume, "bo", mfc="none", label="simulated")
ax.step(
    times, front, "r--", where="post", label="front (bottom of deepest half-full cell)"
)
ax.set_xlabel("Time, in days")
ax.set_ylabel("Interface elevation, in meters")
ax.set_xlim(0, perlen)
ax.legend()
plt.show()
