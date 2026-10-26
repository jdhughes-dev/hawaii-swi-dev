# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Vertical column with a single fluid
#
# A closed, confined, five-layer column with a freshwater well in the top cell
# injects for 20 days and then extracts at the same rate for 20 days. With a
# drainable interface storage (SY) the flat interface moves at dz/dt = Q / (SY
# A), 0.1 m per day, down through two layer faces and back to its starting
# elevation. In the single-fluid case vertical flow is not restricted by
# buoyancy.
#
# Ported from `autotest/test_gwf_swi_vcol.py` on the `feat-swi` branch of
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

name = "swi-vcol"

nlay, nrow, ncol = 5, 1, 1
delr = delc = 1.0
top = 0.0
botm = [-1.0, -2.0, -3.0, -4.0, -5.0]
k = 100.0  # meters per day
sy = 0.2
alphaf = 1000.0 / (1025.0 - 1000.0)  # 40.0

nstp = 20  # time steps of 1 day in each stress period
q = 0.02  # injection and extraction rate, in cubic meters per day
zeta0 = -0.5  # initial interface elevation, in meters
dz_expected = q / (sy * delr * delc)  # interface movement, in meters per day

# %% [markdown]
# ## Build and run the model


# %%
def build_model(name):
    ws = model_root / name
    (ws / "output").mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(
        sim, nper=2, perioddata=[(nstp, nstp, 1.0), (nstp, nstp, 1.0)]
    )
    flopy.mf6.ModflowIms(
        sim,
        outer_maximum=100,
        inner_maximum=100,
        outer_dvclose=1.0e-8,
        inner_dvclose=1.0e-9,
        linear_acceleration="bicgstab",
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
    # freshwater head that places the interface at zeta0
    flopy.mf6.ModflowGwfic(gwf, strt=-zeta0 / alphaf)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=k)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.0, sy=sy, transient=True)
    flopy.mf6.ModflowGwfswi(gwf, zeta_filerecord=f"output/{name}.zta")
    # inject, then extract, in the top cell
    flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data={0: [[(0, 0, 0), q]], 1: [[(0, 0, 0), -q]]},
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"output/{name}.hds",
        budgetcsv_filerecord=f"output/{name}.bud.csv",
        saverecord=[("HEAD", "ALL")],
    )
    return sim


sim = build_model(name)
sim.write_simulation(silent=True)
success, _ = sim.run_simulation(silent=True)
assert success, f"{name} did not run"

# %% [markdown]
# ## Check the results
#
# The interface elevation in each cell, zeta = -alphaf hf, is computed from
# the freshwater head. The interface must stay flat, descend while injecting
# at 0.1 m per day on average, cross the -1 and -2 m layer faces, rise while
# extracting, and return to its initial elevation. The budget must balance.

# %%
ws = Path(sim.sim_path)
hobj = sim.get_model().output.head()
times = np.array(hobj.times)
z = np.array([-alphaf * hobj.get_data(totim=t).flatten() for t in times])
zeta = z[:, 0]

spread = float(np.ptp(z, axis=1).max())
assert spread < 0.1

zeta_p1, zeta_p2 = zeta[:nstp], zeta[nstp:]
assert np.all(np.diff(zeta_p1) < 0.0)
assert np.all(np.diff(zeta_p2) > 0.0)
assert abs(np.diff(zeta_p1).mean() + dz_expected) < 1.0e-3
assert zeta_p1.min() < -2.0
assert abs(zeta[-1] - zeta0) < 1.0e-3

df = pd.read_csv(ws / f"output/{name}.bud.csv")
imbalance = (df["TOTAL_IN"] - df["TOTAL_OUT"]).abs().max()
peak = max(df["TOTAL_IN"].max(), df["TOTAL_OUT"].max())
assert imbalance < 1.0e-6 + 1.0e-4 * peak

print(f"maximum interface spread across the column {spread:.4f} m")
print(f"mean descent {-np.diff(zeta_p1).mean():.4f} m per day")
print(f"lowest interface {zeta_p1.min():.4f} m, final interface {zeta[-1]:.4f} m")
print(f"maximum budget imbalance {imbalance:.3e}")

# %% [markdown]
# ## Plot the results
#
# The top row shows the freshwater (blue) and saltwater (red) in each cell
# from the zeta output; the bottom panel shows the interface elevation with
# time.

# %%
zobj = flopy.utils.HeadFile(ws / f"output/{name}.zta", text="zeta")
ztimes = zobj.times
cell_top = np.array([top] + botm[:-1])
cell_bot = np.array(botm)
snap = np.linspace(0, len(ztimes) - 1, 9).round().astype(int)

fig = plt.figure(figsize=(10, 6.5), constrained_layout=True)
subfigs = fig.subfigures(2, 1, height_ratios=[1.1, 1])
axes = subfigs[0].subplots(1, len(snap), sharey=True)
for ax, ti in zip(axes, snap):
    zc = zobj.get_data(totim=ztimes[ti]).flatten()
    for kk in range(nlay):
        zk = min(max(zc[kk], cell_bot[kk]), cell_top[kk])
        ax.add_patch(
            Rectangle(
                (0, cell_bot[kk]), 1.0, zk - cell_bot[kk], fc="red", ec="k", lw=0.5
            )
        )
        ax.add_patch(
            Rectangle((0, zk), 1.0, cell_top[kk] - zk, fc="blue", ec="k", lw=0.5)
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(botm[-1], top)
    ax.set_xticks([])
    ax.set_title(f"{ztimes[ti]:.0f} d", fontsize=9)
axes[0].set_ylabel("Elevation, in meters")
subfigs[0].suptitle(f"{name} -- freshwater (blue) and saltwater (red)")

ax = subfigs[1].subplots()
for zf in botm[:-1]:
    ax.axhline(zf, color="0.7", lw=0.75)
ax.axvline(nstp, color="0.5", ls=":", lw=1)
ax.plot(times, zeta, "o-", mfc="none", ms=4, label="simulated")
ax.plot(
    times,
    zeta0 - dz_expected * np.minimum(times, 2 * nstp - times),
    "k--",
    lw=1,
    label="Q / (SY A)",
)
ax.text(nstp / 2, -3.5, "injection", ha="center")
ax.text(1.5 * nstp, -3.5, "extraction", ha="center")
ax.set_ylim(botm[-1], top)
ax.set_xlabel("Time, in days")
ax.set_ylabel("Interface elevation, in meters")
ax.legend(loc="lower right")
plt.show()
