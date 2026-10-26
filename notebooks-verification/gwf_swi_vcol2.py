# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Vertical column with two fluids
#
# A confined, five-layer column is simulated with coupled freshwater and
# saltwater models connected by an SWI6-SWI6 exchange. A freshwater well in the
# top cell injects for 20 days and then extracts for 20 days, and a constant
# saltwater head in the bottom cell lets saltwater leave and re-enter the
# column. The interface moves down through two layer faces and back. Unlike
# the single-fluid case, vertical flow of each fluid is restricted by buoyancy
# and controlled by the interface position at each layer face.
#
# Ported from `autotest/test_gwf_swi_vcol2.py` on the `feat-swi` branch of
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

name = "swi-vcol2"

nlay, nrow, ncol = 5, 1, 1
delr = delc = 1.0
top = 0.0
botm = [-1.0, -2.0, -3.0, -4.0, -5.0]
k = 100.0  # meters per day
sy = 0.2
alphaf = 1000.0 / (1025.0 - 1000.0)  # 40.0
alphas = 1025.0 / (1025.0 - 1000.0)  # 41.0

nstp = 20  # time steps of 1 day in each stress period
q = 0.02  # injection and extraction rate, in cubic meters per day
zeta0 = -0.5  # initial interface elevation, in meters

# %% [markdown]
# ## Build and run the model


# %%
def build_gwf(sim, is_salt):
    mname = "saltwater" if is_salt else "freshwater"
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
    # heads that place the interface at zeta0
    strt = 0.0 if is_salt else -zeta0 / alphaf
    flopy.mf6.ModflowGwfic(gwf, strt=strt)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=k, save_specific_discharge=True)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.0, sy=sy, transient=True)
    flopy.mf6.ModflowGwfswi(gwf, zeta_filerecord=f"output/{mname}.zta")
    if is_salt:
        # constant saltwater head in the bottom cell
        flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(nlay - 1, 0, 0), 0.0]])
    else:
        # inject, then extract, in the top cell
        flopy.mf6.ModflowGwfwel(
            gwf, stress_period_data={0: [[(0, 0, 0), q]], 1: [[(0, 0, 0), -q]]}
        )
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"output/{mname}.hds",
        budgetcsv_filerecord=f"output/{mname}.bud.csv",
        saverecord=[("HEAD", "ALL")],
    )
    return gwf


def build_model(name):
    ws = model_root / name
    (ws / "output").mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(
        sim, nper=2, perioddata=[(nstp, nstp, 1.0), (nstp, nstp, 1.0)]
    )
    ims = flopy.mf6.ModflowIms(
        sim,
        no_ptcrecord=True,
        under_relaxation="DBD",
        under_relaxation_gamma=0.1,
        under_relaxation_theta=0.7,
        under_relaxation_kappa=0.07,
        under_relaxation_momentum=0.0,
        outer_maximum=500,
        inner_maximum=600,
        outer_dvclose=1.0e-7,
        inner_dvclose=1.0e-8,
        linear_acceleration="bicgstab",
    )
    build_gwf(sim, False)
    build_gwf(sim, True)
    flopy.mf6.ModflowSwiswi(
        sim,
        exgtype="SWI6-SWI6",
        exgmnamea="freshwater",
        exgmnameb="saltwater",
    )
    sim.register_ims_package(ims, ["freshwater", "saltwater"])
    return sim


sim = build_model(name)
sim.write_simulation(silent=True)
success, _ = sim.run_simulation(silent=True)
assert success, f"{name} did not run"

# %% [markdown]
# ## Check the results
#
# The interface elevation, zeta = -alphaf hf + alphas hs, is computed from the
# freshwater and saltwater heads. The freshwater and saltwater models must
# write the same zeta, the interface must stay flat, descend while injecting,
# cross the -1 and -2 m layer faces, rise while extracting, and return near
# its initial elevation. The descent over the injection period must be within
# 5 percent of Q t / (SY A), which tests the buoyancy-restricted vertical
# saltwater flow rate. Both budgets must balance.

# %%
ws = Path(sim.sim_path)
hf = flopy.utils.HeadFile(ws / "output/freshwater.hds")
hs = flopy.utils.HeadFile(ws / "output/saltwater.hds")
times = np.array(hf.times)
z = np.array(
    [
        -alphaf * hf.get_data(totim=t).flatten()
        + alphas * hs.get_data(totim=t).flatten()
        for t in times
    ]
)
zeta = z[:, 0]


def zeta_output(mname):
    zobj = flopy.utils.HeadFile(ws / f"output/{mname}.zta", text="zeta")
    return np.array([zobj.get_data(totim=t).flatten() for t in zobj.times])


zof, zos = zeta_output("freshwater"), zeta_output("saltwater")
assert np.allclose(zof, zos, atol=1.0e-6)

spread = float(np.ptp(z, axis=1).max())
assert spread < 0.1

zeta_p1, zeta_p2 = zeta[:nstp], zeta[nstp:]
assert np.all(np.diff(zeta_p1) < 0.0)
assert np.all(np.diff(zeta_p2) > 0.0)
assert zeta_p1.min() < -2.0
assert abs(zeta[-1] - zeta0) < 0.1

perlen = float(nstp)
descent = zeta0 - float(zeta_p1[-1])
descent_expected = q / (delr * delc * sy) * perlen
assert abs(descent - descent_expected) < 0.05 * descent_expected

print(f"maximum fresh and salt zeta difference {np.abs(zof - zos).max():.1e} m")
print(f"maximum interface spread across the column {spread:.4f} m")
print(f"descent {descent:.4f} m, expected {descent_expected:.4f} m")
print(f"final interface {zeta[-1]:.4f} m")
for mname in ("freshwater", "saltwater"):
    df = pd.read_csv(ws / f"output/{mname}.bud.csv")
    imbalance = (df["TOTAL_IN"] - df["TOTAL_OUT"]).abs().max()
    peak = max(df["TOTAL_IN"].max(), df["TOTAL_OUT"].max())
    assert imbalance < 1.0e-6 + 1.0e-4 * peak
    print(f"{mname}: maximum budget imbalance {imbalance:.3e}")

# %% [markdown]
# ## Plot the results
#
# The top row shows the freshwater (blue) and saltwater (red) in each cell
# from the zeta output; the bottom panel shows the interface elevation with
# time.

# %%
zobj = flopy.utils.HeadFile(ws / "output/freshwater.zta", text="zeta")
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
    zeta0 - q / (sy * delr * delc) * np.minimum(times, 2 * nstp - times),
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
