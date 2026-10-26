# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Well flow reduction with the SWI Package
#
# With SWI, the WEL AUTO_FLOW_REDUCE taper is applied over the part of the
# cell occupied by the simulated fluid: a freshwater model raises the bottom of
# the interval to max(cell bottom, zeta) and a saltwater model lowers the top
# to min(cell top, zeta). The reduction then also applies to confined cells,
# because the fluid zone can vanish as the interface moves through the cell.
# Four cases compare the simulated well rate with the analytical taper:
#
# 1. `swi-welafr01`: single fluid, unconfined two-cell drawdown; a model
#    without SWI shows that the cell-bottom taper stays inactive.
# 2. `swi-welafr02`: two fluids, freshwater well at the top of a column as the
#    interface rises.
# 3. `swi-welafr03`: single fluid, confined deep cell with the head far above
#    the cell top; the taper acts on (cell top - zeta).
# 4. `swi-welafr04`: two fluids, saltwater well at the bottom of a column; the
#    taper acts on (zeta - cell bottom) as the saltwater zone drains.
#
# A fifth case, `swi-welafr05`, is expected to fail: an out-of-range
# AUTO_FLOW_REDUCE_AUXNAME value for a well in a confined cell must be caught by
# the WEL input check, which is active because SWI makes confined cells
# reducible.
#
# Ported from `autotest/test_gwf_swi_wel_afr.py` on the `feat-swi` branch of
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
    "swi-welafr01": "Single fluid, unconfined",
    "swi-welafr02": "Two fluids, freshwater well",
    "swi-welafr03": "Single fluid, confined",
    "swi-welafr04": "Two fluids, saltwater well",
    "swi-welafr05": "Invalid AUTO_FLOW_REDUCE_AUXNAME value",
}
xfail = ["swi-welafr05"]

alphaf = 1000.0 / (1025.0 - 1000.0)  # 40.0
alphas = 1025.0 / (1025.0 - 1000.0)  # 41.0
afr = 0.1  # AUTO_FLOW_REDUCE fraction of cell thickness

# single fluid, two cells: constant head supply and a well
sf = {
    # unconfined, interface at the cell bottom at the supply head
    "swi-welafr01": {"top": 1.0, "bot": -40.0, "icelltype": 1, "hchd": 1.0, "q": -1.0},
    # confined, interface enters the cell from below
    "swi-welafr03": {
        "top": -30.0,
        "bot": -40.0,
        "icelltype": 0,
        "hchd": 1.0,
        "q": -0.1,
    },
    # confined, invalid per-well reduction value
    "swi-welafr05": {
        "top": -30.0,
        "bot": -40.0,
        "icelltype": 0,
        "hchd": 1.0,
        "q": -0.1,
        "afraux": 0.0,
    },
}
sf_k = 0.01  # meters per day

# two fluids, five-layer column
nlay = 5
tf_top = 0.05
tf_botm = [-1.0, -2.0, -3.0, -4.0, -5.0]
tf_sy = 0.2
zeta0 = -0.5  # initial interface elevation, in meters
tf = {
    # freshwater well in the top cell, saltwater constant head in the bottom cell
    "swi-welafr02": {"saltwell": False, "q": -0.02, "nstp": 20},
    # saltwater well in the bottom cell, freshwater constant head in the top cell
    "swi-welafr04": {"saltwell": True, "q": -0.05, "nstp": 30},
}

# %% [markdown]
# ## Build and run the models
#
# The single-fluid cases with valid input are also run without SWI in
# `models/<case>-noswi/`.


# %%
def dbd_ims(sim):
    return flopy.mf6.ModflowIms(
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


def build_sim_sf(name, p, with_swi=True):
    ws = model_root / (name if with_swi else f"{name}-noswi")
    (ws / "output").mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(400.0, 40, 1.0)])
    dbd_ims(sim)
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=name, save_flows=True, newtonoptions="newton"
    )
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=1,
        ncol=2,
        delr=1.0,
        delc=1.0,
        top=p["top"],
        botm=p["bot"],
    )
    flopy.mf6.ModflowGwfic(gwf, strt=p["hchd"])
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=p["icelltype"], k=sf_k)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.0, sy=0.2, transient=True)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 0), p["hchd"]]])
    wel_kwargs = {"auto_flow_reduce": afr}
    spd = [[(0, 0, 1), p["q"]]]
    if "afraux" in p:
        wel_kwargs["auxiliary"] = ["afr"]
        wel_kwargs["auto_flow_reduce_auxname"] = "afr"
        spd = [[(0, 0, 1), p["q"], p["afraux"]]]
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=spd, **wel_kwargs)
    if with_swi:
        flopy.mf6.ModflowGwfswi(gwf, zeta_filerecord=f"output/{name}.zta")
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"output/{name}.hds",
        budget_filerecord=f"output/{name}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return sim


def build_gwf_tf(sim, is_salt, p):
    mname = "saltwater" if is_salt else "freshwater"
    has_well = is_salt == p["saltwell"]
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=mname, save_flows=True, newtonoptions="newton"
    )
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=1,
        ncol=1,
        delr=1.0,
        delc=1.0,
        top=tf_top,
        botm=tf_botm,
    )
    strt = 0.0 if is_salt else -zeta0 / alphaf
    flopy.mf6.ModflowGwfic(gwf, strt=strt)
    # the saltwater model is confined
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0 if is_salt else 1, k=100.0)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.0, sy=tf_sy, transient=True)
    flopy.mf6.ModflowGwfswi(gwf, zeta_filerecord=f"output/{mname}.zta")
    cell = (nlay - 1, 0, 0) if is_salt else (0, 0, 0)
    if has_well:
        flopy.mf6.ModflowGwfwel(
            gwf, auto_flow_reduce=afr, stress_period_data=[[cell, p["q"]]]
        )
    else:
        flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[cell, strt]])
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"output/{mname}.hds",
        budget_filerecord=f"output/{mname}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return gwf


def build_sim_tf(name, p):
    ws = model_root / name
    (ws / "output").mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(p["nstp"], p["nstp"], 1.0)])
    ims = dbd_ims(sim)
    build_gwf_tf(sim, False, p)
    build_gwf_tf(sim, True, p)
    flopy.mf6.ModflowSwiswi(
        sim,
        exgtype="SWI6-SWI6",
        exgmnamea="freshwater",
        exgmnameb="saltwater",
    )
    sim.register_ims_package(ims, ["freshwater", "saltwater"])
    return sim


sims, sims_noswi = {}, {}
for name in cases:
    if name in sf:
        sim = build_sim_sf(name, sf[name])
    else:
        sim = build_sim_tf(name, tf[name])
    sim.write_simulation(silent=True)
    success, _ = sim.run_simulation(silent=True)
    if name in xfail:
        assert not success, f"{name} should fail"
        continue
    assert success, f"{name} did not run"
    sims[name] = sim
    if name in sf:
        sim = build_sim_sf(name, sf[name], with_swi=False)
        sim.write_simulation(silent=True)
        success, _ = sim.run_simulation(silent=True)
        assert success, f"{name}-noswi did not run"
        sims_noswi[name] = sim

# %% [markdown]
# ## Analytical taper
#
# The WEL reduction uses the MODFLOW 6 quadratic smoothing function over a band
# of `afr` times the cell thickness above the bottom of the flow-reduction
# interval, evaluated at min(h, top of the interval).


# %%
def sq_saturation(top, bot, x):
    b = top - bot
    s = (x - bot) / b
    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    w = x - bot
    return -2.0 / b**3 * w**3 + 3.0 / b**2 * w**2


def expected_qmult(h, cell_top, cell_bot, ict, botred, topred):
    if not (ict != 0 or botred > cell_bot or topred < cell_top):
        return 1.0
    band = afr * (cell_top - cell_bot)
    if band <= 0.0:
        return 1.0
    return sq_saturation(botred + band, botred, min(h, topred))


# %% [markdown]
# ## Check the results
#
# Single-fluid cases: at the end of the simulation the interface must be in the
# well cell, the confined cell must stay pressurized, and the well rate must be
# reduced and match the analytical taper. Without SWI the rate must not be
# reduced.
#
# Two-fluid cases: at every time step the well rate must match the analytical
# taper for the current interface or for the interface at the previous step,
# because the reduction interval can lag the converged interface by one step.
# By the end the well must be throttled to less than half its requested rate.
#
# The expected failure must report the AUTO_FLOW_REDUCE_AUXNAME error in
# `mfsim.lst`.

# %%
results = {}


def single_fluid_rates(name):
    p = sf[name]
    gwf = sims[name].get_model()
    hobj, bud = gwf.output.head(), gwf.output.budget()
    rows = []
    for t in hobj.times:
        hf = float(hobj.get_data(totim=t)[0, 0, 1])
        q_sim = float(bud.get_data(text="WEL", totim=t)[0]["q"][0])
        zeta = -alphaf * hf
        botred = max(p["bot"], zeta)
        q_exp = p["q"] * expected_qmult(
            hf, p["top"], p["bot"], p["icelltype"], botred, p["top"]
        )
        rows.append((t, hf, zeta, q_sim, q_exp))
    return np.array(rows)


for name, p in sf.items():
    if name in xfail:
        continue
    r = single_fluid_rates(name)
    t, hf, zeta, q_sim, q_exp = r[-1]
    assert zeta > p["bot"] + 0.5
    if p["icelltype"] == 0:
        assert hf > p["top"] + 1.0
    assert abs(q_sim) < 0.9 * abs(p["q"])
    assert abs(q_sim - q_exp) < 1.0e-6 + 1.0e-4 * abs(p["q"])
    bud = sims_noswi[name].get_model().output.budget()
    q2 = float(bud.get_data(text="WEL")[-1]["q"][0])
    assert abs(q2 - p["q"]) < 1.0e-6
    results[name] = r
    print(
        f"{name}: zeta {zeta:.4f} m, rate {q_sim:.6f}, analytical {q_exp:.6f}, "
        f"without SWI {q2:.6f}"
    )


def two_fluid_rates(name):
    p = tf[name]
    ws = Path(sims[name].sim_path)
    hf_obj = flopy.utils.HeadFile(ws / "output/freshwater.hds")
    hs_obj = flopy.utils.HeadFile(ws / "output/saltwater.hds")
    wellmodel = "saltwater" if p["saltwell"] else "freshwater"
    bud = sims[name].get_model(wellmodel).output.budget()
    if p["saltwell"]:
        klay, cell_top, cell_bot, ict = nlay - 1, tf_botm[-2], tf_botm[-1], 0
    else:
        klay, cell_top, cell_bot, ict = 0, tf_top, tf_botm[0], 1
    rows = []
    prev_bounds = None
    for t in hf_obj.times:
        hf = float(hf_obj.get_data(totim=t)[klay, 0, 0])
        hs = float(hs_obj.get_data(totim=t)[klay, 0, 0])
        zeta = -alphaf * hf + alphas * hs
        h_well = hs if p["saltwell"] else hf
        botred = cell_bot if p["saltwell"] else max(cell_bot, zeta)
        topred = min(cell_top, zeta) if p["saltwell"] else cell_top
        if prev_bounds is None:
            prev_bounds = (botred, topred)
        q_sim = float(bud.get_data(text="WEL", totim=t)[0]["q"][0])
        q_exp = p["q"] * expected_qmult(h_well, cell_top, cell_bot, ict, botred, topred)
        q_lag = p["q"] * expected_qmult(h_well, cell_top, cell_bot, ict, *prev_bounds)
        prev_bounds = (botred, topred)
        rows.append((t, h_well, zeta, q_sim, q_exp, q_lag))
    return np.array(rows)


for name, p in tf.items():
    r = two_fluid_rates(name)
    tol = 1.0e-6 + 1.0e-3 * abs(p["q"])
    q_sim, q_exp, q_lag = r[:, 3], r[:, 4], r[:, 5]
    lo = np.minimum(q_exp, q_lag) - tol
    hi = np.maximum(q_exp, q_lag) + tol
    assert np.all((lo <= q_sim) & (q_sim <= hi))
    assert abs(q_sim[-1]) < 0.5 * abs(p["q"])
    results[name] = r
    print(
        f"{name}: final zeta {r[-1, 2]:.4f} m, rate {q_sim[-1]:.6f} "
        f"of {p['q']}, maximum departure from taper "
        f"{np.maximum(lo + tol - q_sim, q_sim - hi + tol).max():.1e}"
    )

for name in xfail:
    lst = (model_root / name / "mfsim.lst").read_text()
    assert "AUTO_FLOW_REDUCE_AUXNAME" in lst
    print(f"{name}: failed as expected with an AUTO_FLOW_REDUCE_AUXNAME error")

# %% [markdown]
# ## Plot the results
#
# Simulated well rates (lines) are compared with the analytical taper
# (circles); the dashed line is the interface elevation in the well cell.

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
for ax, name in zip(axes.flat, sims):
    r = results[name]
    p = sf[name] if name in sf else tf[name]
    t, zeta, q_sim, q_exp = r[:, 0], r[:, 2], r[:, 3], r[:, 4]
    ax.plot(t, q_sim, "b-", label="simulated rate")
    ax.plot(t, q_exp, "bo", mfc="none", ms=4, label="analytical taper")
    ax.axhline(p["q"], color="0.5", ls=":", label="requested rate")
    if name in sims_noswi:
        bud = sims_noswi[name].get_model().output.budget()
        q2 = [float(bud.get_data(text="WEL", totim=tt)[0]["q"][0]) for tt in t]
        ax.plot(t, q2, "g-", lw=3, alpha=0.4, label="without SWI")
    ax.set_title(f"{name} -- {cases[name]}", fontsize=10)
    ax.set_ylabel("Well rate, in cubic meters per day")
    ax.set_xlabel("Time, in days")
    axz = ax.twinx()
    axz.plot(t, zeta, "r--", label="interface")
    axz.set_ylabel("Interface elevation, in meters", color="r")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axz.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
plt.show()
