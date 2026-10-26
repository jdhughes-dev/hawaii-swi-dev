# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Relative density difference
#
# A one-layer, confined, steady-state island cross section solved with
# different values of the SWI Package RELATIVE_DENSITY_DIFFERENCE option
# $\delta = (\rho_s - \rho_f) / \rho_f$. With a saltwater head of zero the
# interface is $\zeta = -h_f / \delta$ in every cell, which checks that the
# option reaches the formulation.
#
# Ported from `autotest/test_gwf_swi_density.py` on the `feat-swi` branch of
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
#
# A value of `None` omits the option, so the default of 0.025 is used.

# %%
root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pixi.toml").exists())
model_root = root / "notebooks-verification" / "models"

cases = {
    "swi-density-a": "Default (option omitted)",
    "swi-density-b": "$\\delta$ = 0.025",
    "swi-density-c": "$\\delta$ = 0.05",
}
delta = {"swi-density-a": None, "swi-density-b": 0.025, "swi-density-c": 0.05}

Lx = 10000.0  # island width, in meters
delr, delc = 100.0, 1.0
ncol = int(Lx / delr) + 1
nlay, nrow = 1, 1
top, botm = 0.0, -400.0
recharge = 0.001  # meters per day
k = 10.0  # meters per day
hbnd = 0.0  # sea level, in meters

# %% [markdown]
# ## Build and run the models


# %%
def build_model(name, d):
    ws = model_root / name
    (ws / "output").mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(sim, outer_maximum=500, linear_acceleration="bicgstab")
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
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
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    flopy.mf6.ModflowGwfnpf(gwf, alternative_cell_averaging="amt-hmk", icelltype=0, k=k)
    swi_kwargs = {"zeta_filerecord": f"output/{name}.zta"}
    if d is not None:
        swi_kwargs["relative_density_difference"] = d
    flopy.mf6.ModflowGwfswi(gwf, **swi_kwargs)
    cghb = 10.0 * delr * delc / 10.0
    flopy.mf6.ModflowGwfghb(
        gwf,
        stress_period_data=[[0, 0, 0, hbnd, cghb], [0, 0, ncol - 1, hbnd, cghb]],
    )
    flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"output/{name}.hds",
        saverecord=[("HEAD", "ALL")],
    )
    return sim


sims = {}
for name in cases:
    sim = build_model(name, delta[name])
    sim.write_simulation(silent=True)
    success, _ = sim.run_simulation(silent=True)
    assert success, f"{name} did not run"
    sims[name] = sim

# %% [markdown]
# ## Check the interface
#
# Case b must reproduce case a, and case c ($\alpha_f$ = 20) gives a
# different solution.

# %%
results = {}
for name in cases:
    gwf = sims[name].get_model()
    ws = Path(sims[name].sim_path)
    head = gwf.output.head().get_data().flatten()
    zeta = flopy.utils.HeadFile(ws / f"output/{name}.zta", text="zeta")
    zeta = zeta.get_data().flatten()
    d = 0.025 if delta[name] is None else delta[name]
    expected = -head / d
    assert np.allclose(zeta, expected, rtol=1.0e-9, atol=1.0e-9)
    results[name] = head, zeta
    print(
        f"{name}: delta={d}  alphaf={1.0 / d:g}  "
        f"max |zeta - (-h/delta)| = {np.abs(zeta - expected).max():.3e}"
    )

# %% [markdown]
# ## Expected input errors
#
# A relative density difference of zero (case d) is an input error, and so
# is a two-fluid simulation whose freshwater and saltwater models specify
# different values (case e); the interface is computed from both heads, so two
# values would place it differently in each model.


# %%
def print_errors(ws):
    lines = (Path(ws) / "mfsim.lst").read_text().splitlines()
    i = lines.index("ERROR REPORT:")
    for line in lines[i + 2 :]:
        if not line.strip():
            break
        print(line)


name = "swi-density-d"
sim = build_model(name, 0.0)
sim.write_simulation(silent=True)
success, _ = sim.run_simulation(silent=True)
assert not success, f"{name} should fail"
print(f"{name}: failed as expected")
print_errors(sim.sim_path)


# %%
def build_mismatched_two_fluid(name):
    ws = model_root / name
    (ws / "output").mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(sim, linear_acceleration="bicgstab")
    for nm, d in (("freshwater", 0.025), ("saltwater", 0.030)):
        gwf = flopy.mf6.ModflowGwf(
            sim, modelname=nm, save_flows=True, newtonoptions="newton"
        )
        flopy.mf6.ModflowGwfdis(
            gwf, nlay=1, nrow=1, ncol=5, delr=delr, delc=delc, top=top, botm=botm
        )
        flopy.mf6.ModflowGwfic(gwf, strt=0.0)
        flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=k)
        flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.0, sy=0.2, transient=True)
        flopy.mf6.ModflowGwfswi(gwf, relative_density_difference=d)
        flopy.mf6.ModflowGwfoc(
            gwf, head_filerecord=f"output/{nm}.hds", saverecord=[("HEAD", "ALL")]
        )
    flopy.mf6.ModflowSwiswi(
        sim, exgtype="SWI6-SWI6", exgmnamea="freshwater", exgmnameb="saltwater"
    )
    return sim


name = "swi-density-e"
sim = build_mismatched_two_fluid(name)
sim.write_simulation(silent=True)
success, _ = sim.run_simulation(silent=True)
assert not success, f"{name} should fail"
print(f"{name}: failed as expected")
print_errors(sim.sim_path)

# %% [markdown]
# ## Plot the results

# %%
fig, (ax_h, ax_z) = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
styles = {"swi-density-a": "b-", "swi-density-b": "co", "swi-density-c": "r-"}
for name, title in cases.items():
    x = sims[name].get_model().modelgrid.xcellcenters.flatten()
    head, zeta = results[name]
    kw = {"mfc": "none", "ms": 4} if name == "swi-density-b" else {}
    ax_h.plot(x, head, styles[name], label=f"{name} -- {title}", **kw)
    ax_z.plot(x, zeta, styles[name], label=f"{name} -- {title}", **kw)
ax_h.set_title("Freshwater head")
ax_h.set_ylabel("Head, in meters")
ax_h.legend(loc="lower center")
ax_z.set_title("Interface")
ax_z.set_ylabel("Elevation, in meters")
ax_z.set_xlabel("Distance, in meters")
ax_z.set_ylim(botm, top)
plt.show()
