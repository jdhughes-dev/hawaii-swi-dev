# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Saltwater head from the Time-Varying Grid utility
#
# The saltwater head used by the SWI Package can be specified as the scalar
# SALTWATER_HEAD option or as a grid array in a TVG6 file. A three-layer,
# steady-state cross section with recharge and a constant head at one end is
# run with a uniform TVG array, which must reproduce the scalar option
# exactly, and with a different saltwater head in each layer, which must give
# a different interface in each layer.
#
# Ported from `autotest/test_gwf_swi_tvg.py` on the `feat-swi` branch of
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
    "swi-tvg-uniform": "Uniform saltwater head",
    "swi-tvg-layered": "Layered saltwater head",
}
refname = "swi-tvg-ref"

nlay, nrow, ncol = 3, 1, 6
delr = delc = 100.0
top = 0.0
botm = [-10.0, -20.0, -30.0]
k = 1.0  # meters per day
recharge = 1.0e-4  # meters per day
hsalt = 0.05  # saltwater head, in meters
hsalt_layered = [0.05, 0.10, 0.15]  # saltwater head in each layer, in meters

# %% [markdown]
# ## Build and run the models
#
# The TVG6 file is written beside the model files. A reference model,
# `models/swi-tvg-ref/`, specifies the uniform saltwater head with the
# SALTWATER_HEAD option.


# %%
def build_model(name):
    ws = model_root / name
    (ws / "output").mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim, nper=1)
    flopy.mf6.ModflowIms(sim, linear_acceleration="bicgstab", outer_maximum=200)
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
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=k)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 0), 0.0]])
    flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)
    if name == refname:
        flopy.mf6.ModflowGwfswi(
            gwf, zeta_filerecord=f"output/{name}.zta", saltwater_head=hsalt
        )
    else:
        swi = flopy.mf6.ModflowGwfswi(
            gwf,
            zeta_filerecord=f"output/{name}.zta",
            tvg6_filerecord=f"{name}.swi.tvg",
        )
        if name == "swi-tvg-uniform":
            arr = np.full((nlay, nrow, ncol), hsalt)
        else:
            arr = np.empty((nlay, nrow, ncol))
            for kk in range(nlay):
                arr[kk] = hsalt_layered[kk]
        swi.tvg.initialize(auxiliary=["saltwater_head"], aux={0: [arr]})
    flopy.mf6.ModflowGwfoc(
        gwf, head_filerecord=f"output/{name}.hds", saverecord=[("HEAD", "ALL")]
    )
    return sim


sims = {}
for name in [refname, *cases]:
    sim = build_model(name)
    sim.write_simulation(silent=True)
    success, _ = sim.run_simulation(silent=True)
    assert success, f"{name} did not run"
    sims[name] = sim


def get_zeta(name):
    ws = Path(sims[name].sim_path)
    zobj = flopy.utils.HeadFile(ws / f"output/{name}.zta", text="zeta")
    return zobj.get_data().reshape(nlay, ncol)


def get_head(name):
    return sims[name].get_model().output.head().get_data().reshape(nlay, ncol)


# %% [markdown]
# ## Check the results
#
# The uniform TVG array must give the same head and interface as the scalar
# option, which shows that TVG fills every node and not only the first layer.
# The layered saltwater head must give a distinct mean interface elevation in
# each layer.

# %%
head, href = get_head("swi-tvg-uniform"), get_head(refname)
zeta, zref = get_zeta("swi-tvg-uniform"), get_zeta(refname)
assert np.allclose(head, href, atol=1.0e-12)
assert np.allclose(zeta, zref, atol=1.0e-12)
print(f"swi-tvg-uniform: maximum zeta difference {np.abs(zeta - zref).max():.1e} m")

means = get_zeta("swi-tvg-layered").mean(axis=1)
assert len(set(np.round(means, 6))) == nlay
print(f"swi-tvg-layered: layer mean zeta {np.round(means, 4).tolist()} m")

# %% [markdown]
# ## Plot the results

# %%
x = sims[refname].get_model().modelgrid.xcellcenters.flatten()
colors = ["C0", "C1", "C2"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
ax = axes[0]
for kk in range(nlay):
    ax.plot(x, zref[kk], "-", color=colors[kk], label=f"Layer {kk + 1}, scalar")
    ax.plot(
        x, zeta[kk], "o", mfc="none", color=colors[kk], label=f"Layer {kk + 1}, TVG"
    )
ax.set_title(f"swi-tvg-uniform -- saltwater head {hsalt} m")
ax.set_ylabel("Interface elevation, in meters")
ax = axes[1]
zlay = get_zeta("swi-tvg-layered")
for kk in range(nlay):
    ax.plot(
        x,
        zlay[kk],
        "o-",
        mfc="none",
        color=colors[kk],
        label=f"Layer {kk + 1}, saltwater head {hsalt_layered[kk]} m",
    )
ax.set_title("swi-tvg-layered -- layered saltwater head")
for ax in axes:
    ax.set_xlabel("Distance, in meters")
    ax.legend(loc="lower left", fontsize=8)
plt.show()
