# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Specific yield is required with storage
#
# The SWI Package uses specific yield (SY) as the drainable porosity for
# interface movement, which changes freshwater and saltwater storage even in
# confined cells. SY is therefore required whenever the STO Package is active
# with SWI. This confined, transient, one-layer model sets SY to zero, which is
# what STO uses when SY is omitted, and is expected to stop with an input
# error instead of silently computing zero interface storage.
#
# Ported from `autotest/test_gwf_swi_require_sy.py` on the `feat-swi` branch of
# [christianlangevin/modflow6](https://github.com/christianlangevin/modflow6/tree/feat-swi).

# %%
from pathlib import Path

import flopy

# %% [markdown]
# ## Model parameters
#
# Model files are written to `models/<case>/` beside the notebooks, and output
# to its `output/` subdirectory.

# %%
root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pixi.toml").exists())
model_root = root / "notebooks-verification" / "models"

name = "swi-nosy"
Lx = 10000.0  # island width, in meters
delr, delc = 100.0, 1.0
ncol = int(Lx / delr)
nlay, nrow = 1, 1
top, botm = 0.0, -400.0
k = 10.0  # meters per day
recharge = 0.001  # meters per day
h0 = 0.0  # sea level, in meters

# %% [markdown]
# ## Build and run the model
#
# flopy sets a nonzero SY when the argument is omitted, so SY is
# set to zero explicitly.

# %%
ws = model_root / name
(ws / "output").mkdir(parents=True, exist_ok=True)
sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
# transient, so that storage and SY are required
flopy.mf6.ModflowTdis(sim, perioddata=[(100000.0, 10, 1.0)])
flopy.mf6.ModflowIms(sim, linear_acceleration="bicgstab")
gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
flopy.mf6.ModflowGwfdis(
    gwf, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc, top=top, botm=botm
)
flopy.mf6.ModflowGwfic(gwf)
flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=k)
flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=1.0e-5, sy=0.0)
flopy.mf6.ModflowGwfswi(gwf, zeta_filerecord=f"output/{name}.zta")
cghb = 1.0 * delr * delc / 10.0
flopy.mf6.ModflowGwfghb(
    gwf, stress_period_data=[[0, 0, 0, h0, cghb], [0, 0, ncol - 1, h0, cghb]]
)
flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)
sim.write_simulation(silent=True)
success, _ = sim.run_simulation(silent=True)
assert not success, f"{name} should fail"

# %% [markdown]
# ## Check the error message

# %%
lines = (ws / "mfsim.lst").read_text().splitlines()
error_count = sum("specific yield (SY) be specified" in line for line in lines)
assert error_count == 1, f"expected one SY error in mfsim.lst, found {error_count}"
i = lines.index("ERROR REPORT:")
for line in lines[i + 2 :]:
    if not line.strip():
        break
    print(line)
