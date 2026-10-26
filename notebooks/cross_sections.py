# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cross sections of the island models
#
# Freshwater and saltwater along the model rows of figure 26 of Izuka and
# Rotzoll (2023), from the MODFLOW-2005 SWI2 output of the data release. Where a
# converted MODFLOW 6 model has been run in `notebooks/models/<island>/` (see
# `compare_mf2005_mf6`), its interface is drawn as a line.
#
# Run `pixi run get-data` first.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt

root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pixi.toml").exists())
sys.path.insert(0, str(root / "scripts"))
from cross_sections import (
    SECTIONS,
    mf6_results,
    mf2005_results,
    plot_cross_section,
)

# %% [markdown]
# ## Scenario
#
# Scenario 0 is the Current scenario (2010 conditions) shown in figure 26.

# %%
scenario = 0
islands = ["kauai", "oahu", "maui"]

# %% [markdown]
# ## Plot the cross sections

# %%
fig, axes = plt.subplots(
    len(islands), 1, figsize=(10, 3.4 * len(islands)), constrained_layout=True
)
for ax, island in zip(axes, islands):
    results = mf2005_results(island, scenario)
    ws = root / "notebooks" / "models" / island
    if scenario != 0:
        ws = ws.with_name(f"{island}.{scenario}")
    compare = (
        mf6_results(ws, island) if (ws / "output" / f"{island}.zta").exists() else None
    )
    plot_cross_section(ax, island, results, compare)
# one legend for the entries of every panel
entries = {}
for ax in axes:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        entries.setdefault(label, handle)
axes[0].legend(entries.values(), entries.keys(), loc="lower left", fontsize=8, ncol=6)
axes[-1].set_xlabel("Distance along the row, in miles")
plt.show()

# %% [markdown]
# ## Rows
#
# The model rows, which are one-based as in figure 26.

# %%
for island, (row, left, right) in SECTIONS.items():
    print(f"{island:6s} row {row:3d}  {left}-{right}")

# %% [markdown]
# ## Reference
#
# Izuka, S.K., and Rotzoll, K., 2023, Volcanic aquifers of Hawaii--Contributions
# to assessing groundwater availability on Kauai, Oahu, and Maui: U.S. Geological
# Survey Professional Paper 1876, https://doi.org/10.3133/pp1876.
