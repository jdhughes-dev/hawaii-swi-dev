"""Cross sections of the data release models along the rows of PP 1876 figure 26.

Izuka, S.K., and Rotzoll, K., 2023, Volcanic aquifers of Hawaii--Contributions
to assessing groundwater availability on Kauai, Oahu, and Maui: U.S. Geological
Survey Professional Paper 1876, https://doi.org/10.3133/pp1876.
"""

from pathlib import Path

import flopy
import numpy as np
from mf2005_to_mf6 import load_mf2005, mf2005_heads_zeta

# model row (one-based) and end labels of each cross section in figure 26
SECTIONS = {
    "kauai": (210, "A", "A'"),
    "oahu": (167, "B", "B'"),
    "maui": (119, "C", "C'"),
}

FRESH = "#6fcfdf"
SALT = "#d4ecf2"
FEET_PER_MILE = 5280.0


def mf2005_results(island: str, scenario: int = 0) -> dict:
    """Grid, heads, and interface of a data release model.

    Heads and interface are averages of the last two saved time steps.
    """
    m = load_mf2005(island, scenario)
    head, zeta = mf2005_heads_zeta(island, scenario)
    return {
        "delr": m.dis.delr.array,
        "top": m.dis.top.array,
        "botm": m.dis.botm.array,
        "active": m.bas6.ibound.array != 0,
        "head": head,
        "zeta": zeta,
    }


def mf6_results(ws: Path, name: str) -> dict:
    """Heads and interface of a converted MODFLOW 6 model."""
    ws = Path(ws)
    return {
        "head": flopy.utils.HeadFile(ws / "output" / f"{name}.hds").get_data(),
        "zeta": flopy.utils.HeadFile(
            ws / "output" / f"{name}.zta", text="zeta"
        ).get_data(),
    }


def _steps(x_edges: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Step outline of cell values between cell edges."""
    return np.repeat(x_edges, 2)[1:-1], np.repeat(v, 2)


def plot_cross_section(ax, island: str, results: dict, compare: dict | None = None):
    """Plot freshwater and saltwater along the figure 26 row of an island.

    ``compare`` holds the heads and interface of a second model, which are
    drawn as lines.
    """
    row, left, right = SECTIONS[island]
    i = row - 1
    active = results["active"][:, i, :]
    cols = np.flatnonzero(active.any(axis=0))
    j0, j1 = cols[0], cols[-1] + 1
    x_edges = np.concatenate(([0.0], np.cumsum(results["delr"][j0:j1]))) / FEET_PER_MILE

    top = results["top"][i, j0:j1]
    botm = results["botm"][:, i, j0:j1]
    nlay = botm.shape[0]
    layer_top = np.vstack((top, botm[:-1]))
    for k in range(nlay):
        act = active[k, j0:j1]
        ltop, lbot = layer_top[k], botm[k]
        head = results["head"][k, i, j0:j1]
        zeta = np.clip(results["zeta"][k, i, j0:j1], lbot, ltop)
        wet = act & (head > lbot)
        fresh_top = np.where(wet, np.minimum(head, ltop), lbot)
        fresh_bot = np.where(wet, np.minimum(zeta, fresh_top), lbot)
        salt_top = np.where(act, np.minimum(fresh_bot, ltop), np.nan)
        x, y = _steps(x_edges, fresh_top)
        _, yb = _steps(x_edges, fresh_bot)
        ax.fill_between(
            x, yb, y, color=FRESH, lw=0, label="Fresh water" if k == 0 else None
        )
        _, ys = _steps(x_edges, salt_top)
        _, yl = _steps(x_edges, np.where(act, lbot, np.nan))
        ax.fill_between(
            x, yl, ys, color=SALT, lw=0, label="Salt water" if k == 0 else None
        )
        if k > 0:
            ax.plot(
                *_steps(x_edges, np.where(act, ltop, np.nan)),
                color="0.3",
                lw=0.6,
                label="Layer boundary" if k == 1 else None,
            )

    ax.plot(*_steps(x_edges, top), color="k", lw=0.8, label="Model top")
    ax.axhline(0.0, color="k", lw=0.8, ls=(0, (6, 4)), label="Sea level")

    if compare is not None:
        for k in range(nlay):
            # only where the interface is within the layer, by more than a foot
            zeta = compare["zeta"][k, i, j0:j1]
            within = (
                active[k, j0:j1] & (zeta > botm[k] + 1.0) & (zeta < layer_top[k] - 1.0)
            )
            zeta = np.where(within, zeta, np.nan)
            ax.plot(
                *_steps(x_edges, zeta),
                color="tab:red",
                lw=0.9,
                label="MODFLOW 6 interface" if k == 0 else None,
            )

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylabel("Altitude, in feet relative to sea level")
    ax.text(0.0, 1.01, left, transform=ax.transAxes, fontweight="bold", style="italic")
    ax.text(
        1.0,
        1.01,
        right,
        transform=ax.transAxes,
        ha="right",
        fontweight="bold",
        style="italic",
    )
    ax.set_title(f"{island.title()}, row {row}", loc="left", fontsize=10, pad=14)
    return ax
