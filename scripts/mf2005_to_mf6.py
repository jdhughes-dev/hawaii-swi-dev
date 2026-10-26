#!/usr/bin/env python
"""Convert a data release MODFLOW-2005 SWI2 model to a MODFLOW 6 SWI model.

The Kauai, Maui, and Oahu models of Rotzoll and Izuka (2023) are one-layer,
steady-state flow models in which SWI2 moves the interface to equilibrium over
a long stress period. The MODFLOW 6 model is a steady-state, single-model SWI
simulation, in which the interface is in equilibrium with the freshwater head.
The input is read from data/pp1876/ (pixi run get-data).
"""

import argparse
import sys
from pathlib import Path

import flopy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "pp1876"

# model name prefix of each island in the data release
PREFIX = {"kauai": "Kauai_01", "maui": "Maui_08", "oahu": "Oahu_04"}


def mf2005_name(island: str, scenario: int) -> str:
    return f"{PREFIX[island]}_{scenario}"


def load_mf2005(island: str, scenario: int = 0) -> flopy.modflow.Modflow:
    """Load the input of a data release model."""
    ws = DATA_DIR / "model" / f"input.{island}.{scenario}"
    if not ws.is_dir():
        sys.exit(f"'{ws}' does not exist; run `pixi run get-data {island}`")
    return flopy.modflow.Modflow.load(
        f"{mf2005_name(island, scenario)}.nam",
        model_ws=ws,
        check=False,
        verbose=False,
        load_only=["dis", "bas6", "lpf", "ghb", "drn", "wel", "rch", "swi2"],
    )


def mf2005_output(island: str, scenario: int = 0) -> Path:
    """Directory holding the MODFLOW-2005 output of a data release model."""
    return DATA_DIR / "output" / f"output.{island}.{scenario}"


def stress_period_data(package, fields: list[str]) -> list:
    rec = package.stress_period_data[0]
    return [
        [(int(r["k"]), int(r["i"]), int(r["j"])), *[float(r[f]) for f in fields]]
        for r in rec
    ]


def build_gwf(
    sim: flopy.mf6.MFSimulation,
    m: flopy.modflow.Modflow,
    name: str,
    saltwater: bool = False,
) -> flopy.mf6.ModflowGwf:
    """Build the freshwater, or saltwater, GWF model of a data release model."""
    dis, bas, lpf, swi2 = m.dis, m.bas6, m.lpf, m.swi2

    # the SWI Package requires the Newton-Raphson formulation
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=name, save_flows=True, newtonoptions="newton"
    )
    ibound = bas.ibound.array
    if np.any(ibound < 0):
        raise NotImplementedError("constant-head cells are not converted")
    flopy.mf6.ModflowGwfdis(
        gwf,
        length_units="feet",
        nlay=dis.nlay,
        nrow=dis.nrow,
        ncol=dis.ncol,
        delr=dis.delr.array,
        delc=dis.delc.array,
        top=dis.top.array,
        botm=dis.botm.array,
        idomain=(ibound != 0).astype(int),
    )
    # saltwater heads start at sea level
    flopy.mf6.ModflowGwfic(gwf, strt=0.0 if saltwater else bas.strt.array)

    # HANI is the ratio of the hydraulic conductivity along columns to that
    # along rows, which is K22 / K in MODFLOW 6
    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=lpf.laytyp.array,
        k=lpf.hk.array,
        k22=lpf.hk.array * lpf.hani.array,
        k33=lpf.vka.array,
        save_specific_discharge=True,
    )

    # nu holds the density of each zone relative to freshwater
    nu = swi2.nu.array
    flopy.mf6.ModflowGwfswi(
        gwf,
        relative_density_difference=float(nu[-1] - nu[0]),
        zeta_filerecord=f"output/{name}.zta",
    )

    # the seafloor GHB heads are equivalent freshwater heads of the sea; in
    # the saltwater model they are sea level, which is zero
    ghb = stress_period_data(m.ghb, ["bhead", "cond"])
    if saltwater:
        ghb = [[cellid, 0.0, cond] for cellid, _, cond in ghb]
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=ghb)
    if not saltwater:
        flopy.mf6.ModflowGwfdrn(
            gwf, stress_period_data=stress_period_data(m.drn, ["elev", "cond"])
        )
        flopy.mf6.ModflowGwfwel(
            gwf, stress_period_data=stress_period_data(m.wel, ["flux"])
        )
        flopy.mf6.ModflowGwfrcha(gwf, recharge=m.rch.rech.array[0, 0])

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"output/{name}.hds",
        budget_filerecord=f"output/{name}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("BUDGET", "ALL")],
    )
    return gwf


def convert(
    island: str,
    scenario: int = 0,
    ws: Path | None = None,
    two_fluid: bool = False,
    exe_name: str = "mf6",
) -> flopy.mf6.MFSimulation:
    """Build the MODFLOW 6 simulation for a data release model.

    The freshwater model is named after the island. With ``two_fluid``, a
    saltwater model (``<island>_salt``) is coupled to it with an SWI-SWI
    Exchange. Model files are written to ``ws`` and output to its output/
    subdirectory.
    """
    m = load_mf2005(island, scenario)
    if ws is None:
        ws = ROOT / "notebooks" / "models" / island
    ws = Path(ws)
    (ws / "output").mkdir(parents=True, exist_ok=True)

    sim = flopy.mf6.MFSimulation(sim_name=island, sim_ws=ws, exe_name=exe_name)
    flopy.mf6.ModflowTdis(sim, time_units="days", perioddata=[(1.0, 1, 1.0)])
    # the outer closure matches the MODFLOW-2005 PCG HCLOSE; the Oahu model
    # takes about 930 outer iterations
    ims = flopy.mf6.ModflowIms(
        sim,
        complexity="complex",
        outer_maximum=2000,
        inner_maximum=300,
        outer_dvclose=0.1,
        inner_dvclose=1.0e-2,
        rcloserecord=[10.0, "strict"],
        linear_acceleration="bicgstab",
    )
    models = [build_gwf(sim, m, island)]
    if two_fluid:
        models.append(build_gwf(sim, m, f"{island}_salt", saltwater=True))
        flopy.mf6.ModflowSwiswi(
            sim,
            exgtype="SWI6-SWI6",
            exgmnamea=models[0].name,
            exgmnameb=models[1].name,
        )
        sim.register_ims_package(ims, [gwf.name for gwf in models])
    return sim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("island", choices=sorted(PREFIX))
    parser.add_argument(
        "--scenario", type=int, default=0, help="Scenario number; default 0."
    )
    parser.add_argument("--ws", type=Path, help="Model workspace.")
    parser.add_argument(
        "--two-fluid", action="store_true", help="Add a saltwater model."
    )
    parser.add_argument("--run", action="store_true", help="Run the model.")
    args = parser.parse_args()

    sim = convert(args.island, args.scenario, args.ws, args.two_fluid)
    sim.write_simulation(silent=True)
    print(f"wrote '{sim.sim_path}'")
    if args.run:
        success, _ = sim.run_simulation(silent=True)
        if not success:
            sys.exit("MODFLOW 6 did not converge")
        print("MODFLOW 6 ran")
    return 0


if __name__ == "__main__":
    sys.exit(main())
