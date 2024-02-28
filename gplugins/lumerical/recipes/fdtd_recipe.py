import hashlib
from pathlib import Path

import gdsfactory as gf
import pandas as pd
from gdsfactory import Component
from gdsfactory.pdk import LayerStack, get_layer_stack
from gdsfactory.typings import PathType

from gplugins.design_recipe.DesignRecipe import DesignRecipe, eval_decorator
from gplugins.lumerical.convergence_settings import (
    LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
    ConvergenceSettingsLumericalFdtd,
)
from gplugins.lumerical.fdtd import LumericalFdtdSimulation
from gplugins.lumerical.simulation_settings import (
    SIMULATION_SETTINGS_LUMERICAL_FDTD,
    SimulationSettingsLumericalFdtd,
)


def main():
    from functools import partial

    from gdsfactory.components.taper_cross_section import taper_cross_section

    xs_wg = partial(
        gf.cross_section.cross_section,
        layer=(1, 0),
        width=0.5,
    )

    xs_wg_wide = partial(
        gf.cross_section.cross_section,
        layer=(1, 0),
        width=2.0,
    )

    taper = taper_cross_section(
        cross_section1=xs_wg,
        cross_section2=xs_wg_wide,
        length=5,
        width_type="parabolic",
    )

    layer_map = {
        "si": "Si (Silicon) - Palik",
        "sio2": "SiO2 (Glass) - Palik",
        "sin": "Si3N4 (Silicon Nitride) - Phillip",
        "TiN": "TiN - Palik",
        "Aluminum": "Al (Aluminium) Palik",
    }
    from gdsfactory.technology.layer_stack import LayerLevel, LayerStack

    layerstack_lumerical2021 = LayerStack(
        layers={
            "clad": LayerLevel(
                name=None,
                layer=(99999, 0),
                thickness=3.0,
                thickness_tolerance=None,
                zmin=0.0,
                zmin_tolerance=None,
                material="sio2",
                sidewall_angle=0.0,
                sidewall_angle_tolerance=None,
                width_to_z=0.0,
                z_to_bias=None,
                mesh_order=9,
                layer_type="background",
                mode=None,
                into=None,
                resistivity=None,
                bias=None,
                derived_layer=None,
                info={},
                background_doping_concentration=None,
                background_doping_ion=None,
                orientation="100",
            ),
            "box": LayerLevel(
                name=None,
                layer=(99999, 0),
                thickness=3.0,
                thickness_tolerance=None,
                zmin=-3.0,
                zmin_tolerance=None,
                material="sio2",
                sidewall_angle=0.0,
                sidewall_angle_tolerance=None,
                width_to_z=0.0,
                z_to_bias=None,
                mesh_order=9,
                layer_type="background",
                mode=None,
                into=None,
                resistivity=None,
                bias=None,
                derived_layer=None,
                info={},
                background_doping_concentration=None,
                background_doping_ion=None,
                orientation="100",
            ),
            "core": LayerLevel(
                name=None,
                layer=(1, 0),
                thickness=0.22,
                thickness_tolerance=None,
                zmin=0.0,
                zmin_tolerance=None,
                material="si",
                sidewall_angle=10.0,
                sidewall_angle_tolerance=None,
                width_to_z=0.5,
                z_to_bias=None,
                mesh_order=2,
                layer_type="grow",
                mode=None,
                into=None,
                resistivity=None,
                bias=None,
                derived_layer=None,
                info={"active": True},
                background_doping_concentration=100000000000000.0,
                background_doping_ion="Boron",
                orientation="100",
            ),
            # KNOWN ISSUE: Lumerical 2021 version of Layer Builder does not support dopants in process file
        }
    )

    recipe = FdtdRecipe(
        component=taper,
        material_map=layer_map,
        layer_stack=layerstack_lumerical2021,
        convergence_setup=LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
        simulation_setup=SIMULATION_SETTINGS_LUMERICAL_FDTD,
        dirpath="/root/PycharmProjects/gdsfactory_sean/gplugins/gplugins/lumerical/tests/test1",
    )

    recipe.eval()


class FdtdRecipe(DesignRecipe):
    # Setup
    simulation_setup: SimulationSettingsLumericalFdtd | None = (
        SIMULATION_SETTINGS_LUMERICAL_FDTD
    )
    convergence_setup: ConvergenceSettingsLumericalFdtd | None = (
        LUMERICAL_FDTD_CONVERGENCE_SETTINGS
    )

    # Results
    sparameters: pd.DataFrame | None = None

    def __init__(
        self,
        component: Component | None = None,
        material_map: dict[str, str] | None = None,
        layer_stack: LayerStack | None = None,
        simulation_setup: SimulationSettingsLumericalFdtd
        | None = SIMULATION_SETTINGS_LUMERICAL_FDTD,
        convergence_setup: ConvergenceSettingsLumericalFdtd
        | None = LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
        dirpath: PathType | None = None,
    ):
        layer_stack = layer_stack or get_layer_stack()
        super().__init__(
            cell=component, material_map=material_map, layer_stack=layer_stack
        )
        self.dirpath = dirpath or Path(__file__).resolve().parent
        self.simulation_setup = simulation_setup
        self.convergence_setup = convergence_setup

    def __hash__(self) -> int:
        """
        Returns a hash of all state and setup this DesignRecipe contains.
        This is used to determine 'freshness' of a recipe (i.e. if it needs to be rerun)

        Hashed items:
        - design intent
        - simulation setup
        - convergence setup
        """
        h = hashlib.sha1()
        int_hash = super().__hash__()
        h.update(int_hash.to_bytes(int_hash.bit_length() + 7 // 8, byteorder="big"))
        h.update(self.simulation_setup.model_dump_json().encode("utf-8"))
        h.update(self.convergence_setup.model_dump_json().encode("utf-8"))
        return int.from_bytes(h.digest(), "big")

    @eval_decorator
    def eval(self):
        sim = LumericalFdtdSimulation(
            component=self.cell,
            material_map=self.material_map,
            layerstack=self.layer_stack,
            simulation_settings=self.simulation_setup,
            convergence_settings=self.convergence_setup,
            dirpath=self.dirpath,
            hide=False,  # TODO: Make global var to decide when to show sims
            run_mesh_convergence=True,
            run_port_convergence=True,
            run_field_intensity_convergence=False,
        )

        self.sparameters = sim.write_sparameters(
            overwrite=True, delete_fsp_files=True, plot=True
        )

        # Rehash since convergence testing updates simulation parameters
        self.last_hash = hash(self)


if __name__ == "__main__":
    main()
