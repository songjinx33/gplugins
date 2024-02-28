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

    recipe = FdtdRecipe(
        component=taper,
        material_map=layer_map,
        convergence_setup=LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
        simulation_setup=SIMULATION_SETTINGS_LUMERICAL_FDTD,
    )

    recipe.eval()


class FdtdRecipe(DesignRecipe):
    """
    FDTD recipe that extracts sparams
    """

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
        """
        Run FDTD recipe to extract sparams

        1. Performs port convergence by resizing ports to ensure E-field intensities decay to specified threshold
        2. Performs mesh convergence to ensure sparams converge to certain sparam_diff
        3. Extracts sparams after updating simulation with optimal simulation params
        """
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
