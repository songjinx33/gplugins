import hashlib
from pathlib import Path

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


class FdtdRecipe(DesignRecipe):
    """
    FDTD recipe that extracts sparams

    Attributes:
        simulation_setup: FDTD simulation setup
        convergence_setup: FDTD convergence setup
        dirpath: Directory to store files.
        sparameters: s-parameter results.
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
        layer_stack: LayerStack | None = None,
        simulation_setup: SimulationSettingsLumericalFdtd
        | None = SIMULATION_SETTINGS_LUMERICAL_FDTD,
        convergence_setup: ConvergenceSettingsLumericalFdtd
        | None = LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
        dirpath: PathType | None = None,
    ):
        """
        Set up FDTD recipe

        Parameters:
            component: Component
            layer_stack: PDK layer stack
            simulation_setup: FDTD simulation setup
            convergence_setup: FDTD convergence setup
            dirpath: Directory to store files.
        """
        layer_stack = layer_stack or get_layer_stack()
        super().__init__(cell=component, layer_stack=layer_stack)
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
        h.update(int_hash.to_bytes(int_hash.bit_length(), byteorder="big"))
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
            layerstack=self.layer_stack,
            simulation_settings=self.simulation_setup,
            convergence_settings=self.convergence_setup,
            dirpath=self.dirpath,
            hide=False,  # TODO: Make global var to decide when to show sims
            run_mesh_convergence=True,
            run_port_convergence=True,
            run_field_intensity_convergence=True,
        )

        self.sparameters = sim.write_sparameters(
            overwrite=True, delete_fsp_files=True, plot=True
        )
