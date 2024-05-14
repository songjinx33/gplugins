from gplugins.design_recipe.DesignRecipe import DesignRecipe
from pydantic import BaseModel
from gdsfactory.components.straight import straight
from gdsfactory.typings import ComponentFactory, PathType
from gdsfactory.technology.layer_stack import LayerStack
from gdsfactory.pdk import get_layer_stack
from gplugins.lumerical.simulation_settings import (
    SimulationSettingsLumericalMode,
    LUMERICAL_MODE_SIMULATION_SETTINGS,
)
from gplugins.lumerical.convergence_settings import (
    ConvergenceSettingsLumericalMode,
    LUMERICAL_MODE_CONVERGENCE_SETTINGS,
)
from gplugins.design_recipe.DesignRecipe import eval_decorator
from gplugins.lumerical.mode import LumericalModeSimulation
import pandas as pd

class WaveguideSweepDesignIntent(BaseModel):
    r"""
    Design intent for sweeping waveguide geometry and characterizing optical characteristics

    Attributes:
        param_name: Waveguide cell parameter name
        param_vals: Parameter values to sweep the waveguide geometry
    """
    param_name: str = ""
    param_vals: list = []

    class Config:
        arbitrary_types_allowed = True

class WaveguideSweepRecipe(DesignRecipe):
    def __init__(
        self,
        cell: ComponentFactory = straight,
        design_intent: WaveguideSweepDesignIntent | None = None,
        layer_stack: LayerStack | None = None,
        simulation_setup: SimulationSettingsLumericalMode
        | None = LUMERICAL_MODE_SIMULATION_SETTINGS,
        convergence_setup: ConvergenceSettingsLumericalMode
        | None = LUMERICAL_MODE_CONVERGENCE_SETTINGS,
        dirpath: PathType | None = None,
    ):
        r"""
        Set up waveguide geometry sweep to characterize optical characteristics

        Parameters:
            cell: Waveguide cell
            design_intent: Waveguide sweep design intent
            layer_stack: PDK layerstack
            simulation_setup: MODE simulation setup
            convergence_setup: MODE convergence setup
            dirpath: Directory to save files
        """
        layer_stack = layer_stack or get_layer_stack()
        super().__init__(cell=cell, layer_stack=layer_stack, dirpath=dirpath)
        # Add information to recipe setup. NOTE: This is used for hashing
        self.recipe_setup.simulation_setup = simulation_setup
        self.recipe_setup.convergence_setup = convergence_setup
        self.recipe_setup.design_intent = design_intent or WaveguideSweepDesignIntent()

    @eval_decorator
    def eval(self, run_convergence: bool = True):
        r"""
        Run taper design recipe.

        1. Sweep waveguide geometry in MODE and extract relevant waveguide parameters

        Parameters:
            run_convergence: Run convergence if True
        """
        di = self.recipe_setup.design_intent
        ss = self.recipe_setup.simulation_setup

        # Set up sweep
        params = [{di.param_name: val} for val in di.param_vals]

        neff = []
        ng = []
        te_polarization = []
        for param in params:
            # Update simulation setup based on component
            c = self.cell(**param)
            simulation_setup = self.recipe_setup.simulation_setup.copy()
            simulation_setup.x = c.x
            simulation_setup.y = c.y

            sim = LumericalModeSimulation(
                component=c,
                layerstack=self.recipe_setup.layer_stack,
                simulation_settings=simulation_setup,
                convergence_settings=self.recipe_setup.convergence_setup,
                dirpath=self.dirpath,
                run_port_convergence=run_convergence,
                run_mesh_convergence=run_convergence,
                hide=False, # TODO: Add global to show or hide sims
            )

            s = sim.session

            # Get modes
            s.run()
            s.mesh()
            s.findmodes()
            s.selectmode(ss.target_mode)

            # Get optical characteristics
            neff.append(s.getresult(f"FDE::data::mode{ss.target_mode}", "neff")[0][0])
            ng.append(s.getresult(f"FDE::data::mode{ss.target_mode}", "ng")[0][0])
            te_polarization.append(s.getresult(f"FDE::data::mode{ss.target_mode}", "TE polarization fraction"))

        self.recipe_results.waveguide_sweep_characteristics = pd.DataFrame({di.param_name: di.param_vals,
                                                                            "neff": neff,
                                                                            "ng": ng,
                                                                            "TE_polarization": te_polarization})
        return True