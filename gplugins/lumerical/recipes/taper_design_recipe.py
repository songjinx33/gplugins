import typing
from functools import partial
from pathlib import Path

import gdsfactory as gf
import numpy as np
import pandas as pd
from gdsfactory import Component
from gdsfactory.components.taper_cross_section import taper_cross_section
from gdsfactory.config import logger
from gdsfactory.pdk import LayerStack, get_layer_stack
from gdsfactory.typings import ComponentFactory, CrossSectionSpec, PathType, WidthTypes
from pydantic import BaseModel

from gplugins.design_recipe.DesignRecipe import DesignRecipe, eval_decorator
from gplugins.lumerical.convergence_settings import (
    LUMERICAL_EME_CONVERGENCE_SETTINGS,
    ConvergenceSettingsLumericalEme,
)
from gplugins.lumerical.eme import LumericalEmeSimulation
from gplugins.lumerical.simulation_settings import (
    LUMERICAL_EME_SIMULATION_SETTINGS,
    SimulationSettingsLumericalEme,
)

um = 1e-6
cm = 1e-2


class RoutingTaperDesignIntent(BaseModel):
    r"""
    Design intent for routing taper design recipe

    Attributes:
        narrow_waveguide_routing_loss_per_cm: Narrow waveguide routing loss (dB/cm)
        narrow_waveguide_cross_section: Narrow waveguide cross section
        wide_waveguide_cross_section: Wide waveguide cross section
        start_length: Starting length in length sweep (um)
        stop_length: Ending length in length sweep (um)
        num_pts: Number of points to consider in length sweep

                         |       |
                         |      /|---------
                         |    /  |
                         |  /    |
        -----------------|/      |
        narrow waveguide | taper | wide waveguide
        -----------------|\      |
                         |  \    |
                         |    \  |
                         |      \|---------
                         |       |

    """

    narrow_waveguide_routing_loss_per_cm: float = 3  # dB / cm
    narrow_waveguide_cross_section: CrossSectionSpec = partial(
        gf.cross_section.cross_section,
        layer=(1, 0),
        width=0.5,
    )
    wide_waveguide_cross_section: CrossSectionSpec = partial(
        gf.cross_section.cross_section,
        layer=(1, 0),
        width=3.0,
    )

    # Length Sweep
    start_length: float = 1  # um
    stop_length: float = 200  # um
    num_pts: int = 200  # um

    class Config:
        arbitrary_types_allowed = True


class RoutingTaperDesignRecipe(DesignRecipe):
    """
    Routing taper design recipe.
    """

    # Design intent
    design_intent: RoutingTaperDesignIntent | None = None

    # Setup
    simulation_setup: SimulationSettingsLumericalEme | None = (
        LUMERICAL_EME_SIMULATION_SETTINGS
    )
    convergence_setup: ConvergenceSettingsLumericalEme | None = (
        LUMERICAL_EME_CONVERGENCE_SETTINGS
    )

    # Results
    component: Component | None = None  # Optimal taper component
    length_sweep: pd.DataFrame | None = None  # Length sweep results

    def __init__(
        self,
        cell: ComponentFactory = taper_cross_section,
        design_intent: RoutingTaperDesignIntent | None = None,
        material_map: dict[str, str] | None = None,
        layer_stack: LayerStack | None = None,
        simulation_setup: SimulationSettingsLumericalEme
        | None = LUMERICAL_EME_SIMULATION_SETTINGS,
        convergence_setup: ConvergenceSettingsLumericalEme
        | None = LUMERICAL_EME_CONVERGENCE_SETTINGS,
        dirpath: PathType | None = None,
    ):
        layer_stack = layer_stack or get_layer_stack()
        super().__init__(cell=cell, material_map=material_map, layer_stack=layer_stack)
        self.dirpath = dirpath or Path(__file__).resolve().parent
        self.design_intent = design_intent or RoutingTaperDesignIntent()
        self.simulation_setup = simulation_setup
        self.convergence_setup = convergence_setup

    @eval_decorator
    def eval(
        self,
        simulation_setup: SimulationSettingsLumericalEme | None = None,
        convergence_setup: ConvergenceSettingsLumericalEme | None = None,
    ):
        r"""
        Run taper design recipe.

                         |       |
                         |      /|---------
                         |    /  |
                         |  /    |
        -----------------|/      |
        narrow waveguide | taper | wide waveguide
        -----------------|\      |
                         |  \    |
                         |    \  |
                         |      \|---------
                         |       |

        1. Sweep taper geometry in EME and get best geometry and length for component.
                Best component is derived from the following (in order):
                a) The dB/cm loss for the narrow waveguide routing must match the derived dB/cm loss for the taper
                b) The component must have the lowest reflections
                c) The component must be the shortest
        2. Run FDTD simulation to extract s-params for best component

        Parameters
            simulation_setup: Simulation settings
            convergence_setup: Convergence settings

        """
        ss = simulation_setup or self.simulation_setup
        cs = convergence_setup or self.convergence_setup
        di = self.design_intent

        self.simulation_setup = ss
        self.convergence_setup = cs

        # Sweep geometry
        components = [
            self.cell(
                cross_section1=di.narrow_waveguide_cross_section,
                cross_section2=di.wide_waveguide_cross_section,
                length=10,  # um
                width_type=wtype,
            )
            for wtype in [typing.get_args(WidthTypes)]
        ]

        optimal_lengths = []
        transmission_coefficients = []
        reflection_coefficients = []
        sims = []
        length_sweeps = []
        for component in components:
            try:
                sim = LumericalEmeSimulation(
                    component=component,
                    material_map=self.material_map,
                    layerstack=self.layer_stack,
                    simulation_settings=ss,
                    convergence_settings=cs,
                    hide=False,  # TODO: Make global variable for switching debug modes
                    run_overall_convergence=True,
                    run_mesh_convergence=False,
                    run_mode_convergence=True,
                    run_cell_convergence=False,
                    dirpath=self.dirpath,
                )
                sims.append(sim)
            except Exception as err:
                logger.warning(
                    f"{err}\n{component.name} failed to simulate. Moving onto next component"
                )
                continue

            # Get length of taper that has lower loss than routing loss
            length_sweep = sim.get_length_sweep(
                start_length=di.start_length,
                stop_length=di.stop_length,
                num_pts=di.num_pts,
            )
            length_sweeps.append(length_sweep)

            length = length_sweep.loc[:, "length"]
            s21 = 10 * np.log10(abs(length_sweep.loc[:, "s21"]) ** 2)
            s11 = 10 * np.log10(abs(length_sweep.loc[:, "s11"]) ** 2)
            try:
                ind = next(
                    k
                    for k, value in enumerate(list(s21))
                    if value > -di.narrow_waveguide_routing_loss_per_cm * length[k] / cm
                )
                optimal_lengths.append(length[ind] / um)
                transmission_coefficients.append(s21[ind])
                reflection_coefficients.append(s11[ind])
            except StopIteration:
                logger.warning(
                    f"{component.name} cannot achieve specified routing loss of "
                    + f"-{di.narrow_waveguide_routing_loss_per_cm}dB/cm. Use maximal length of {di.stop_length}um."
                )
                optimal_lengths.append(di.stop_length)
                transmission_coefficients.append(s21[-1])
                reflection_coefficients.append(s11[-1])

        # Get best component
        # Most optimal component is one with smallest length AND least reflections
        # If not both, choose component with least reflections
        ind1 = optimal_lengths.index(min(optimal_lengths))
        ind2 = reflection_coefficients.index(min(reflection_coefficients))

        if ind1 == ind2:
            # Select component with minimal length and minimal reflections
            optimal_component = self.cell(
                cross_section1=di.narrow_waveguide_cross_section,
                cross_section2=di.wide_waveguide_cross_section,
                length=optimal_lengths[ind1],  # um
                width_type=list(typing.get_args(WidthTypes))[ind1],
            )
            opt_length_sweep_data = length_sweeps[ind1]
        else:
            # Select component with minimal reflections
            optimal_component = self.cell(
                cross_section1=di.narrow_waveguide_cross_section,
                cross_section2=di.wide_waveguide_cross_section,
                length=optimal_lengths[ind2],  # um
                width_type=list(typing.get_args(WidthTypes))[ind2],
            )
            opt_length_sweep_data = length_sweeps[ind2]

        # Save results
        self.component = optimal_component
        self.length_sweep = opt_length_sweep_data


if __name__ == "__main__":
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
                sidewall_angle=2.0,
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

    LUMERICAL_EME_CONVERGENCE_SETTINGS.sparam_diff = 1 - 10 ** (-0.0025 / 10)
    taper_recipe = RoutingTaperDesignRecipe(
        material_map=layer_map,
        simulation_setup=LUMERICAL_EME_SIMULATION_SETTINGS,
        convergence_setup=LUMERICAL_EME_CONVERGENCE_SETTINGS,
        layer_stack=layerstack_lumerical2021,
    )
    success = taper_recipe.eval()
    logger.info("Done")
