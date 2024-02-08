from __future__ import annotations

from typing import Literal
from pydantic import BaseModel
from gdsfactory.component import Component
from gdsfactory.technology.layer_stack import LayerStack
from gdsfactory.pdk import get_layer_stack
from gdsfactory.typings import PathType
from gdsfactory.config import logger
from pathlib import Path

import math
import lumapi
import gdsfactory as gf

um = 1e-6


class SimulationSettingsLumericalEme(BaseModel):
    """Lumerical EME simulation_settings.

    Parameters:

    """

    wavelength: float = 1.55
    wavelength_start: float = 1.5
    wavelength_stop: float = 1.6
    material_fit_tolerance: float = 0.001

    group_cells: list[int] = [1, 50, 1]
    group_spans: list[float] = [1, 10, 1]
    group_subcell_methods: list[str] = [None, "CVCS", None]
    num_modes: int = 50
    energy_conservation: Literal[
        "make passive", "conserve energy"
    ] | None = "make passive"

    mesh_cells_per_wavelength: int = 50

    ymin_boundary: Literal[
        "Metal", "PML", "Anti-Symmetric", "Symmetric"
    ] = "Anti-Symmetric"
    ymax_boundary: Literal["Metal", "PML", "Anti-Symmetric", "Symmetric"] = "Metal"
    zmin_boundary: Literal["Metal", "PML", "Anti-Symmetric", "Symmetric"] = "Metal"
    zmax_boundary: Literal["Metal", "PML", "Anti-Symmetric", "Symmetric"] = "Metal"

    port_extension: float = 1.0

    pml_layers: int = 12

    class Config:
        """pydantic basemodel config."""

        arbitrary_types_allowed = True


class ConvergenceSettingsLumericalEme(BaseModel):
    sparam_diff: float = 0.01

    class Config:
        arbitrary_types_allowed = True


LUMERICAL_EME_SIMULATION_SETTINGS = SimulationSettingsLumericalEme()
LUMERICAL_EME_CONVERGENCE_SETTINGS = ConvergenceSettingsLumericalEme()


class LumericalEmeSimulation:
    def __init__(
        self,
        component: Component,
        material_map: dict[str, str],
        layerstack: LayerStack | None = None,
        session: lumapi.MODE | None = None,
        simulation_settings: SimulationSettingsLumericalEme = LUMERICAL_EME_SIMULATION_SETTINGS,
        convergence_settings: ConvergenceSettingsLumericalEme = LUMERICAL_EME_CONVERGENCE_SETTINGS,
        dirpath: PathType | None = "",
        hide: bool = False,
        **settings,
    ):
        # Set up variables
        dirpath = dirpath or Path(__file__).resolve().parent
        simulation_settings = dict(simulation_settings)

        if hasattr(component.info, "simulation_settings"):
            simulation_settings |= component.info.simulation_settings
            logger.info(
                f"Updating {component.name!r} sim settings {component.simulation_settings}"
            )
        for setting in settings:
            if setting not in simulation_settings:
                raise ValueError(
                    f"Invalid setting {setting!r} not in ({list(simulation_settings.keys())})"
                )

        simulation_settings.update(**settings)
        ss = SimulationSettingsLumericalEme(**simulation_settings)

        # Save class variables
        self.component = component
        self.material_map = material_map
        self.simulation_settings = ss
        self.convergence_settings = convergence_settings
        self.layerstack = layerstack

        # Set up EME simulation based on provided simulation settings
        if not session:
            session = lumapi.MODE(hide=hide)
        self.session = session

        ports = component.get_ports_list(port_type="optical")
        if not ports:
            raise ValueError(f"{component.name!r} does not have any optical ports")
        if len(ports) > 2:
            raise ValueError(
                f"{component.name!r} has more than 2 ports. EME only supports 2 port devices."
            )

        # Extend component ports beyond simulation region
        component_with_booleans = layerstack.get_component_with_derived_layers(
            component
        )
        component_with_padding = gf.add_padding_container(
            component_with_booleans, default=0
        )

        component_extended = gf.components.extend_ports(
            component_with_padding, length=ss.port_extension
        )

        component_extended_beyond_pml = gf.components.extension.extend_ports(
            component=component_extended, length=ss.port_extension
        )
        component_extended_beyond_pml.name = "top"
        gdspath = component_extended_beyond_pml.write_gds()
