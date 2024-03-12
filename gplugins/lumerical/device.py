from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory.config import logger
from gdsfactory.pdk import get_layer_stack
from gdsfactory.technology import LayerStack

from gplugins.lumerical.convergence_settings import (
    LUMERICAL_CHARGE_CONVERGENCE_SETTINGS,
    ConvergenceSettingsLumericalCharge,
)
from gplugins.lumerical.simulation_settings import (
    LUMERICAL_CHARGE_SIMULATION_SETTINGS,
    SimulationSettingsLumericalCharge,
)
from gplugins.lumerical.utils import draw_geometry, layerstack_to_lbr

try:
    import lumapi
except ModuleNotFoundError as e:
    print(
        "Cannot import lumapi (Python Lumerical API). "
        "You can add set the PYTHONPATH variable or add it with `sys.path.append()`"
    )
    raise e
except OSError as e:
    raise e

if TYPE_CHECKING:
    from gdsfactory.typings import PathType

# Format for colors (R, G, B, opacity)
OPACITY = 0.4
MATERIAL_COLORS = [
    np.array([1, 0, 0, OPACITY]),
    np.array([1, 0.5, 0, OPACITY]),
    np.array([1, 1, 0, OPACITY]),
    np.array([0.5, 1, 0, OPACITY]),
    np.array([0, 1, 1, OPACITY]),
    np.array([0, 0.5, 1, OPACITY]),
    np.array([0, 0, 1, OPACITY]),
    np.array([0.5, 0, 1, OPACITY]),
    np.array([1, 0, 1, OPACITY]),
] * 10


def main():
    from functools import partial

    import gdsfactory as gf

    ### Create curved PN junction
    c = gf.Component()
    cross_section_pn = partial(
        gf.cross_section.pn,
        width_doping=2.425,
        width_slab=2 * 2.425,
        layer_via="VIAC",
        width_via=0.5,
        layer_metal="M1",
        width_metal=0.5,
    )

    doped_path = gf.Path()
    doped_path.append(gf.path.arc(radius=10.0, angle=361))
    c << doped_path.extrude(cross_section=cross_section_pn)

    ### TODO: Update generic PDK with dopants in layer_stack
    from gdsfactory.generic_tech.layer_map import LAYER
    from gdsfactory.pdk import get_layer_stack
    from gdsfactory.technology.layer_stack import LayerLevel

    layer_stack = get_layer_stack()
    layer_stack.layers["substrate"].layer_type = "background"
    layer_stack.layers["substrate"].background_doping_ion = None
    layer_stack.layers["substrate"].background_doping_concentration = None
    layer_stack.layers["box"].layer_type = "background"
    layer_stack.layers["clad"].layer_type = "background"
    layer_stack.layers["core"].sidewall_angle = 0
    layer_stack.layers["slab90"].sidewall_angle = 0
    layer_stack.layers["via_contact"].sidewall_angle = 0
    layer_stack.layers["N"] = LayerLevel(
        layer=LAYER.N,
        thickness=0.22,
        zmin=0,
        material="si",
        mesh_order=4,
        background_doping_concentration=5e17,
        background_doping_ion="n",
        orientation="100",
        layer_type="doping",
    )
    layer_stack.layers["P"] = LayerLevel(
        layer=LAYER.P,
        thickness=0.22,
        zmin=0,
        material="si",
        mesh_order=4,
        background_doping_concentration=7e17,
        background_doping_ion="p",
        orientation="100",
        layer_type="doping",
    )
    layer_stack.layers["NP"] = LayerLevel(
        layer=LAYER.NP,
        thickness=0.09,
        zmin=0,
        material="si",
        mesh_order=4,
        background_doping_concentration=3e18,
        background_doping_ion="n",
        orientation="100",
        layer_type="doping",
    )
    layer_stack.layers["PP"] = LayerLevel(
        layer=LAYER.PP,
        thickness=0.09,
        zmin=0,
        material="si",
        mesh_order=4,
        background_doping_concentration=2e18,
        background_doping_ion="p",
        orientation="100",
        layer_type="doping",
    )
    layer_stack.layers["NPP"] = LayerLevel(
        layer=LAYER.NPP,
        thickness=0.09,
        zmin=0,
        material="si",
        mesh_order=4,
        background_doping_concentration=1e19,
        background_doping_ion="n",
        orientation="100",
        layer_type="doping",
    )
    layer_stack.layers["PPP"] = LayerLevel(
        layer=LAYER.PPP,
        thickness=0.09,
        zmin=0,
        material="si",
        mesh_order=4,
        background_doping_concentration=1e19,
        background_doping_ion="p",
        orientation="100",
        layer_type="doping",
    )

    c.show()

    ### Set up simulation settings
    charge_settings = SimulationSettingsLumericalCharge(x=10, y=10)

    sim = LumericalChargeSimulation(
        component=c,
        layerstack=layer_stack,
        simulation_settings=charge_settings,
        convergence_settings=LUMERICAL_CHARGE_CONVERGENCE_SETTINGS,
        hide=False,
    )


class LumericalChargeSimulation:
    """
    Lumerical CHARGE simulation

    Set up CHARGE simulation based on component geometry and simulation settings. Optionally, run convergence.

    Attributes:
        component: Component geometry to simulate
        layerstack: PDK layerstack
        session: Lumerical session
        simulation_settings: CHARGE simulation settings
        convergence_settings: CHARGE convergence settings
        dirpath: Directory where simulation files are saved

    """

    def __init__(
        self,
        component: Component,
        layerstack: LayerStack | None = None,
        session: lumapi.DEVICE | None = None,
        simulation_settings: SimulationSettingsLumericalCharge = LUMERICAL_CHARGE_SIMULATION_SETTINGS,
        convergence_settings: ConvergenceSettingsLumericalCharge = LUMERICAL_CHARGE_CONVERGENCE_SETTINGS,
        dirpath: PathType | None = "",
        hide: bool = True,
        **settings,
    ):
        if isinstance(dirpath, str) and not dirpath == "":
            dirpath = Path(dirpath)
        self.dirpath = dirpath or Path(__file__).resolve().parent

        self.convergence_settings = (
            convergence_settings or LUMERICAL_CHARGE_CONVERGENCE_SETTINGS
        )
        self.component = gf.get_component(component)
        self.layerstack = layerstack or get_layer_stack()

        sim_settings = dict(simulation_settings)
        if hasattr(component.info, "simulation_settings"):
            sim_settings |= component.info.simulation_settings
            logger.info(
                f"Updating {component.name!r} sim settings {component.simulation_settings}"
            )
        for setting in settings:
            if setting not in sim_settings:
                raise ValueError(
                    f"Invalid setting {setting!r} not in ({list(sim_settings.keys())})"
                )

        sim_settings.update(**settings)
        self.simulation_settings = SimulationSettingsLumericalCharge(**sim_settings)

        ss = self.simulation_settings
        cs = self.convergence_settings

        ### Get new CHARGE session
        self.session = s = session or lumapi.DEVICE(hide=hide)
        s.newproject()
        s.deleteall()

        # Add materials before drawing geometry. This is necessary since Layer Builder references these materials
        self.add_charge_materials()

        # Draw geometry
        combined_material_map = ss.optical_material_name_to_lumerical.copy()
        combined_material_map.update(ss.ele_therm_material_name_to_lumerical)
        process_file_path = layerstack_to_lbr(
            material_map=combined_material_map,
            layerstack=self.layerstack,
            dirpath=self.dirpath,
            use_pdk_material_names=True,
        )
        gdspath = self.component.write_gds()
        draw_geometry(session=s, gdspath=gdspath, process_file_path=process_file_path)

        # Add and configure simulation region
        print()

        # Add and configure solver
        print()

    def add_charge_materials(
        self,
        optical_layer_map: dict[str, str] = None,
        ele_therm_layer_map: dict[str, str] = None,
        material_fit_tolerance: float | None = None,
    ):
        """
        Add materials to simulation.

        Materials need to be added prior to assigning these materials to structures.

        Parameters:
            optical_layer_map: Map of optical materials from PDK materials to Lumerical materials
            ele_therm_layer_map: Map of electrical and thermal materials from PDK materials to Lumerical materials
            material_fit_tolerance: Optical material fit tolerance
        """

        s = self.session
        optical_layer_map = (
            optical_layer_map
            or self.simulation_settings.optical_material_name_to_lumerical
        )
        ele_therm_layer_map = (
            ele_therm_layer_map
            or self.simulation_settings.ele_therm_material_name_to_lumerical
        )
        material_fit_tolerance = (
            material_fit_tolerance or self.simulation_settings.material_fit_tolerance
        )

        s.addmodelmaterial()
        ele_therm_materials = s.addmaterialproperties("CT").split("\n")
        opt_materials = s.addmaterialproperties("EM").split("\n")
        s.delete("New Material")

        # Add materials only supported by Lumerical
        i = 0
        for name, material in optical_layer_map.items():
            if material not in opt_materials:
                logger.warning(
                    f"{material} is not a supported optical material in Lumerical and will be skipped."
                )
                continue
            else:
                s.addmodelmaterial()
                s.set("name", name)
                s.set("color", MATERIAL_COLORS[i])
                s.addmaterialproperties("EM", material)
                s.set("tolerance", material_fit_tolerance)
                i += 1

        for name, material in ele_therm_layer_map.items():
            if material not in ele_therm_materials:
                logger.warning(
                    f"{material} is not a supported electrical or thermal material in Lumerical and will be skipped."
                )
                continue
            else:
                if not s.materialexists(name):
                    s.addmodelmaterial()
                    s.set("name", name)
                    s.set("color", MATERIAL_COLORS[i])
                    i += 1
                else:
                    s.select(f"materials::{name}")
                s.addmaterialproperties("CT", material)


if __name__ == "__main__":
    main()
