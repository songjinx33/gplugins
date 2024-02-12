from __future__ import annotations

from typing import Literal
from pydantic import BaseModel
from gdsfactory.component import Component
from gdsfactory.technology.layer_stack import LayerStack
from gdsfactory.pdk import get_layer_stack
from gdsfactory.typings import PathType
from gdsfactory.config import logger
from gplugins.lumerical.utils import to_lbr
from pathlib import Path

import math
import lumapi
import gdsfactory as gf
import numpy as np
import matplotlib.pyplot as plt

um = 1e-6

def main():
    from gdsfactory.components.taper_cross_section import taper_cross_section
    from gdsfactory.cross_section import cross_section
    from functools import partial

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

    taper = taper_cross_section(cross_section1=xs_wg, cross_section2=xs_wg_wide, length=5)

    layer_map = {
        "si": "Si (Silicon) - Palik",
        "sio2": "SiO2 (Glass) - Palik",
        "sin": "Si3N4 (Silicon Nitride) - Phillip",
        "TiN": "TiN - Palik",
        "Aluminum": "Al (Aluminium) Palik",
    }
    sim = LumericalEmeSimulation(taper, layer_map, run_mesh_convergence=True)

    print('done')

class SimulationSettingsLumericalEme(BaseModel):
    """Lumerical EME simulation_settings.

    Parameters:
        wavelength: Wavelength (um)
        wavelength_start: Starting wavelength in wavelength range (um)
        wavelength_stop: Stopping wavelength in wavelength range (um)
        material_fit_tolerance: Material fit coefficient
        group_cells: Number of cells in each group
        group_spans: Span size in each group (um)
        group_subcell_methods: Methods to analyze each cross section
        num_modes: Number of modes
        energy_conservation: Ensure results are passive or conserve energy.
        mesh_cells_per_wavelength: Number of mesh cells per wavelength
        ymin_boundary: y min boundary condition
        ymax_boundary: y max boundary condition
        zmin_boundary: z min boundary condition
        zmax_boundary: z max boundary condition
        port_extension: Port extension beyond the simulation boundary (um)
        pml_layers: Number of PML layers used if PML boundary conditions used.
        ymargin: Y margin from component to simulation boundary (um)
        zmargin: Z margin from component to simulation boundary (um)
    """

    wavelength: float = 1.55
    wavelength_start: float = 1.5
    wavelength_stop: float = 1.6
    material_fit_tolerance: float = 0.001

    group_cells: list[int] = [1, 30, 1]
    group_subcell_methods: list[Literal["CVCS"] | None] = [None, "CVCS", None]
    num_modes: int = 30
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

    ymargin: float = 2.0
    zmargin: float = 1.0


    class Config:
        """pydantic basemodel config."""

        arbitrary_types_allowed = True


class ConvergenceSettingsLumericalEme(BaseModel):
    passes: int = 10
    sparam_diff: float = 0.01

    class Config:
        arbitrary_types_allowed = True


LUMERICAL_EME_SIMULATION_SETTINGS = SimulationSettingsLumericalEme()
LUMERICAL_EME_CONVERGENCE_SETTINGS = ConvergenceSettingsLumericalEme()


class LumericalEmeSimulation:
    '''
    Lumerical EME simulation

    Set up EME simulation based on component geometry and simulation settings. Optionally, run convergence.

    Parameters:
        component: Component geometry to simulate
        material_map: Map of PDK materials to Lumerical materials
        layerstack: PDK layerstack
        session: Lumerical session
        simulation_settings: EME simulation settings
        convergence_settings: EME convergence settings
        dirpath: Directory where simulation files are saved
        hide: Hide simulation if True, else show GUI

    Attributes:
        component: Component geometry to simulate
        material_map: Map of PDK materials to Lumerical materials
        layerstack: PDK layerstack
        session: Lumerical session
        simulation_settings: EME simulation settings
        convergence_settings: EME convergence settings
        dirpath: Directory where simulation files are saved

    '''

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
        run_mesh_convergence: bool = False,
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

        # Check number of cell groups are aligned
        if not (len(ss.group_cells) == len(ss.group_subcell_methods)):
            raise ValueError(f'Number of cell groups are not aligned.\n' +
                             f'Group Cells ({len(ss.group_cells)}): {ss.group_cells}\n' +
                             f'Group Subcell Methods ({len(ss.group_subcell_methods)}): {ss.group_subcell_methods}')

        layerstack = layerstack or get_layer_stack()

        # Save instance variables
        self.component = component
        self.material_map = material_map
        self.simulation_settings = ss
        self.convergence_settings = convergence_settings
        self.layerstack = layerstack
        self.dirpath = dirpath

        # Set up EME simulation based on provided simulation settings
        if not session:
            session = lumapi.MODE(hide=hide)
        self.session = session
        s = session

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

        process_file_path = to_lbr(material_map, layerstack, dirpath)

        # Create device geometry
        s.addlayerbuilder()
        s.set("x", 0)
        s.set("y", 0)
        s.set("z", 0)
        s.loadgdsfile(str(gdspath))
        s.loadprocessfile(str(dirpath / "process.lbr"))

        # Fit material models
        for layer_name in layerstack.to_dict():
            s.select("layer group")
            material_name = s.getlayer(layer_name, "pattern material")
            try:
                s.setmaterial(material_name, "wavelength min", ss.wavelength_start * um)
                s.setmaterial(material_name, "wavelength max", ss.wavelength_stop * um)
                s.setmaterial(material_name, "tolerance", ss.material_fit_tolerance)
            except lumapi.LumApiError:
                logger.warning(
                    f"Material {material_name} cannot be found in database, skipping material fit."
                )

        # Create simulation region
        x_min = component_extended.xmin * um
        x_max = component_extended.xmax * um
        y_min = (component_extended.ymin - ss.ymargin) * um
        y_max = (component_extended.ymax + ss.ymargin) * um

        layer_to_thickness = layerstack.get_layer_to_thickness()
        layer_to_zmin = layerstack.get_layer_to_zmin()
        layers_thickness = [
            layer_to_thickness[layer]
            for layer in component_with_booleans.get_layers()
            if layer in layer_to_thickness
        ]
        if not layers_thickness:
            raise ValueError(
                f"no layers for component {component.get_layers()}"
                f"in layer stack {layerstack}"
            )
        layers_zmin = [
            layer_to_zmin[layer]
            for layer in component_with_booleans.get_layers()
            if layer in layer_to_zmin
        ]
        component_thickness = max(layers_thickness)
        component_zmin = min(layers_zmin)

        z = (component_zmin + component_thickness) / 2 * um
        z_span = (2 * ss.zmargin + component_thickness) * um

        x_span = x_max - x_min
        y_span = y_max - y_min

        s.addeme()
        s.set('display cells', 1)
        s.set('x min', x_min)
        s.set('y min', y_min)
        s.set('y max', y_max)
        s.set('z', z)
        s.set('z span', z_span)

        s.set('wavelength', ss.wavelength * um)
        s.setemeanalysis('Wavelength sweep', 1)
        s.setemeanalysis('start wavelength', ss.wavelength_start)
        s.setemeanalysis('stop wavelength', ss.wavelength_stop)

        s.set('number of cell groups', len(ss.group_cells))
        s.set('cells', np.array(ss.group_cells))

        # Use component bounds for the group spans
        group_spans = []
        mid_span = (x_max - x_min - 2 * ss.port_extension * um) / (len(ss.group_cells) - 2)
        for i in range(0, len(ss.group_cells)):
            if i == 0 or i == len(ss.group_cells) - 1:
                group_spans.append(ss.port_extension * um)
            else:
                group_spans.append(mid_span)
        group_spans = np.array(group_spans)

        s.set('group spans', group_spans)

        # Convert subcell methods to int for Lumerical to interpret
        # 1 = CVCS, 0 = none
        subcell_methods = []
        for method in ss.group_subcell_methods:
            if method == 'CVCS':
                subcell_methods.append(1)
            else:
                subcell_methods.append(0)
        s.set('subcell method', np.array(subcell_methods))

        s.set('number of modes for all cell groups', ss.num_modes)
        s.set('energy conservation', ss.energy_conservation)

        s.set('define y mesh by', 'maximum mesh step')
        s.set('define z mesh by', 'maximum mesh step')

        s.set('dy', ss.wavelength / ss.mesh_cells_per_wavelength * um)
        s.set('dz', ss.wavelength / ss.mesh_cells_per_wavelength * um)

        s.set('y min bc', ss.ymin_boundary)
        s.set('y max bc', ss.ymax_boundary)
        s.set('z min bc', ss.zmin_boundary)
        s.set('z max bc', ss.zmax_boundary)

        s.set('pml layers', ss.pml_layers)

        s.save(str(dirpath / f'{component.name}.lms'))

        if run_mesh_convergence:
            self.update_mesh_convergence(plot = True)

        if run_

    def update_mesh_convergence(self, plot: bool = False):
        s = self.session
        cs = self.convergence_settings
        ss = self.simulation_settings

        s21 = []
        s11 = []
        mesh_cells_per_wavl = []
        converged = False
        while not converged:
            s.switchtolayout()
            s.set('dy', ss.wavelength / ss.mesh_cells_per_wavelength * um)
            s.set('dz', ss.wavelength / ss.mesh_cells_per_wavelength * um)
            # Get sparams and refine mesh
            s.run()
            s.emepropagate()
            S = s.getresult("EME", "user s matrix")
            s11.append(abs(S[0, 0]) ** 2)
            s21.append(abs(S[1, 0]) ** 2)
            mesh_cells_per_wavl.append(ss.mesh_cells_per_wavelength)

            ss.mesh_cells_per_wavelength += 1

            # Check whether convergence has been reached
            if len(s21) > cs.passes or len(s11) > cs.passes:
                # Calculate maximum diff in sparams
                sparam_diff = max([max(np.diff(s21[-(cs.passes + 1):-1])), max(np.diff(s11[-(cs.passes + 1):-1]))])
                if sparam_diff < cs.sparam_diff:
                    converged = True
                else:
                    converged = False

        if plot:
            plt.figure()
            plt.plot(mesh_cells_per_wavl, s21)
            plt.plot(mesh_cells_per_wavl, s11)
            plt.legend(['|S21|^2', '|S11|^2'])
            plt.grid('on')
            plt.xlabel('Mesh Cells Per Wavelength')
            plt.ylabel('Magnitude')
            plt.title(f'Mesh Convergence Wavelength={ss.wavelength}um')
            plt.savefig(str(self.dirpath / 'mesh_convergence.png'))

    def update_cell_convergence(self, plot: bool = False):
        pass


    def update_mode_convergence(self, plot: bool = False):
        pass


if __name__ == "__main__":
    main()




