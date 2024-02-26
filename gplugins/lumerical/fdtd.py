"""Write Sparameters with Lumerical FDTD."""
from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path
from gdsfactory.config import __version__, logger
from gdsfactory.component import Component
from gplugins.lumerical.simulation_settings import (
    SIMULATION_SETTINGS_LUMERICAL_FDTD,
    SimulationSettingsLumericalFdtd,
)
from gplugins.lumerical.convergence_settings import (
    ConvergenceSettingsLumericalFdtd,
    LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
)
from gdsfactory.pdk import get_layer_stack
from gdsfactory.technology import LayerStack

from gplugins.common.utils.get_sparameters_path import (
    get_sparameters_path_lumerical as get_sparameters_path,
)
from gplugins.lumerical.utils import draw_geometry, layerstack_to_lbr
import gdsfactory as gf
import numpy as np
import yaml
import pandas as pd
import shutil
import time


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
    from gdsfactory.typings import ComponentSpec, MaterialSpec, PathType

run_false_warning = """
You have passed run=False to debug the simulation

run=False returns the simulation session for you to debug and make sure it is correct

To compute the Sparameters you need to pass run=True
"""

um = 1e-6

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

    sim = LumericalFdtdSimulation(component=taper,
                                  material_map=layer_map,
                                  hide=False
                                  )
    sp = sim.write_sparameters()
    print('Done')

class LumericalFdtdSimulation:
    """
        Lumerical FDTD simulation

        Set up FDTD simulation based on component geometry and simulation settings. Optionally, run convergence.

        Attributes:
            component: Component geometry to simulate
            material_map: Map of PDK materials to Lumerical materials
            layerstack: PDK layerstack
            session: Lumerical session
            simulation_settings: EME simulation settings
            convergence_settings: EME convergence settings
            dirpath: Directory where simulation files are saved
            filepath: S-parameter filepath (npz)
            mesh_convergence_data: Mesh convergence results

        """
    def __init__(self,
                 component: Component,
                 material_map: dict[str, str],
                 layerstack: LayerStack | None = None,
                 session: lumapi.FDTD | None = None,
                 simulation_settings: SimulationSettingsLumericalFdtd = SIMULATION_SETTINGS_LUMERICAL_FDTD,
                 convergence_settings: ConvergenceSettingsLumericalFdtd = LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
                 dirpath: PathType | None = "",
                 hide: bool = True,
                 run_mesh_convergence: bool = False,
                 xmargin: float = 0,
                 ymargin: float = 0,
                 xmargin_left: float = 0,
                 xmargin_right: float = 0,
                 ymargin_top: float = 0,
                 ymargin_bot: float = 0,
                 zmargin: float = 1.0,
                 **settings,
                 ):
        r"""Creates FDTD simulation for extracting s-parameters

            Your components need to have ports, that will extend over the PML.

            .. image:: https://i.imgur.com/dHAzZRw.png

            For your Fab technology you can overwrite

            - simulation_settings
            - dirpath
            - layerStack

            converts gdsfactory units (um) to Lumerical units (m)

            Disclaimer: This function tries to create a generalized FDTD simulation to extract Sparameters.
            It is hard to make a function that will fit all your possible simulation settings.
            You can use this function for inspiration to create your own.

            Args:
                component: Component to simulate.
                material_map: Map of PDK materials to Lumerical materials
                layerstack: PDK layerstack
                session: you can pass a session=lumapi.FDTD() or it will create one.
                simulation_settings: dataclass with all simulation_settings.
                convergence_settings: FDTD convergence settings
                dirpath: Directory where simulation files and sparams (.npz) are saved
                    Defaults to active Pdk.sparameters_path.
                hide: Hide simulation if True, else show GUI
                run_mesh_convergence: If True, run sweep of mesh and monitor sparam convergence.
                xmargin: left/right distance from component to PML.
                xmargin_left: left distance from component to PML.
                xmargin_right: right distance from component to PML.
                ymargin: left/right distance from component to PML.
                ymargin_top: top distance from component to PML.
                ymargin_bot: bottom distance from component to PML.
                zmargin: thickness for cladding above and below core.

            Keyword Args:
                background_material: for the background.
                port_margin: on both sides of the port width (um).
                port_height: port height (um).
                port_extension: port extension (um).
                mesh_accuracy: 2 (1: coarse, 2: fine, 3: superfine).
                wavelength_start: 1.2 (um).
                wavelength_stop: 1.6 (um).
                wavelength_points: 500.
                simulation_time: (s) related to max path length 3e8/2.4*10e-12*1e6 = 1.25mm.
                simulation_temperature: in kelvin (default = 300).
                frequency_dependent_profile: computes mode profiles for different wavelengths.
                field_profile_samples: number of wavelengths to compute field profile.


            .. code::

                 top view
                      ________________________________
                     |                               |
                     | xmargin                       | port_extension
                     |<------>          port_margin ||<-->
                  o2_|___________          _________||_o3
                     |           \        /          |
                     |            \      /           |
                     |             ======            |
                     |            /      \           |
                  o1_|___________/        \__________|_o4
                     |   |                           |
                     |   |ymargin                    |
                     |   |                           |
                     |___|___________________________|

                side view
                      ________________________________
                     |                               |
                     |                               |
                     |                               |
                     |ymargin                        |
                     |<---> _____         _____      |
                     |     |     |       |     |     |
                     |     |     |       |     |     |
                     |     |_____|       |_____|     |
                     |       |                       |
                     |       |                       |
                     |       |zmargin                |
                     |       |                       |
                     |_______|_______________________|

            """
        self.dirpath = dirpath = dirpath or Path(__file__).resolve().parent
        self.convergence_settings = convergence_settings = convergence_settings or LUMERICAL_FDTD_CONVERGENCE_SETTINGS
        self.component = component = gf.get_component(component)
        sim_settings = dict(simulation_settings)

        self.layerstack = layer_stack = layerstack or get_layer_stack()

        layer_to_thickness = layer_stack.get_layer_to_thickness()
        layer_to_zmin = layer_stack.get_layer_to_zmin()

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
        self.simulation_settings = ss = SimulationSettingsLumericalFdtd(**sim_settings)

        component_with_booleans = layer_stack.get_component_with_derived_layers(component)
        component_with_padding = gf.add_padding_container(
            component_with_booleans,
            default=0,
            top=ymargin or ymargin_top,
            bottom=ymargin or ymargin_bot,
            left=xmargin or xmargin_left,
            right=xmargin or xmargin_right,
        )

        component_extended = gf.components.extend_ports(
            component_with_padding, length=ss.distance_monitors_to_pml
        )

        ports = component.get_ports_list(port_type="optical")
        if not ports:
            raise ValueError(f"{component.name!r} does not have any optical ports")

        component_extended_beyond_pml = gf.components.extension.extend_ports(
            component=component_extended, length=ss.port_extension
        )
        component_extended_beyond_pml.name = "top"
        gdspath = component_extended_beyond_pml.write_gds()

        x_min = (component_extended.xmin - xmargin) * um
        x_max = (component_extended.xmax + xmargin) * um
        y_min = (component_extended.ymin - ymargin) * um
        y_max = (component_extended.ymax + ymargin) * um

        layers_thickness = [
            layer_to_thickness[layer]
            for layer in component_with_booleans.get_layers()
            if layer in layer_to_thickness
        ]
        if not layers_thickness:
            raise ValueError(
                f"no layers for component {component.get_layers()}"
                f"in layer stack {layer_stack}"
            )
        layers_zmin = [
            layer_to_zmin[layer]
            for layer in component_with_booleans.get_layers()
            if layer in layer_to_zmin
        ]
        component_thickness = max(layers_thickness)
        component_zmin = min(layers_zmin)

        z = (component_zmin + component_thickness) / 2 * um
        z_span = (2 * zmargin + component_thickness) * um

        x_span = x_max - x_min
        y_span = y_max - y_min

        sim_settings.update(dict(layer_stack=layer_stack.to_dict()))

        sim_settings = dict(
            simulation_settings=sim_settings,
            component=component.to_dict(),
            version=__version__,
        )

        logger.info(
            f"Simulation size = {x_span / um:.3f}, {y_span / um:.3f}, {z_span / um:.3f} um"
        )

        self.session = s = session or lumapi.FDTD(hide=hide)
        s.newproject()
        s.selectall()
        s.deleteall()

        material_name_to_lumerical_new = material_map or {}
        material_name_to_lumerical = ss.material_name_to_lumerical.copy()
        material_name_to_lumerical.update(**material_name_to_lumerical_new)
        self.material_map = material_name_to_lumerical

        s.addfdtd(
            dimension="3D",
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z=z,
            z_span=z_span,
            mesh_accuracy=ss.mesh_accuracy,
            use_early_shutoff=True,
            simulation_time=ss.simulation_time,
            simulation_temperature=ss.simulation_temperature,
        )

        ### Create Layer Builder object and insert geometry
        process_file_path = layerstack_to_lbr(
            material_name_to_lumerical, layer_stack, dirpath
        )
        draw_geometry(s, gdspath, process_file_path)

        # Add ports
        for i, port in enumerate(ports):
            zmin = layer_to_zmin[port.layer]
            thickness = layer_to_thickness[port.layer]
            z = (zmin + thickness) / 2
            zspan = 2 * ss.port_margin + thickness

            s.addport()
            p = f"FDTD::ports::port {i + 1}"
            s.setnamed(p, "x", port.x * um)
            s.setnamed(p, "y", port.y * um)
            s.setnamed(p, "z", z * um)
            s.setnamed(p, "z span", zspan * um)
            s.setnamed(p, "frequency dependent profile", ss.frequency_dependent_profile)
            s.setnamed(p, "number of field profile samples", ss.field_profile_samples)

            deg = port.orientation
            if -45 <= deg <= 45:
                direction = "Backward"
                injection_axis = "x-axis"
                dxp = 0
                dyp = 2 * ss.port_margin + port.width
            elif 45 < deg < 90 + 45:
                direction = "Backward"
                injection_axis = "y-axis"
                dxp = 2 * ss.port_margin + port.width
                dyp = 0
            elif 90 + 45 < deg < 180 + 45:
                direction = "Forward"
                injection_axis = "x-axis"
                dxp = 0
                dyp = 2 * ss.port_margin + port.width
            elif 180 + 45 < deg < 180 + 45 + 90:
                direction = "Forward"
                injection_axis = "y-axis"
                dxp = 2 * ss.port_margin + port.width
                dyp = 0

            else:
                raise ValueError(
                    f"port {port.name!r} orientation {port.orientation} is not valid"
                )

            s.setnamed(p, "direction", direction)
            s.setnamed(p, "injection axis", injection_axis)
            s.setnamed(p, "y span", dyp * um)
            s.setnamed(p, "x span", dxp * um)
            s.setnamed(p, "name", port.name)

            logger.info(
                f"port {p} {port.name!r}: at ({port.x}, {port.y}, 0)"
                f"size = ({dxp}, {dyp}, {zspan})"
            )

        s.setglobalsource("wavelength start", ss.wavelength_start * um)
        s.setglobalsource("wavelength stop", ss.wavelength_stop * um)
        s.setnamed("FDTD::ports", "monitor frequency points", ss.wavelength_points)

        # Add base sparam sweep
        s.deletesweep("s-parameter sweep")
        s.addsweep(3)
        s.setsweep("s-parameter sweep", "Excite all ports", 1)
        s.setsweep("S sweep", "auto symmetry", True)

        # Save simulation and settings
        self.filepath = get_sparameters_path(
            component=component,
            dirpath=dirpath,
            layer_stack=layer_stack,
            **settings,
        )
        filepath_dat = self.filepath.with_suffix(".dat")
        filepath_sim_settings = filepath_dat.with_suffix(".yml")
        filepath_fsp = filepath_dat.with_suffix(".fsp")

        s.save(str(filepath_fsp))
        filepath_sim_settings.write_text(yaml.dump(sim_settings))

    def write_sparameters(self, run: bool = True, overwrite: bool = False, delete_fsp_files: bool = True) -> pd.DataFrame:
        s = self.session

        filepath = self.filepath.with_suffix(".dat")
        filepath_sim_settings = filepath.with_suffix(".yml")
        fspdir = filepath.parent / f"{filepath.stem}_s-parametersweep"
        filepath_fsp = filepath.with_suffix(".fsp")

        if run and self.filepath.exists() and not overwrite:
            logger.info(f"Reading Sparameters from {self.filepath.absolute()!r}")
            return np.load(self.filepath)

        if not run and self.session is None:
            print(run_false_warning)

        logger.info(f"Writing Sparameters to {self.filepath.absolute()!r}")

        start = time.time()
        if run:
            s.save(str(filepath_fsp))
            s.runsweep("s-parameter sweep")
            sp = s.getsweepresult("s-parameter sweep", "S parameters")
            s.exportsweep("s-parameter sweep", str(filepath))
            logger.info(f"wrote sparameters to {str(filepath)!r}")

            sp["wavelengths"] = sp.pop("lambda").flatten() / um
            np.savez_compressed(filepath, **sp)

            # keys = [key for key in sp.keys() if key.startswith("S")]
            # ra = {
            #     f"{key.lower()}a": list(np.unwrap(np.angle(sp[key].flatten())))
            #     for key in keys
            # }
            # rm = {f"{key.lower()}m": list(np.abs(sp[key].flatten())) for key in keys}
            # results = {"wavelengths": wavelengths}
            # results.update(ra)
            # results.update(rm)
            # df = pd.DataFrame(results, index=wavelengths)
            # df.to_csv(filepath_npz, index=False)

            end = time.time()
            sim_settings = self.simulation_settings.model_dump()
            sim_settings.update(compute_time_seconds=end - start)
            sim_settings.update(compute_time_minutes=(end - start) / 60)
            filepath_sim_settings.write_text(yaml.dump(sim_settings))
            if delete_fsp_files and fspdir.exists():
                shutil.rmtree(fspdir)
                logger.info(
                    f"deleting simulation files in {str(fspdir)!r}. "
                    "To keep them, use delete_fsp_files=False flag"
                )

            return sp


if __name__ == "__main__":
    # import lumapi
    #
    # s = lumapi.FDTD()
    # component = gf.components.mmi1x2()
    # material_name_to_lumerical = dict(
    #     si="Si (Silicon) - Palik",
    #     substrate="Si (Silicon) - Palik",
    #     box="SiO2 (Glass) - Palik",
    #     clad="SiO2 (Glass) - Palik",
    # )  # or dict(si=3.45+2j)
    #
    # r = write_sparameters_lumerical(
    #     component=component,
    #     material_name_to_lumerical=material_name_to_lumerical,
    #     run=False,
    #     session=s,
    # )
    main()
