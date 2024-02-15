"""
Lumerical EME Plugin

Author: Sean Lam
Contact: seanl@ece.ubc.ca
"""

from __future__ import annotations

from pathlib import Path

import gdsfactory as gf
import lumapi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gdsfactory.component import Component
from gdsfactory.config import logger
from gdsfactory.pdk import get_layer_stack
from gdsfactory.technology.layer_stack import LayerStack
from gdsfactory.typings import PathType

from gplugins.lumerical.convergence_settings import (
    LUMERICAL_EME_CONVERGENCE_SETTINGS,
    ConvergenceSettingsLumericalEme,
)
from gplugins.lumerical.simulation_settings import (
    LUMERICAL_EME_SIMULATION_SETTINGS,
    SimulationSettingsLumericalEme,
)
from gplugins.lumerical.utils import draw_geometry, layerstack_to_lbr

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
        cross_section1=xs_wg, cross_section2=xs_wg_wide, length=5
    )

    layer_map = {
        "si": "Si (Silicon) - Palik",
        "sio2": "SiO2 (Glass) - Palik",
        "sin": "Si3N4 (Silicon Nitride) - Phillip",
        "TiN": "TiN - Palik",
        "Aluminum": "Al (Aluminium) Palik",
    }
    sim = LumericalEmeSimulation(
        taper,
        layer_map,
        run_mesh_convergence=True,
        run_cell_convergence=True,
        run_mode_convergence=True,
        hide=False,
    )

    sim.plot_length_sweep()

    print("done")


class LumericalEmeSimulation:
    """
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
        run_mesh_convergence: If True, run sweep of mesh and monitor sparam convergence.
        run_cell_convergence: If True, run sweep of number of cells in central group span and monitor sparam convergence.
        run_mode_convergence: If True, run sweep of number of modes and monitor sparam convergence.

    Attributes:
        component: Component geometry to simulate
        material_map: Map of PDK materials to Lumerical materials
        layerstack: PDK layerstack
        session: Lumerical session
        simulation_settings: EME simulation settings
        convergence_settings: EME convergence settings
        dirpath: Directory where simulation files are saved

    """

    def __init__(
        self,
        component: Component,
        material_map: dict[str, str],
        layerstack: LayerStack | None = None,
        session: lumapi.MODE | None = None,
        simulation_settings: SimulationSettingsLumericalEme = LUMERICAL_EME_SIMULATION_SETTINGS,
        convergence_settings: ConvergenceSettingsLumericalEme = LUMERICAL_EME_CONVERGENCE_SETTINGS,
        dirpath: PathType | None = "",
        hide: bool = True,
        run_mesh_convergence: bool = False,
        run_cell_convergence: bool = False,
        run_mode_convergence: bool = False,
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
            raise ValueError(
                "Number of cell groups are not aligned.\n"
                + f"Group Cells ({len(ss.group_cells)}): {ss.group_cells}\n"
                + f"Group Subcell Methods ({len(ss.group_subcell_methods)}): {ss.group_subcell_methods}"
            )

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

        process_file_path = layerstack_to_lbr(material_map, layerstack, dirpath)

        # Create device geometry
        draw_geometry(s, gdspath, process_file_path)

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

        s.addeme()
        s.set("display cells", 1)
        s.set("x min", x_min)
        s.set("y min", y_min)
        s.set("y max", y_max)
        s.set("z", z)
        s.set("z span", z_span)

        s.set("wavelength", ss.wavelength * um)
        s.setemeanalysis("Wavelength sweep", 1)
        s.setemeanalysis("start wavelength", ss.wavelength_start)
        s.setemeanalysis("stop wavelength", ss.wavelength_stop)

        s.set("number of cell groups", len(ss.group_cells))
        s.set("cells", np.array(ss.group_cells))

        # Use component bounds for the group spans
        group_spans = []
        mid_span = (x_max - x_min - 2 * ss.port_extension * um) / (
            len(ss.group_cells) - 2
        )
        for i in range(0, len(ss.group_cells)):
            if i == 0 or i == len(ss.group_cells) - 1:
                group_spans.append(ss.port_extension * um)
            else:
                group_spans.append(mid_span)
        group_spans = np.array(group_spans)

        s.set("group spans", group_spans)

        # Convert subcell methods to int for Lumerical to interpret
        # 1 = CVCS, 0 = none
        subcell_methods = []
        for method in ss.group_subcell_methods:
            if method == "CVCS":
                subcell_methods.append(1)
            else:
                subcell_methods.append(0)
        s.set("subcell method", np.array(subcell_methods))

        s.set("number of modes for all cell groups", ss.num_modes)
        s.set("energy conservation", ss.energy_conservation)

        s.set("define y mesh by", "maximum mesh step")
        s.set("define z mesh by", "maximum mesh step")

        s.set("dy", ss.wavelength / ss.mesh_cells_per_wavelength * um)
        s.set("dz", ss.wavelength / ss.mesh_cells_per_wavelength * um)

        s.set("y min bc", ss.ymin_boundary)
        s.set("y max bc", ss.ymax_boundary)
        s.set("z min bc", ss.zmin_boundary)
        s.set("z max bc", ss.zmax_boundary)

        s.set("pml layers", ss.pml_layers)

        s.save(str(dirpath / f"{component.name}.lms"))

        if run_mesh_convergence:
            self.update_mesh_convergence(plot=not hide)

        if run_cell_convergence:
            self.update_cell_convergence(plot=not hide)

        if run_mode_convergence:
            self.update_mode_convergence(plot=not hide)

        if not hide:
            plt.show()

    def update_mesh_convergence(self, plot: bool = False) -> pd.DataFrame:
        """
        Update simulation based on mesh convergence testing. Updates both Lumerical session and simulation settings.

        Parameters:
            plot: Plot and save convergence results

        Returns:
            Convergence results
        """

        s = self.session
        cs = self.convergence_settings
        ss = self.simulation_settings

        s21 = []
        s11 = []
        mesh_cells_per_wavl = []
        converged = False
        while not converged:
            s.switchtolayout()
            s.set("dy", ss.wavelength / ss.mesh_cells_per_wavelength * um)
            s.set("dz", ss.wavelength / ss.mesh_cells_per_wavelength * um)
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
                sparam_diff = max(
                    [
                        max(np.diff(s21[-(cs.passes + 1) : -1])),
                        max(np.diff(s11[-(cs.passes + 1) : -1])),
                    ]
                )
                if sparam_diff < cs.sparam_diff:
                    converged = True
                else:
                    converged = False

        if plot:
            plt.figure()
            plt.plot(mesh_cells_per_wavl, s21)
            plt.plot(mesh_cells_per_wavl, s11)
            plt.legend(["|S21|^2", "|S11|^2"])
            plt.grid("on")
            plt.xlabel("Mesh Cells Per Wavelength")
            plt.ylabel("Magnitude")
            plt.title(f"Mesh Convergence | Wavelength={ss.wavelength}um")
            plt.savefig(str(self.dirpath / "mesh_convergence.png"))

        return pd.DataFrame.from_dict(
            {"num_cells": mesh_cells_per_wavl, "s21": list(s21), "s11": list(s11)}
        )

    def update_cell_convergence(self, plot: bool = False) -> pd.DataFrame:
        """
        Update simulation based on cell convergence testing (number of slices across the device center).
        Updates both Lumerical session and simulation settings.

        Parameters:
            plot: Plot and save convergence results

        Returns:
            Convergence results
        """
        s = self.session
        cs = self.convergence_settings
        ss = self.simulation_settings

        s21 = []
        s11 = []
        num_cells = []
        converged = False
        while not converged:
            s.switchtolayout()
            s.setnamed("EME", "cells", np.array(ss.group_cells))
            s.run()
            s.emepropagate()
            S = s.getresult("EME", "user s matrix")
            s11.append(abs(S[0, 0]) ** 2)
            s21.append(abs(S[1, 0]) ** 2)
            num_cells.append(ss.group_cells[1:-1])

            for i in range(1, len(ss.group_cells) - 1):
                ss.group_cells[i] += 1

            # Check whether convergence has been reached
            if len(s21) > cs.passes or len(s11) > cs.passes:
                # Calculate maximum diff in sparams
                sparam_diff = max(
                    [
                        max(np.diff(s21[-(cs.passes + 1) : -1])),
                        max(np.diff(s11[-(cs.passes + 1) : -1])),
                    ]
                )
                if sparam_diff < cs.sparam_diff:
                    converged = True
                else:
                    converged = False

        num_cells = np.array(num_cells)
        if plot:
            plt.figure()
            plt.plot(list(num_cells[:, 0]), s21)
            plt.plot(list(num_cells[:, 0]), s11)
            plt.xticks(list(num_cells[:, 0]), [f"{list(row)}" for row in num_cells])
            plt.setp(plt.xticks()[1], rotation=75, horizontalalignment="center")
            plt.legend(["|S21|^2", "|S11|^2"])
            plt.grid("on")
            plt.xlabel("Number of Cells")
            plt.ylabel("Magnitude")
            plt.title(f"Cell Convergence | Wavelength={ss.wavelength}um")
            plt.tight_layout()
            plt.savefig(str(self.dirpath / "cell_convergence.png"))

        return pd.DataFrame.from_dict(
            {"num_cells": list(num_cells[:, 0]), "s21": list(s21), "s11": list(s11)}
        )

    def update_mode_convergence(self, plot: bool = False) -> pd.DataFrame:
        """
        Update simulation based on mode convergence testing (number of modes required to be accurate).
        Updates both Lumerical session and simulation settings.

        Parameters:
            plot: Plot and save convergence results

        Returns:
            Convergence results
        """
        s = self.session
        cs = self.convergence_settings
        ss = self.simulation_settings

        converged = False
        while not converged:
            s.switchtolayout()
            s.setnamed("EME", "number of modes for all cell groups", ss.num_modes)
            s.run()
            s.emepropagate()

            s.setemeanalysis("Mode convergence sweep", 1)
            s.emesweep("mode convergence sweep")

            # get mode convergence sweep result
            S = s.getemesweep("S_mode_convergence_sweep")

            # plot S21 vs number of modes
            s21 = abs(S["s21"]) ** 2
            s11 = abs(S["s11"]) ** 2
            modes = S["modes"]

            # Check whether convergence has been reached
            if len(s21) > cs.passes or len(s11) > cs.passes:
                # Calculate maximum diff in sparams
                sparam_diff = max(
                    [
                        max(np.diff(s21[-(cs.passes + 1) : -1])),
                        max(np.diff(s11[-(cs.passes + 1) : -1])),
                    ]
                )
                if sparam_diff < cs.sparam_diff:
                    converged = True
                    break
                else:
                    converged = False

            ss.num_modes += 5

        if plot:
            plt.figure()
            plt.plot(modes, s21)
            plt.plot(modes, s11)
            plt.legend(["|S21|^2", "|S11|^2"])
            plt.grid("on")
            plt.xlabel("Number of Modes")
            plt.ylabel("Magnitude")
            plt.title(f"Mode Convergence | Wavelength={ss.wavelength}um")
            plt.tight_layout()
            plt.savefig(str(self.dirpath / "mode_convergence.png"))

        return pd.DataFrame.from_dict(
            {"modes": list(modes), "s21": list(s21), "s11": list(s11)}
        )

    def get_length_sweep(
        self,
        start_length: float = 1,
        stop_length: float = 100,
        num_pts: int = 100,
        group: int = 2,
    ) -> pd.DataFrame:
        """
        Get length sweep sparams.

        Parameters:
            start_length: Start length (um)
            stop_length: Stop length (um)
            num_pts: Number of points along sweep
            group: Group span to sweep

        Returns:
            Dataframe with length sweep and complex sparams
        """

        s = self.session
        s.run()
        s.emepropagate()

        # set propagation sweep settings
        s.setemeanalysis("propagation sweep", 1)
        s.setemeanalysis("parameter", f"group span {group}")
        s.setemeanalysis("start", start_length * um)
        s.setemeanalysis("stop", stop_length * um)
        s.setemeanalysis("number of points", num_pts)

        s.emesweep()

        S = s.getemesweep("S")

        # Get sparams
        s21 = list(S["s21"])
        s22 = list(S["s22"])
        s11 = list(S["s11"])
        s12 = list(S["s12"])
        group_span = list(S[f"group_span_{group}"])

        return pd.DataFrame.from_dict(
            {
                "length": group_span,
                "s11": s11,
                "s21": s21,
                "s12": s12,
                "s22": s22,
            }
        )

    def plot_length_sweep(
        self,
        start_length: float = 1,
        stop_length: float = 100,
        num_pts: int = 100,
        group: int = 2,
    ) -> None:
        """
        Plot length sweep.

        Parameters:
            start_length: Start length (um)
            stop_length: Stop length (um)
            num_pts: Number of points along sweep
            group: Group span to sweep

        Returns:
            Figure handle
        """
        sweep_data = self.get_length_sweep(start_length, stop_length, num_pts, group)

        fig = plt.figure()
        plt.plot(sweep_data.loc[:, "length"] / um, abs(sweep_data.loc[:, "s11"]) ** 2)
        plt.plot(sweep_data.loc[:, "length"] / um, abs(sweep_data.loc[:, "s21"]) ** 2)
        plt.plot(sweep_data.loc[:, "length"] / um, abs(sweep_data.loc[:, "s12"]) ** 2)
        plt.plot(sweep_data.loc[:, "length"] / um, abs(sweep_data.loc[:, "s22"]) ** 2)
        plt.legend(["|S11|^2", "|S21|^2", "|S12|^2", "|S22|^2"])
        plt.grid("on")
        plt.xlabel("Length (um)")
        plt.ylabel("Magnitude")
        plt.title(f"Length Sweep | Wavelength={self.simulation_settings.wavelength}um")
        plt.tight_layout()
        plt.savefig(str(self.dirpath / "length_sweep.png"))

        return fig


if __name__ == "__main__":
    main()
