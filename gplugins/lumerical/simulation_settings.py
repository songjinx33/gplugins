from typing import Literal

from pydantic import BaseModel

simulation_settings_fdtd = [
    "allow grading in x",
    "allow grading in y",
    "allow grading in z",
    "allow symmetry on all boundaries",
    "always use complex fields",
    "angle phi",
    "angle theta",
    "auto shutoff max",
    "auto shutoff min",
    "background material",
    "bfast alpha",
    "bfast dt multiplier",
    "bloch units",
    "checkpoint at shutoff",
    "checkpoint during simulation",
    "checkpoint period",
    "conformal meshing refinement",
    "define x mesh by",
    "define y mesh by",
    "define z mesh by",
    "dimension",
    "direction",
    "down sample time",
    "dt",
    "dt stability factor",
    "dx",
    "dy",
    "dz",
    "enabled",
    "extend structure through pml",
    "force symmetric x mesh",
    "force symmetric y mesh",
    "force symmetric z mesh",
    "global monitor custom frequency samples",
    "global monitor frequency center",
    "global monitor frequency points",
    "global monitor frequency span",
    "global monitor maximum frequency",
    "global monitor maximum wavelength",
    "global monitor minimum frequency",
    "global monitor minimum wavelength",
    "global monitor sample spacing",
    "global monitor use source limits",
    "global monitor use wavelength spacing",
    "global monitor wavelength center",
    "global monitor wavelength span",
    "global source bandwidth",
    "global source center frequency",
    "global source center wavelength",
    "global source eliminate discontinuities",
    "global source frequency",
    "global source frequency span",
    "global source frequency start",
    "global source frequency stop",
    "global source offset",
    "global source optimize for short pulse",
    "global source pulse type",
    "global source pulselength",
    "global source set frequency",
    "global source set time domain",
    "global source set wavelength",
    "global source wavelength span",
    "global source wavelength start",
    "global source wavelength stop",
    "grading factor",
    "index",
    "injection axis",
    "kx",
    "ky",
    "kz",
    "max source time signal length",
    "mesh accuracy",
    "mesh allowed size increase factor",
    "mesh cells per wavelength",
    "mesh cells x",
    "mesh cells y",
    "mesh cells z",
    "mesh distance between fixed points",
    "mesh frequency max",
    "mesh frequency min",
    "mesh merge distance",
    "mesh minimum neighbor size",
    "mesh refinement",
    "mesh size reduction factor",
    "mesh step for metals",
    "mesh type",
    "mesh wavelength max",
    "mesh wavelength min",
    "meshing refinement",
    "meshing tolerance",
    "min mesh step",
    "nx",
    "ny",
    "nz",
    "override simulation bandwidth for mesh generation",
    "param1",
    "param2",
    "pml alpha",
    "pml alpha polynomial",
    "pml kappa",
    "pml layers",
    "pml max layers",
    "pml min layers",
    "pml polynomial",
    "pml profile",
    "pml sigma",
    "pml type",
    "same settings on all boundaries",
    "set based on source angle",
    "set process grid",
    "set simulation bandwidth",
    "simulation frequency max",
    "simulation frequency min",
    "simulation temperature",
    "simulation time",
    "simulation wavelength max",
    "simulation wavelength min",
    "snap pec to yee cell boundary",
    "source index",
    "type",
    "use auto shutoff",
    "use bfast fdtd",
    "use divergence checking",
    "use early shutoff",
    "use legacy conformal interface detection",
    "use mesh step for metals",
    "use relative coordinates",
]


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

    group_cells: list[int] = [1, 50, 1]
    group_subcell_methods: list[Literal["CVCS"] | None] = [None, "CVCS", None]
    num_modes: int = 30
    energy_conservation: Literal[
        "make passive", "conserve energy"
    ] | None = "make passive"

    mesh_cells_per_wavelength: int = 100

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


LUMERICAL_EME_SIMULATION_SETTINGS = SimulationSettingsLumericalEme()


material_name_to_lumerical_default = {
    "si": "Si (Silicon) - Palik",
    "sio2": "SiO2 (Glass) - Palik",
    "sin": "Si3N4 (Silicon Nitride) - Phillip",
    "tungsten": "W (tungsten) - Palik",
    "cu": "Cu (copper) - CRC",
    "air": "Air",
}


class SimulationSettingsLumericalFdtd(BaseModel):
    """Lumerical FDTD simulation_settings.

    Parameters:
        background_material: for the background.
        port_margin: on both sides of the port width (um).
        port_height: port height (um).
        port_extension: port extension (um).
        mesh_accuracy: 2 (1: coarse, 2: fine, 3: superfine).
        zmargin: for the FDTD region (um).
        ymargin: for the FDTD region (um).
        xmargin: for the FDTD region (um).
        wavelength_start: 1.2 (um).
        wavelength_stop: 1.6 (um).
        wavelength_points: 500.
        simulation_time: (s) related to max path length
            3e8/2.4*10e-12*1e6 = 1.25mm.
        simulation_temperature: in kelvin (default = 300).
        frequency_dependent_profile: compute mode profiles for each wavelength.
        field_profile_samples: number of wavelengths to compute field profile.
    """

    port_margin: float = 0.2
    port_extension: float = 5.0
    mesh_accuracy: int = 1
    wavelength_start: float = 1.5
    wavelength_stop: float = 1.6
    wavelength_points: int = 200
    simulation_time: float = 10e-12
    simulation_temperature: float = 300
    frequency_dependent_profile: bool = True
    field_profile_samples: int = 15
    distance_monitors_to_pml: float = 0.5
    material_name_to_lumerical: dict[str, str] = material_name_to_lumerical_default

    class Config:
        """pydantic basemodel config."""

        arbitrary_types_allowed = True


SIMULATION_SETTINGS_LUMERICAL_FDTD = SimulationSettingsLumericalFdtd()
