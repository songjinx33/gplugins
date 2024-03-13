import gdsfactory as gf

from gplugins.lumerical.config import DEBUG_LUMERICAL
from gplugins.lumerical.convergence_settings import LUMERICAL_FDTD_CONVERGENCE_SETTINGS
from gplugins.lumerical.fdtd import LumericalFdtdSimulation
from gplugins.lumerical.simulation_settings import SIMULATION_SETTINGS_LUMERICAL_FDTD


def test_lumerical_fdtd_simulation():
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
        width=1.0,
    )

    taper = taper_cross_section(
        cross_section1=xs_wg, cross_section2=xs_wg_wide, length=2.0
    )

    SIMULATION_SETTINGS_LUMERICAL_FDTD.mesh_accuracy = 1
    LUMERICAL_FDTD_CONVERGENCE_SETTINGS.sparam_diff = 0.1
    LUMERICAL_FDTD_CONVERGENCE_SETTINGS.port_field_intensity_threshold = 1e-5

    sim = LumericalFdtdSimulation(
        taper,
        simulation_settings=SIMULATION_SETTINGS_LUMERICAL_FDTD,
        convergence_settings=LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
        run_port_convergence=False,
        run_mesh_convergence=True,
        run_field_intensity_convergence=False,
        # hide=not DEBUG_LUMERICAL,
        hide=False
    )

    sim.write_sparameters(overwrite=True)
