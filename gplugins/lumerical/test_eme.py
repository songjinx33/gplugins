from gplugins.lumerical.eme import LumericalEmeSimulation
import gdsfactory as gf


def test_lumerical_eme_simulation_setup():
    from gdsfactory.components.taper_cross_section import taper_cross_section
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
    )

    sim.plot_length_sweep()
