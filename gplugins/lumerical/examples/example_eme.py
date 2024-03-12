import gdsfactory as gf
from gdsfactory.config import logger

from gplugins.lumerical.eme import LumericalEmeSimulation


def example_eme():
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

    sim = LumericalEmeSimulation(
        taper,
        run_mesh_convergence=False,
        run_cell_convergence=False,
        run_mode_convergence=False,
        hide=False,
    )

    data = sim.get_length_sweep()
    logger.info(f"{data}\nDone")


if __name__ == "__main__":
    example_eme()
