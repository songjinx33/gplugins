import gdsfactory as gf
from gdsfactory.generic_tech import LAYER

from gplugins.lumerical.recipes.fdtd_recipe import FdtdRecipe


def test_fdtd_recipe():
    c = gf.Component("two_references")
    wr1 = c << gf.components.straight(width=0.6, length=3.0, layer=LAYER.WG)
    c.add_ports(wr1.get_ports_list(), prefix="bot_")

    recipe = FdtdRecipe(c)
    recipe.eval()

    # Change the component and check whether recipe is still fresh
    comp = gf.components.taper()
    recipe.cell = comp
    if recipe.is_fresh():
        raise AssertionError(
            f"Expected: False | Got: {recipe.is_fresh()}. Recipe should not be fresh after component is changed."
            + "The recipe should be re-eval'ed before it is fresh again."
        )
