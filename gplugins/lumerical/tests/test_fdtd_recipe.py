import gdsfactory as gf
from gdsfactory.generic_tech import LAYER

from gplugins.lumerical.recipes.fdtd_recipe import FdtdRecipe


def test_fdtd_recipe():
    c = gf.Component("two_references")
    wr1 = c << gf.components.straight(width=0.6, layer=LAYER.WG)
    wr2 = c << gf.components.straight(width=0.6, layer=LAYER.WG)
    wr2.movey(10)
    c.add_ports(wr1.get_ports_list(), prefix="bot_")
    c.add_ports(wr2.get_ports_list(), prefix="top_")

    recipe = FdtdRecipe(c)
    recipe.eval()
