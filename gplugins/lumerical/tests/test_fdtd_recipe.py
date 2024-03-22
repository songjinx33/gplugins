import gdsfactory as gf
from gdsfactory.generic_tech import LAYER

from gplugins.lumerical.recipes.fdtd_recipe import FdtdRecipe
from pathlib import Path


def test_fdtd_recipe():
    c = gf.components.straight(width=0.6, length=1.0, layer=LAYER.WG)

    recipe = FdtdRecipe(c, dirpath=Path(__file__).resolve().parent / "test_runs")
    recipe.eval()
