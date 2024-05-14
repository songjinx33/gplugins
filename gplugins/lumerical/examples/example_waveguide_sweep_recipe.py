from pathlib import Path
from gplugins.lumerical.recipes.waveguide_recipe import (
    WaveguideSweepRecipe,
    WaveguideSweepDesignIntent,
    grillot_strip_waveguide_loss_model,
)
from gplugins.lumerical.simulation_settings import SimulationSettingsLumericalMode
from gplugins.lumerical.convergence_settings import ConvergenceSettingsLumericalMode
from gdsfactory.config import logger
from gdsfactory.components.straight import straight
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


### 0. DEFINE WHERE FILES ARE SAVED
dirpath = Path("../recipes/recipe_runs")
dirpath.mkdir(parents=True, exist_ok=True)

### 1. DEFINE DESIGN INTENT
design_intent = WaveguideSweepDesignIntent(
    param_name="width",
    param_vals=list(np.linspace(0.5, 3.0, 26))
)

### 2. DEFINE LAYER STACK
from gdsfactory.technology.layer_stack import LayerLevel, LayerStack

layerstack_lumerical = LayerStack(
    layers={
        "box": LayerLevel(
            layer=(99999, 0),
            thickness=3.0,
            zmin=-3.0,
            material="sio2",
            sidewall_angle=0.0,
            mesh_order=9,
            layer_type="background",
        ),
        "core": LayerLevel(
            layer=(1, 0),
            thickness=0.22,
            zmin=0.0,
            material="si",
            sidewall_angle=2.0,
            width_to_z=0.5,
            mesh_order=2,
            layer_type="grow",
            info={"active": True},
        ),
    }
)

### 3. DEFINE SIMULATION AND CONVERGENCE SETTINGS
convergence_setup = ConvergenceSettingsLumericalMode(neff_diff=5e-3,
                                                     ng_diff=5e-3)
simulation_setup = SimulationSettingsLumericalMode(injection_axis="2D X normal",
                                                   mesh_cells_per_wavl=80,
                                                   num_modes=10,
                                                   target_mode=1,
                                                   )


wg_recipe = WaveguideSweepRecipe(cell=straight,
                                    design_intent=design_intent,
                                    simulation_setup=simulation_setup,
                                    convergence_setup=convergence_setup,
                                    layer_stack=layerstack_lumerical,
                                    dirpath=dirpath)
wg_recipe.override_recipe = False
success = wg_recipe.eval()
if success:
    logger.info("Completed waveguide recipe.")
else:
    logger.info("Incomplete run of waveguide recipe.")

# Get waveguide loss model
neff_vs_width = pd.DataFrame({"neff": [
    1.4354,
    1.4369,
    1.4388,
    1.4410,
    1.4440,
    1.4472,
    1.4513,
    1.4562,
    1.4626,
    1.4690,
    1.4768,
    1.4855,
    1.4960,
    1.5060,
    1.5225,
    1.5570,
    1.6013,
    1.6481,
    1.7039,
    1.7620,
    1.8255,
    1.8823,
    1.9415,
    1.9965,
    2.0513,
    2.0983,
    2.1428,
    2.1843,
    2.2248,
    2.2587,
    2.2911,
    2.3213,
    2.3507,
    2.3751,
    2.3989,
    2.4211,
    2.4428,
],
"width": [
    0.0500,
    0.0625,
    0.0750,
    0.0875,
    0.1000,
    0.1125,
    0.1250,
    0.1375,
    0.1500,
    0.1625,
    0.1750,
    0.1875,
    0.2000,
    0.2125,
    0.2250,
    0.2375,
    0.2500,
    0.2625,
    0.2750,
    0.2875,
    0.3000,
    0.3125,
    0.3250,
    0.3375,
    0.3500,
    0.3625,
    0.3750,
    0.3875,
    0.4000,
    0.4125,
    0.4250,
    0.4375,
    0.4500,
    0.4625,
    0.4750,
    0.4875,
    0.5000,
]})

loss_model = grillot_strip_waveguide_loss_model(neff_vs_width=wg_recipe.recipe_results.waveguide_sweep_characteristics,
                                                wavelength=simulation_setup.wavl,
                                                )
loss_model = grillot_strip_waveguide_loss_model(neff_vs_width=neff_vs_width,
                                                wavelength=simulation_setup.wavl)

plt.figure()
plt.plot(loss_model.loc[:, "width"], loss_model.loc[:, "loss_dB_per_cm"], marker="*")
plt.xlabel("Width (um)")
plt.ylabel("Loss (dB/cm)")
plt.title("Waveguide Propagation Loss")
plt.grid("on")
plt.show()