from pathlib import Path
from gplugins.lumerical.recipes.taper_design_recipe import RoutingTaperDesignRecipe, RoutingTaperDesignIntent
from gplugins.lumerical.recipes.waveguide_recipe import (
    WaveguideSweepRecipe,
    WaveguideSweepDesignIntent,
    grillot_strip_waveguide_loss_model,
)
from functools import partial
import gdsfactory as gf
from gplugins.lumerical.simulation_settings import (
    SimulationSettingsLumericalEme,
    SimulationSettingsLumericalFdtd,
    SimulationSettingsLumericalMode,
)
from gplugins.lumerical.convergence_settings import (
    ConvergenceSettingsLumericalEme,
    ConvergenceSettingsLumericalFdtd,
    ConvergenceSettingsLumericalMode,
)
from gdsfactory.config import logger
from gdsfactory.components.taper_cross_section import taper_cross_section
from gdsfactory.components.straight import straight
import numpy as np

### 0. DEFINE WHERE FILES ARE SAVED
dirpath = Path("../recipes/recipe_runs")
dirpath.mkdir(parents=True, exist_ok=True)

### 1. DEFINE DESIGN INTENT
design_intent = RoutingTaperDesignIntent(
    narrow_waveguide_routing_loss_per_cm=3,  # dB/cm
)

narrow_waveguide_cross_section = partial(
    gf.cross_section.cross_section,
    layer=(1, 0),
    width=0.5,
)
wide_waveguide_cross_section = partial(
    gf.cross_section.cross_section,
    layer=(1, 0),
    width=3.0,
)

# Waveguide loss model
wg_design_intent = WaveguideSweepDesignIntent(
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
eme_convergence_setup = ConvergenceSettingsLumericalEme(
    sparam_diff=1 - 10 ** (-0.005 / 10)
)
eme_simulation_setup = SimulationSettingsLumericalEme()

fdtd_convergence_setup = ConvergenceSettingsLumericalFdtd(
    sparam_diff=0.01
)
fdtd_simulation_setup = SimulationSettingsLumericalFdtd(
    mesh_accuracy=4, port_translation=1.0, port_field_intensity_threshold=1e-5,
)

mode_convergence_setup = ConvergenceSettingsLumericalMode(neff_diff=5e-3,
                                                     ng_diff=5e-3)
mode_simulation_setup = SimulationSettingsLumericalMode(injection_axis="2D X normal",
                                                   mesh_cells_per_wavl=80,
                                                   num_modes=10,
                                                   target_mode=1,
                                                   )

wg_recipe = WaveguideSweepRecipe(cell=straight,
                                    design_intent=wg_design_intent,
                                    simulation_setup=mode_simulation_setup,
                                    convergence_setup=mode_convergence_setup,
                                    layer_stack=layerstack_lumerical,
                                    dirpath=dirpath)
wg_recipe.override_recipe = False
success = wg_recipe.eval()
if success:
    logger.info("Completed waveguide recipe.")
else:
    logger.info("Incomplete run of waveguide recipe.")

# Get waveguide loss model
loss_model = grillot_strip_waveguide_loss_model(neff_vs_width=wg_recipe.recipe_results.waveguide_sweep_characteristics,
                                                wavelength=mode_simulation_setup.wavl,
                                                )


taper_recipe = RoutingTaperDesignRecipe(cell=taper_cross_section,
                                        cross_section1=narrow_waveguide_cross_section,
                                        cross_section2=wide_waveguide_cross_section,
                                        design_intent=design_intent,
                                        waveguide_loss_model=loss_model,
                                        eme_simulation_setup=eme_simulation_setup,
                                        eme_convergence_setup=eme_convergence_setup,
                                        fdtd_simulation_setup=fdtd_simulation_setup,
                                        fdtd_convergence_setup=fdtd_convergence_setup,
                                        layer_stack=layerstack_lumerical,
                                        dirpath=dirpath)
taper_recipe.override_recipe = True
success = taper_recipe.eval()
if success:
    logger.info("Completed taper design recipe.")
else:
    logger.info("Incomplete run of taper design recipe.")