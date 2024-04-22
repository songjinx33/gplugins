from gdsfactory.components.coupler_ring import coupler_ring
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.straight import straight
import gdsfactory as gf
from functools import partial
from gdsfactory.cross_section import (
    strip,
    Section,
    pn,
    rib,
)
from gdsfactory.typings import CrossSectionFactory
from gdsfactory.component import Component
from pathlib import Path
from gdsfactory.technology.layer_stack import LayerLevel, LayerStack
from gdsfactory.generic_tech.layer_map import LAYER

from gplugins.lumerical.recipes.microring_modulator_recipe import (
    PNJunctionDesignIntent,
    PNMicroringModulatorRecipe,
)
from gplugins.lumerical.simulation_settings import (
    SimulationSettingsLumericalCharge,
    SimulationSettingsLumericalMode,
    SIMULATION_SETTINGS_LUMERICAL_FDTD,
    LUMERICAL_INTERCONNECT_SIMULATION_SETTINGS,
)
from gplugins.lumerical.convergence_settings import (
    ConvergenceSettingsLumericalCharge,
    ConvergenceSettingsLumericalMode,
    LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
)

@gf.cell
def ring_double_pn_2seg(
    add_gap: float = 0.6,
    thru_gap: float = 0.6,
    radius: float = 15.0,
    length_coupling: float = 0.0,
    length_pn: float = 25.0,
    waveguide_cross_section: CrossSectionFactory = rib,
    pn_cross_section: CrossSectionFactory = pn,
) -> Component:
    """Double bus ring modulator with 2 segments of PN junction phaseshifters.

    NOTE: N++, N+ overlaps with N doping areas, and P++, P+ overlaps with P
    doping areas.


                    Top View of Double Bus Ring Modulator
              DROP  ────────────────────────────────────  ADD
              PORT  ────────────────────────────────────  PORT
                                                 ▲
                              length_coupling    │ add_gap
                                  <------>       ▼
                                  ────────       ▲
                                /   ────   \     │
                               /  /      \  \    │  radius
                              /  /        \  \   │
                             /  /          \  \  ▼
                            │  │            │  │ ▲
                            │  │            │  │ │
                            │  │            │  │ │ length_pn
                            │  │            │  │ │
                            │  │            │  │ ▼
                             \  \          /  /  ▲
                              \  \        /  /   │
                               \  \      /  /    │ radius
                                \   ────   /     │
                                  ────────       ▼
                                  <------>       ▲
                              length_coupling    │ thru_gap
                                                 ▼
              IN    ────────────────────────────────────  THRU
             PORT   ────────────────────────────────────  PORT

    Args:
        add_gap: gap to add waveguide.
        thru_gap: gap to drop waveguide.
        radius: bend radius for coupler
        length_coupling: length of coupling region
        length_pn: length of PN junction phaseshifters
        waveguide_cross_section: waveguide cross section
        pn_cross_section: PN junction cross section

    """
    c = gf.Component()

    # Create couplers
    thru_coupler = coupler_ring(gap=thru_gap,
                            radius=radius,
                            length_x=length_coupling,
                            bend=bend_circular,
                            length_extension=0,
                            cross_section=waveguide_cross_section,
                            )
    add_coupler = coupler_ring(gap=add_gap,
                                radius=radius,
                                length_x=length_coupling,
                                bend=bend_circular,
                                length_extension=0,
                                cross_section=waveguide_cross_section,
                                )

    ref_thru_coupler = c.add_ref(thru_coupler)
    ref_add_coupler = c.add_ref(add_coupler)

    # Shift device such that 0,0 is at port o1 (bottom left)
    ref_thru_coupler.movex(radius)
    ref_add_coupler.movex(-radius)

    # Create PN phase shifters
    pn_junction = partial(straight,
                          cross_section=pn_cross_section,
                          length=length_pn)

    phase_shifter1 = pn_junction()
    ps1 = phase_shifter1.rotate(90)
    ref_ps1 = c.add_ref(ps1)
    ref_ps1.x = ref_thru_coupler.ports["o2"].center[0]
    ref_ps1.ymin = ref_thru_coupler.ports["o2"].center[1]

    ref_ps2 = c.add_ref(ps1)
    ref_ps2.rotate(180)
    ref_ps2.x = ref_thru_coupler.ports["o3"].center[0]
    ref_ps2.ymin = ref_thru_coupler.ports["o3"].center[1]

    # Place add coupler above the PN phase shifters
    ref_add_coupler.rotate(180)
    ref_add_coupler.ymin = ref_ps1.ymax

    # Add ports to component
    c.add_port("o1", port=ref_thru_coupler["o1"])
    c.add_port("o2", port=ref_thru_coupler["o4"])
    c.add_port("o3", port=ref_add_coupler["o1"])
    c.add_port("o4", port=ref_add_coupler["o4"])
    c.auto_rename_ports()

    return c


### 0. DEFINE WHERE FILES ARE SAVED
dirpath = Path("../recipes/recipe_runs")
dirpath.mkdir(parents=True, exist_ok=True)

### 1. DEFINE DESIGN INTENT
design_intent = PNJunctionDesignIntent()

"""
                            PN Phaseshifter Cross Section
                                   width_ridge
                         |<------------------------->|
                         |         offset_low_doping |
                         |             <------->     |
                         |             |       |     |
                         |            wg    junction |
 width_contact           |           center  center  |             width_contact
|<---------->|           |             |       |     |            |<---------->|
┌────────────┐           ┌─────────────|───────|─────┐            ┌────────────┐
│    |       │           │             |       |     │            │      |     │
│    |       └───────────┘             |       |     └────────────┘      |     │
│    | P++  |  P+  |         P         |       |     N     |  N+  |  N++ |     │
└────|──────|──────|───────────────────|───────────────────|──────|──────|─────┘
|    |      |      |<----------------->|<----------------->|      |      |     |
|    |      |      | gap_medium_doping | gap_medium_doping |      |      |     |
|    |      |      |                   |                   |      |      |     |
|    |      |<------------------------>|<------------------------>|      |     |
|    |      |     gap_high_doping      |      gap_high_doping     |      |     |
|    |<----------------------------------------------------------------->|     |
|                                  width_doping                                |
|<---------------------------------------------------------------------------->|
                                   width_slab
                            
                                   
                            Waveguide Cross Section
                                   width_ridge
                         |<------------------------->|
                         ┌───────────────────────────┐       
                         │                           │            
┌────────────────────────┘                           └─────────────────────────┐ 
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
|<---------------------------------------------------------------------------->|
                                   width_slab
"""
# Waveguide cross section parameters
waveguide_layer = (1,0)
width_ridge = 0.45
width_slab=5.45
width_contact = 1.0

# Ring parameters
radius = 15
ring_gap = 0.6

# PN Phaseshifter parameters
metal_layer = (40, 0)
width_doping = 8.0
offset_low_doping=-0.065
gap_medium_doping=0.595
gap_high_doping=0.595 + 0.055

rib = partial(
    strip,
    width=width_ridge,
    sections=(Section(width=width_slab, layer="SLAB90", name="slab90"),),
    radius=radius,
)
pn_contacts = partial(pn, width=width_ridge,
                          offset_low_doping=offset_low_doping,
                          gap_medium_doping=gap_medium_doping,
                          gap_high_doping=gap_high_doping,
                          width_doping=width_doping,
                          width_slab=width_slab,
                          sections=(Section(width=width_contact,
                                            offset=(width_slab - width_contact) / 2,
                                            layer=waveguide_layer),
                                    Section(width=width_contact,
                                            offset=-(width_slab - width_contact) / 2,
                                            layer=waveguide_layer),
                                    Section(width=width_contact,
                                            offset=(width_slab - width_contact) / 2,
                                            layer=metal_layer),
                                    Section(width=width_contact,
                                            offset=-(width_slab - width_contact) / 2,
                                            layer=metal_layer),
                                    ))
c = ring_double_pn_2seg(pn_cross_section=pn_contacts,
                        waveguide_cross_section=rib)
pn_junction_component = c.named_references["rotate_1"].parent
c.show()

### 2. DEFINE LAYER STACK
core_thickness = 0.155 # um

layerstack_lumerical = LayerStack(
    layers={
        "clad": LayerLevel(
            layer=(99999, 0),
            thickness=3.0,
            zmin=0.0,
            material="sio2",
            sidewall_angle=0.0,
            mesh_order=9,
            layer_type="background",
        ),
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
            thickness=core_thickness,
            zmin=0.0,
            material="si",
            sidewall_angle=2.0,
            width_to_z=0.5,
            mesh_order=2,
            layer_type="grow",
            info={"active": True},
        ),
        "slab90": LayerLevel(
            layer=(3, 0),
            thickness=0.05,
            zmin=0.0,
            material="si",
            sidewall_angle=2.0,
            width_to_z=0.5,
            mesh_order=2,
            layer_type="grow",
            info={"active": True},
        ),
        "N": LayerLevel(
            layer=LAYER.N,
            thickness=core_thickness/2,
            zmin=core_thickness,
            material="si",
            mesh_order=4,
            background_doping_concentration=5e17,
            background_doping_ion="n",
            orientation="100",
            layer_type="doping",
        ),
        "NP": LayerLevel(
            layer=LAYER.NP,
            thickness=core_thickness/2,
            zmin=core_thickness,
            material="si",
            mesh_order=4,
            background_doping_concentration=3e18,
            background_doping_ion="n",
            orientation="100",
            layer_type="doping",
        ),
        "NPP": LayerLevel(
            layer=LAYER.NPP,
            thickness=core_thickness/2,
            zmin=core_thickness,
            material="si",
            mesh_order=4,
            background_doping_concentration=1e19,
            background_doping_ion="n",
            orientation="100",
            layer_type="doping",
        ),
        "P": LayerLevel(
            layer=LAYER.P,
            thickness=core_thickness/2,
            zmin=core_thickness,
            material="si",
            mesh_order=4,
            background_doping_concentration=7e17,
            background_doping_ion="p",
            orientation="100",
            layer_type="doping",
        ),
        "PP": LayerLevel(
            layer=LAYER.PP,
            thickness=core_thickness/2,
            zmin=core_thickness,
            material="si",
            mesh_order=4,
            background_doping_concentration=2e18,
            background_doping_ion="p",
            orientation="100",
            layer_type="doping",
        ),
        "PPP": LayerLevel(
            layer=LAYER.PPP,
            thickness=core_thickness/2,
            zmin=core_thickness,
            material="si",
            mesh_order=4,
            background_doping_concentration=1e19,
            background_doping_ion="p",
            orientation="100",
            layer_type="doping",
        ),
        "via": LayerLevel(
            layer=LAYER.VIAC,
            thickness=1.0,
            zmin=core_thickness,
            material="Aluminum",
            mesh_order=4,
            orientation="100",
            layer_type="grow",
        ),
    }
)

# 3. DEFINE SIMULATION AND CONVERGENCE SETTINGS
charge_settings = SimulationSettingsLumericalCharge(x=pn_junction_component.x,
                                                    y=pn_junction_component.y,
                                                    dimension="2D Y-Normal",
                                                    xspan=width_slab + 1.0)
charge_convergence_settings = ConvergenceSettingsLumericalCharge(global_iteration_limit=500,
                                                                 gradient_mixing="fast")
mode_settings = SimulationSettingsLumericalMode(injection_axis="2D Y normal",
                                                wavl_pts=11,
                                                x=charge_settings.x,
                                                y=charge_settings.y,
                                                z=charge_settings.z,
                                                xspan=charge_settings.xspan,
                                                yspan=charge_settings.yspan,
                                                zspan=charge_settings.zspan,
                                                target_mode=3
                                                )
mode_convergence_settings = ConvergenceSettingsLumericalMode()

# 4. CREATE AND RUN DESIGN RECIPE
mrm_recipe = PNMicroringModulatorRecipe(component=c,
                 layer_stack=layerstack_lumerical,
                 pn_design_intent=design_intent,
                 mode_simulation_setup=mode_settings,
                 mode_convergence_setup=mode_convergence_settings,
                 charge_simulation_setup=charge_settings,
                 charge_convergence_setup=charge_convergence_settings,
                 fdtd_simulation_setup=SIMULATION_SETTINGS_LUMERICAL_FDTD,
                 fdtd_convergence_setup=LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
                 interconnect_simulation_setup=LUMERICAL_INTERCONNECT_SIMULATION_SETTINGS,
                 dirpath=dirpath,
                )
mrm_recipe.override_recipe = False
mrm_recipe.eval()