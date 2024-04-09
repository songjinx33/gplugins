
from gdsfactory.components.coupler_ring import coupler_ring
from gdsfactory.components.bend_circular import bend_circular
from pathlib import Path
import gdsfactory as gf
from functools import partial
import numpy as np
import pandas as pd
from gdsfactory.cross_section import strip
from gdsfactory.cross_section import Section
from gdsfactory.cross_section import pn, rib
from gdsfactory.components.straight import straight
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionFactory
from gdsfactory.config import logger
from pydantic import BaseModel
from typing import Literal
from gplugins.design_recipe.DesignRecipe import DesignRecipe, eval_decorator
from gdsfactory.pdk import LayerStack, get_layer_stack

@gf.cell
def ring_double_pn_2seg(
    add_gap: float = 0.6,
    drop_gap: float = 0.6,
    radius: float = 15.0,
    length_coupling: float = 0.0,
    length_pn: float = 25.0,
    waveguide_cross_section: CrossSectionFactory = rib,
    pn_cross_section: CrossSectionFactory = pn,
) -> Component:
    """Double bus ring modulator with 2 segments of PN junction phaseshifters.

    Args:
        width: width of the ridge in um.
        add_gap: gap to add waveguide.
        drop_gap: gap to drop waveguide.
        radius: bend radius for coupler
        offset_low_doping: from center to junction center.
        gap_low_doping: from waveguide center to low doping. Only used for PIN.
        gap_medium_doping: from waveguide center to medium doping. None removes it.
        gap_high_doping: from center to high doping. None removes it.
        length_coupling: length of coupling region
        length_pn: length of PN junction phaseshifters
        waveguide_cross_section: waveguide cross section
        pn_cross_section: PN junction cross section

    """
    c = gf.Component()

    # Create couplers
    drop_coupler = coupler_ring(gap=drop_gap,
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

    ref_drop_coupler = c.add_ref(drop_coupler)
    ref_add_coupler = c.add_ref(add_coupler)

    # Shift device such that 0,0 is at port o1 (bottom left)
    ref_drop_coupler.movex(radius)
    ref_add_coupler.movex(-radius)

    # Create PN phase shifters
    pn_junction = partial(straight,
                          cross_section=pn_cross_section,
                          length=length_pn)

    phase_shifter1 = pn_junction()
    ps1 = phase_shifter1.rotate(90)
    ref_ps1 = c.add_ref(ps1)
    ref_ps1.x = ref_drop_coupler.ports["o2"].center[0]
    ref_ps1.ymin = ref_drop_coupler.ports["o2"].center[1]

    ref_ps2 = c.add_ref(ps1)
    ref_ps2.rotate(180)
    ref_ps2.x = ref_drop_coupler.ports["o3"].center[0]
    ref_ps2.ymin = ref_drop_coupler.ports["o3"].center[1]

    # Place add coupler above the PN phase shifters
    ref_add_coupler.rotate(180)
    ref_add_coupler.ymin = ref_ps1.ymax

    # Add ports to component
    c.add_port("o1", port=ref_drop_coupler["o1"])
    c.add_port("o2", port=ref_drop_coupler["o4"])
    c.add_port("o3", port=ref_add_coupler["o1"])
    c.add_port("o4", port=ref_add_coupler["o4"])
    c.auto_rename_ports()

    return c


### 0. DEFINE WHERE FILES ARE SAVED
dirpath = Path("../recipes/recipe_runs")
dirpath.mkdir(parents=True, exist_ok=True)

### 1. DEFINE DESIGN INTENT
width_ridge = 0.45
width_slab=5.45
width_contact = 1.0

waveguide_layer = (1,0)
metal_layer = (40, 0)

radius = 15
ring_gap = 0.6

offset_low_doping=-0.065
gap_medium_doping=0.595
gap_high_doping=0.595 + 0.055

nm = 1e-9
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

### 2. DEFINE LAYER STACK
from gdsfactory.technology.layer_stack import LayerLevel, LayerStack
from gdsfactory.generic_tech.layer_map import LAYER

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
            thickness=0.155,
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
            thickness=0.155/2,
            zmin=0.155,
            material="si",
            mesh_order=4,
            background_doping_concentration=5e17,
            background_doping_ion="n",
            orientation="100",
            layer_type="doping",
        ),
        "NP": LayerLevel(
            layer=LAYER.NP,
            thickness=0.155/2,
            zmin=0.155,
            material="si",
            mesh_order=4,
            background_doping_concentration=3e18,
            background_doping_ion="n",
            orientation="100",
            layer_type="doping",
        ),
        "NPP": LayerLevel(
            layer=LAYER.NPP,
            thickness=0.155/2,
            zmin=0.155,
            material="si",
            mesh_order=4,
            background_doping_concentration=1e19,
            background_doping_ion="n",
            orientation="100",
            layer_type="doping",
        ),
        "P": LayerLevel(
            layer=LAYER.P,
            thickness=0.155/2,
            zmin=0.155,
            material="si",
            mesh_order=4,
            background_doping_concentration=7e17,
            background_doping_ion="p",
            orientation="100",
            layer_type="doping",
        ),
        "PP": LayerLevel(
            layer=LAYER.PP,
            thickness=0.155/2,
            zmin=0.155,
            material="si",
            mesh_order=4,
            background_doping_concentration=2e18,
            background_doping_ion="p",
            orientation="100",
            layer_type="doping",
        ),
        "PPP": LayerLevel(
            layer=LAYER.PPP,
            thickness=0.155/2,
            zmin=0.155,
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
            zmin=0.155,
            material="Aluminum",
            mesh_order=4,
            orientation="100",
            layer_type="grow",
        ),
    }
)
c = ring_double_pn_2seg(pn_cross_section=pn_contacts,
                        waveguide_cross_section=rib)

coupler_component = c.named_references["coupler_ring_1"].parent
pn_junction_component = c.named_references["rotate_1"].parent

c.show()

from gplugins.lumerical.device import LumericalChargeSimulation
from gplugins.lumerical.simulation_settings import (
    SimulationSettingsLumericalCharge,
    LUMERICAL_CHARGE_SIMULATION_SETTINGS,
    SimulationSettingsLumericalMode,
    LUMERICAL_MODE_SIMULATION_SETTINGS,
)
from gplugins.lumerical.convergence_settings import (
    ConvergenceSettingsLumericalCharge,
    LUMERICAL_CHARGE_CONVERGENCE_SETTINGS,
    ConvergenceSettingsLumericalMode,
    LUMERICAL_MODE_CONVERGENCE_SETTINGS,
)


class PNJunctionDesignIntent(BaseModel):
    r"""
    Design intent for PN junction

    Attributes:
        contact1_name: Name of first contact connected to PN junction
        contact2_name: Name of second contact connected to PN junction
        voltage_start: Start voltage in sweep
        voltage_stop: Stop voltage in sweep
        voltage_pts: Number of voltage points in sweep
    """
    contact1_name: str = "anode"
    contact2_name: str = "cathode"
    voltage_start: float = -1
    voltage_stop: float = 3
    voltage_pts: int = 21

    class Config:
        arbitrary_types_allowed = True





design_intent = PNJunctionDesignIntent()

charge_settings = SimulationSettingsLumericalCharge(x=pn_junction_component.x,
                                                    y=pn_junction_component.y,
                                                    dimension="2D Y-Normal",
                                                    xspan=width_slab + 1.0)
charge_convergence_settings = ConvergenceSettingsLumericalCharge(global_iteration_limit=500,
                                                                 gradient_mixing="fast")

class PNJunctionChargeRecipe(DesignRecipe):
    """
    PN junction CHARGE recipe.

    Attributes:
        recipe_setup:
            simulation_setup: CHARGE simulation setup
            convergence_setup: CHARGE convergence setup
            design_intent: PN junction design intent
        recipe_results:

    """
    def __init__(
        self,
        component: Component | None = None,
        layer_stack: LayerStack | None = None,
        design_intent: PNJunctionDesignIntent | None = None,
        simulation_setup: SimulationSettingsLumericalCharge
        | None = LUMERICAL_CHARGE_SIMULATION_SETTINGS,
        convergence_setup: ConvergenceSettingsLumericalCharge
        | None = LUMERICAL_CHARGE_CONVERGENCE_SETTINGS,
        dirpath: Path | None = None,
    ):
        layer_stack = layer_stack or get_layer_stack()
        super().__init__(cell=component, layer_stack=layer_stack,
                         dirpath=dirpath)
        # Add information to recipe setup. NOTE: This is used for hashing
        self.recipe_setup.simulation_setup = simulation_setup
        self.recipe_setup.convergence_setup = convergence_setup
        self.recipe_setup.design_intent = design_intent

    @eval_decorator
    def eval(self, run_convergence: bool = True):
        r"""
        Run PN junction recipe

        1. Sweep PN junction voltage and extract charge profile

        Parameters:
            run_convergence: Run convergence if True
        """
        # Set up simulation to extract charge profile vs. voltage
        boundary_settings = {
            "b0": {
                "name": self.recipe_setup.design_intent.contact1_name,
                "bc mode": "steady state",
                "sweep type": "single",
                "force ohmic": True,
                "voltage": 0,
            },
            "b1": {
                "name": self.recipe_setup.design_intent.contact2_name,
                "bc mode": "steady state",
                "sweep type": "range",
                "force ohmic": True,
                "range start": self.recipe_setup.design_intent.voltage_start,
                "range stop": self.recipe_setup.design_intent.voltage_stop,
                "range num points": self.recipe_setup.design_intent.voltage_pts,
                "range backtracking": "enabled",
            },
        }
        sim = LumericalChargeSimulation(component=self.cell,
                                        layerstack=self.recipe_setup.layer_stack,
                                        simulation_settings=self.recipe_setup.simulation_setup,
                                        convergence_settings=self.recipe_setup.convergence_setup,
                                        boundary_settings=boundary_settings,
                                        dirpath=self.dirpath,
                                        hide=False,
                                        )
        s = sim.session
        s.run()
        self.recipe_results.charge_profile_path = sim.simulation_dirpath / "charge.mat"

        # Set up simulation to extract RC electrical characteristics
        s.switchtolayout()
        s.setnamed("CHARGE::monitor", "save data", 0)
        s.setnamed("CHARGE", "solver mode", "ssac")
        s.setnamed("CHARGE", "solver type", "newton")


        boundary_settings = {
            self.recipe_setup.design_intent.contact2_name: {
                "apply ac small signal": "all",
            },
        }
        sim.set_boundary_conditions(boundary_settings=boundary_settings)
        s.save()
        s.run()

        # Get electrical results
        results = s.getresult("CHARGE", f"ac_{self.recipe_setup.design_intent.contact2_name}")
        vac = s.getnamed("CHARGE", "perturbation amplitude")
        iac = results["dI"][0,:,0][:,0]
        vbias = results[f"V_{self.recipe_setup.design_intent.contact2_name}"][:,0]
        f = results["f"][0][0]

        Z = vac / iac
        Y = 1 / Z

        C = np.abs(np.imag(Y) / (2 * np.pi * f)) # F/cm
        R = np.abs(np.real(Z)) # ohm-cm

        BW = 1/(2 * np.pi * R * C)

        self.recipe_results.ac_voltage = vac
        self.recipe_results.frequency = f
        self.recipe_results.electrical_characteristics = pd.DataFrame({"vbias": list(vbias),
                                                                       "impedance": list(Z),
                                                                       "capacitance_F_per_cm": list(C),
                                                                       "resistance_ohm_cm": list(R),
                                                                       "bandwidth_GHz": list(BW)})
        return True

# pn_charge_recipe = PNJunctionChargeRecipe(component=pn_junction_component,
#                              layer_stack=layerstack_lumerical,
#                              design_intent=design_intent,
#                              simulation_setup=charge_settings,
#                              convergence_setup=charge_convergence_settings,
#                              dirpath=dirpath)
# pn_charge_recipe.eval()


from gplugins.lumerical.mode import LumericalModeSimulation
from gplugins.lumerical.simulation_settings import SimulationSettingsLumericalMode
from gplugins.lumerical.convergence_settings import ConvergenceSettingsLumericalMode
from gplugins.lumerical.config import um

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

# Create MODE simulation



class PNJunctionRecipe(DesignRecipe):
    """
    PN junction MODE recipe.

    Attributes:
        recipe_setup:
            simulation_setup: MODE simulation setup
            convergence_setup: MODE convergence setup
            design_intent: PN junction design intent
        recipe_results:

    """
    def __init__(
        self,
        component: Component | None = None,
        layer_stack: LayerStack | None = None,
        design_intent: PNJunctionDesignIntent | None = None,
        mode_simulation_setup: SimulationSettingsLumericalMode | None = LUMERICAL_MODE_SIMULATION_SETTINGS,
        mode_convergence_setup: ConvergenceSettingsLumericalMode | None = LUMERICAL_MODE_CONVERGENCE_SETTINGS,
        charge_simulation_setup: SimulationSettingsLumericalCharge | None = LUMERICAL_CHARGE_SIMULATION_SETTINGS,
        charge_convergence_setup: ConvergenceSettingsLumericalCharge | None = LUMERICAL_CHARGE_CONVERGENCE_SETTINGS,
        dependencies: list[DesignRecipe] | None = None,
        dirpath: Path | None = None,
    ):
        layer_stack = layer_stack or get_layer_stack()
        dependencies = [PNJunctionChargeRecipe(component=component,
                                               layer_stack=layer_stack,
                                               design_intent=design_intent,
                                               simulation_setup=charge_simulation_setup,
                                               convergence_setup=charge_convergence_setup,
                                               dirpath=dirpath)] or dependencies
        super().__init__(cell=component, layer_stack=layer_stack,
                         dirpath=dirpath, dependencies=dependencies)
        # Add information to recipe setup. NOTE: This is used for hashing
        self.recipe_setup.mode_simulation_setup = mode_simulation_setup
        self.recipe_setup.mode_convergence_setup = mode_convergence_setup
        self.recipe_setup.charge_simulation_setup = charge_simulation_setup
        self.recipe_setup.charge_convergence_setup = charge_convergence_setup
        self.recipe_setup.design_intent = design_intent

    @eval_decorator
    def eval(self, run_convergence: bool = True):
        r"""
        Run PN junction recipe

        1. Sweep PN junction voltage and extract charge profile

        Parameters:
            run_convergence: Run convergence if True
        """
        # Issue: Metals affect MODE's ability to effectively calculate waveguide modes.
        # Solution: Use a different layerstack where metal layer(s) are removed
        layerstack_lumerical_mode = self.recipe_setup.layer_stack.model_copy()
        layer_name = self.recipe_setup.layer_stack.get_layer_to_layername()[
            self.recipe_setup.charge_simulation_setup.metal_layer][0]
        layerstack_lumerical_mode.layers.pop(layer_name)

        mode_sim = LumericalModeSimulation(component=self.cell,
                                           layerstack=layerstack_lumerical_mode,
                                           simulation_settings=self.recipe_setup.mode_simulation_setup,
                                           convergence_settings=self.recipe_setup.mode_convergence_setup,
                                           dirpath=self.dirpath,
                                           hide=False)

        pn_recipe = self.dependencies.constituent_recipes[0]
        self.recipe_results.charge_profile_path = pn_recipe.recipe_results.charge_profile_path
        self.recipe_results.ac_voltage = pn_recipe.recipe_results.ac_voltage
        self.recipe_results.frequency = pn_recipe.recipe_results.frequency
        self.recipe_results.electrical_characteristics = pn_recipe.recipe_results.electrical_characteristics

        # Set up MODE sim to import charge profile
        s = mode_sim.session
        s.addgridattribute("np density")
        s.set("x", 0)
        s.set("y", 0)
        s.set("z", 0)
        s.importdataset(str(pn_recipe.recipe_results.charge_profile_path.resolve()))

        # Get original material
        s.select("layer group")
        layer_name = layerstack_lumerical_mode.get_layer_to_layername()[self.recipe_setup.charge_simulation_setup.dopant_layer][0]
        base_material = s.getlayer(layer_name, "pattern material")

        # Create material with relation between free carriers and index
        new_material_name = f"{layer_name}_doped"
        material_props = {"Name": new_material_name,
                          "np density model": "Soref and Bennett",
                          "Coefficients": "Nedeljkovic, Soref & Mashanovich, 2011",
                          "Base Material": base_material}

        mname = s.addmaterial("Index perturbation")
        s.setmaterial(mname, material_props)

        # Set layer material with free carriers
        s.select("layer group")
        s.setlayer(layer_name, "pattern material", new_material_name)

        # If convergence setup has changed or convergence results are not available
        if (not mode_sim.convergence_is_fresh() or not mode_sim.convergence_results.available()) \
                and run_convergence:
            # Run convergence sweeps to ensure simulation accuracy
            mode_sim.convergence_results.mesh_convergence_data = mode_sim.update_mesh_convergence(verbose=True,
                                             plot=True)
            # Save convergence results as to skip convergence sweeps in future runs
            mode_sim.save_convergence_results()
            logger.info("Saved convergence results.")

        # Add voltage sweep
        s.addsweep()
        s.setsweep("sweep", "number of points", self.recipe_setup.design_intent.voltage_pts)
        para = {"Name": "voltage",
                "Parameter": f"::model::np density::V_{self.recipe_setup.design_intent.contact2_name}_index",
                "Type": "Number",
                "Start": 1,
                "Stop": self.recipe_setup.design_intent.voltage_pts,
                }
        s.addsweepparameter("sweep", para)
        res1 = {"Name": "neff",
               "Result": f"::model::FDE::data::mode{self.recipe_setup.mode_simulation_setup.target_mode}::neff"}
        res2 = {"Name": "swn",
                "Result": "::model::FDE::data::frequencysweep::neff sweep"}
        s.addsweepresult("sweep", res1)
        s.addsweepresult("sweep", res2)
        s.save()

        # Run MODE sim
        s.mesh()
        s.findmodes()

        s.selectmode(self.recipe_setup.mode_simulation_setup.target_mode);
        s.setanalysis('track selected mode', 1);
        s.setanalysis('number of points', self.recipe_setup.mode_simulation_setup.wavl_pts);
        s.setanalysis('number of test modes', self.recipe_setup.mode_simulation_setup.num_modes);
        s.setanalysis('detailed dispersion calculation', 1);
        s.frequencysweep()

        # Save base waveguide profile
        self.recipe_results.waveguide_profile_path = mode_sim.simulation_dirpath / "waveguide.ldf"
        s.savedcard(str(self.recipe_results.waveguide_profile_path.resolve()),
                  "::model::FDE::data::frequencysweep");

        # Run voltage sweep
        s.runsweep("sweep")

        # Extract voltage vs. neff and loss results
        V = np.linspace(self.recipe_setup.design_intent.voltage_start,
                        self.recipe_setup.design_intent.voltage_stop,
                        self.recipe_setup.design_intent.voltage_pts)
        neff = s.getsweepdata("sweep", "neff")

        self.recipe_results.neff_vs_voltage = pd.DataFrame({"voltage": list(V),
                                                            "neff_r": list(np.real(neff[:,0])),
                                                            "neff_i": list(np.imag(neff[:,0]))})

        dneff_per_cm = np.real(neff[:,0] - neff[0]) * \
                2 / (self.recipe_setup.mode_simulation_setup.wavl * um) * 1e-2
        loss_dB_per_cm = -.20 * np.log10(np.exp(-2 * np.pi * np.imag(neff[:,0]) /
                                                (self.recipe_setup.mode_simulation_setup.wavl * um)))
        self.recipe_results.phase_loss_vs_voltage = pd.DataFrame({"voltage": list(V),
                                                            "phase_per_cm": list(dneff_per_cm),
                                                            "loss_dB_per_cm": list(loss_dB_per_cm)})

        return True

pn_recipe = PNJunctionRecipe(component=pn_junction_component,
                 layer_stack=layerstack_lumerical,
                 design_intent=design_intent,
                 mode_simulation_setup=mode_settings,
                 mode_convergence_setup=mode_convergence_settings,
                 charge_simulation_setup=charge_settings,
                 charge_convergence_setup=charge_convergence_settings,
                 dirpath=dirpath,
                 )
pn_recipe.eval()

print("Done")


#
# import lumapi
#
# data = pd.DataFrame([[0, 0, 0],
# [1.11111, 3.1511e-05, -1.88599e-06],
# [2.22222, 5.77673e-05, -3.47821e-06],
# [3.33333, 8.05975e-05, -4.8639e-06],
# [4.44444, 0.000100747, -6.07648e-06],
# [5.55556, 0.000119181, -7.19313e-06],
# [6.66667, 0.000136481, -8.25009e-06],
# [7.77778, 0.000152736, -9.2406e-06],
# [8.88889, 0.000167833, -1.01482e-05],
# [10, 0.000181674, -1.09838e-05]], columns=["voltage", "neff_r", "neff_i"])
#
# ldf_path = Path("./waveguide.ldf")
#
# s = lumapi.INTERCONNECT(hide=False)
# s.addelement("Optical Modulator Measured")
# s.set("measurement type", "effective index")
# s.set("load from file", False)
# s.set("frequency", s.c()/1550e-9)
# s.set("length", 1e-6)
# s.set("input parameter", "table")
# s.set("measurement", np.array(data))
# print("Done")

