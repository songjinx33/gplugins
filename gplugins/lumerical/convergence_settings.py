from pydantic import BaseModel


class ConvergenceSettingsLumericalEme(BaseModel):
    passes: int = 5
    sparam_diff: float = 0.01

    class Config:
        arbitrary_types_allowed = True


LUMERICAL_EME_CONVERGENCE_SETTINGS = ConvergenceSettingsLumericalEme()


class ConvergenceSettingsLumericalFdtd(BaseModel):
    sparam_diff: float = 0.01
    port_field_intensity_threshold: float = 1e-5

    class Config:
        arbitrary_types_allowed = True

LUMERICAL_FDTD_CONVERGENCE_SETTINGS = ConvergenceSettingsLumericalFdtd()