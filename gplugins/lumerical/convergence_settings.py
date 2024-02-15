from pydantic import BaseModel


class ConvergenceSettingsLumericalEme(BaseModel):
    passes: int = 10
    sparam_diff: float = 0.01

    class Config:
        arbitrary_types_allowed = True


LUMERICAL_EME_CONVERGENCE_SETTINGS = ConvergenceSettingsLumericalEme()
