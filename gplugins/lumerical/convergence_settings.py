from pydantic import BaseModel


class ConvergenceSettingsLumericalEme(BaseModel):
    """
    Lumerical EME convergence settings

    Parameters:
        passes: Number of passes / simulations sweeping a convergence parameter before checking for convergence
        sparam_diff: Maximum difference in sparams after x passes sweeping a convergence parameter. Used to check for convergence.
    """

    passes: int = 5
    sparam_diff: float = 0.01

    class Config:
        arbitrary_types_allowed = True


LUMERICAL_EME_CONVERGENCE_SETTINGS = ConvergenceSettingsLumericalEme()


class ConvergenceSettingsLumericalFdtd(BaseModel):
    """
    Lumerical FDTD convergence settings

    Parameters:
        sparam_diff: Maximum difference in sparams after x passes sweeping a convergence parameter. Used to check for convergence.
    """

    sparam_diff: float = 0.005

    class Config:
        arbitrary_types_allowed = True


LUMERICAL_FDTD_CONVERGENCE_SETTINGS = ConvergenceSettingsLumericalFdtd()
