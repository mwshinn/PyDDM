import numpy as np

from .base import Dependence

class InitialCondition(Dependence):
    """Subclass this to compute the initial conditions of the simulation.

    This abstract class describes initial PDF at the beginning of a
    simulation.  To subclass it, implement get_IC(x).

    Also, since it inherits from Dependence, subclasses must also
    assign a `name` and `required_parameters` (see documentation for
    Dependence.)
    """
    depname = "IC"
    def get_IC(self, x, dx, **kwargs):
        """Get the initial conditions (a PDF) withsupport `x`.

        `x` is a length N ndarray representing the support of the
        initial condition PDF, i.e. the x-domain.  This returns a
        length N ndarray describing the distribution.
        """
        raise NotImplementedError

class ICPointSourceCenter(InitialCondition):
    """Initial condition: a dirac delta function in the center of the domain."""
    name = "point_source_center"
    required_parameters = []
    def get_IC(self, x, dx, **kwargs):
        pdf = np.zeros(len(x))
        pdf[int((len(x)-1)/2)] = 1. # Initial condition at x=0, center of the channel.
        return pdf

# Dependence for testing.
class ICUniform(InitialCondition):
    """Initial condition: a uniform distribution."""
    name = "uniform"
    required_parameters = []
    def get_IC(self, x, dx, **kwargs):
        pdf = np.zeros(len(x))
        pdf = 1/(len(x))*np.ones((len(x)))
        return pdf

