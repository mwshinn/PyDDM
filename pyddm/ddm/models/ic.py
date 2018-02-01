__ALL__ = ["InitialCondition", "ICPointSourceCenter", "ICUniform"]

import numpy as np

from .base import Dependence
from paranoid import accepts, returns, requires, ensures, verifiedclass
from paranoid.types import NDArray, Number
from paranoid.types import Self

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

@verifiedclass
class ICPointSourceCenter(InitialCondition):
    """Initial condition: a dirac delta function in the center of the domain."""
    name = "point_source_center"
    required_parameters = []
    @staticmethod
    def _test(v):
        pass
    @staticmethod
    def _generate():
        yield ICPointSourceCenter()
    @accepts(Self, NDArray(d=1))
    @returns(NDArray(t=Number, d=1))
    @ensures('sum(return) == 1')
    @ensures('list(reversed(return)) == list(return)')
    @ensures('len(set(return)) in [1, 2]')
    def get_IC(self, x, *args, **kwargs):
        pdf = np.zeros(len(x))
        pdf[int((len(x)-1)/2)] = 1. # Initial condition at x=0, center of the channel.
        return pdf

@verifiedclass
class ICUniform(InitialCondition):
    """Initial condition: a uniform distribution."""
    name = "uniform"
    required_parameters = []
    @staticmethod
    def _test(v):
        pass
    @staticmethod
    def _generate():
        yield ICUniform()
    @accepts(Self, NDArray(d=1))
    @returns(NDArray(t=Number, d=1))
    @ensures('sum(return) == 1')
    @ensures('list(reversed(return)) == list(return)')
    @ensures('len(set(return)) in [1, 2]')
    def get_IC(self, x, *args, **kwargs):
        pdf = np.zeros(len(x))
        pdf = 1/(len(x))*np.ones((len(x)))
        return pdf

