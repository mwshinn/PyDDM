__all__ = ["Bound", "BoundConstant", "BoundCollapsingLinear", "BoundCollapsingExponential"]

import numpy as np

from .base import Dependence
from paranoid import *

class Bound(Dependence):
    """Subclass this to specify how bounds vary with time.

    This abstract class provides the methods which define a dependence
    of the bounds on t.  To subclass it, implement get_bound.  All
    subclasses must include a parameter `B` in required_parameters,
    which is the upper bound at the start of the simulation.  (The
    lower bound is symmetrically -B.)

    Also, since it inherits from Dependence, subclasses must also
    assign a `name` and `required_parameters` (see documentation for
    Dependence.)
    """
    depname = "Bound"
    ## Second effect of Collapsing Bounds: Collapsing Center: Positive
    ## and Negative states are closer to each other over time.
    def get_bound(self, t, conditions, **kwargs):
        """Return the bound at time `t`."""
        raise NotImplementedError

@paranoidclass
class BoundConstant(Bound):
    """Bound dependence: bound is constant throuhgout the simulation.

    Takes only one parameter: `B`, the constant bound."""
    name = "constant"
    required_parameters = ["B"]
    @staticmethod
    def _test(v):
        assert v.B in Positive()
    @staticmethod
    def _generate():
        yield BoundConstant(B=1)
        yield BoundConstant(B=100)
    @accepts(Self)
    @returns(Positive)
    def get_bound(self, *args, **kwargs):
        return self.B

@paranoidclass
class BoundCollapsingLinear(Bound):
    """Bound dependence: bound collapses linearly over time.

    Takes two parameters: 

    `B` - the bound at time t = 0.
    `t` - the slope, i.e. the coefficient of time, should be greater
    than zero.
    """
    name = "collapsing_linear"
    required_parameters = ["B", "t"]
    @staticmethod
    def _test(v):
        assert v.B in Positive()
        assert v.t in Number()
    @staticmethod
    def _generate():
        yield BoundCollapsingLinear(B=1, t=1)
        yield BoundCollapsingLinear(B=100, t=50.1)
    @accepts(Self, Positive0)
    @returns(Positive0)
    def get_bound(self, t, *args, **kwargs):
        return max(self.B - self.t*t, 0.)

@paranoidclass
class BoundCollapsingExponential(Bound):
    """Bound dependence: bound collapses exponentially over time.

    Takes two parameters: 

    `B` - the bound at time t = 0.
    `tau` - the time constant for the collapse, should be greater than
    zero.
    """
    name = "collapsing_exponential"
    required_parameters = ["B", "tau"]
    @staticmethod
    def _test(v):
        assert v.B in Positive()
        assert v.tau in Positive()
    @staticmethod
    def _generate():
        yield BoundCollapsingExponential(B=1, tau=1)
        yield BoundCollapsingExponential(B=.1, tau=.001)
        yield BoundCollapsingExponential(B=100, tau=100)
    @accepts(Self, Positive0)
    @returns(Positive0)
    def get_bound(self, t, *args, **kwargs):
        return self.B * np.exp(-self.tau*t)

