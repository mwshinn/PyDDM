# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

__all__ = ["Bound", "BoundConstant", "BoundCollapsingLinear", "BoundCollapsingExponential"]

import numpy as np

from .base import Dependence
from paranoid import *

class Bound(Dependence):
    """Subclass this to specify how bounds vary with time.

    This abstract class provides the methods which define a dependence
    of the bounds on t.  To subclass it, implement get_bound.  All
    bounds must be symmetric, so the lower bound is -get_bound.

    Also, since it inherits from Dependence, subclasses must also
    assign a `name` and `required_parameters` (see documentation for
    Dependence.)
    """
    depname = "Bound"
    def _uses_t(self):
        return self._uses(self.get_bound, "t")
    def get_bound(self, t, conditions, **kwargs):
        """Calculate the bounds which particles cross to determine response time.

        This function must be redefined in subclasses.

        It may take up to two arguments:

        - `t` - The time at which bound should be calculated
        - `conditions` - A dictionary describing the task conditions

        It should return a non-negative number indicating the upper
        bound at that particular time, and task conditions.  The lower
        bound is taken to be the negative of the upper bound.

        Definitions of this method in subclasses should only have
        arguments for needed variables and should always be followed
        by "**kwargs".  For example, if the function does not depend
        on task conditions but does depend on time, this should
        be:

          | def get_bound(self, t, **kwargs):

        Of course, the function would still work properly if
        `conditions` were included as an argument, but this convention
        allows PyDDM to automatically select the best simulation
        methods for the model.
        """
        raise NotImplementedError("Bound model %s invalid: must define the get_bound function" % self.__class__.__name__)

@paranoidclass
class BoundConstant(Bound):
    """Bound dependence: bound is constant throuhgout the simulation.

    Takes only one parameter: `B`, the constant bound.

    Example usage:

      | bound = BoundConstant(B=1.5) # Bound at 1.5 and -1.5
    """
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

    - `B` - the bound at time t = 0.
    - `t` - the slope, i.e. the coefficient of time, should be greater than zero.

    Example usage:

      | bound = BoundCollapsingLinear(B=1, t=.5) # Collapsing at .5 units per second
    """
    name = "collapsing_linear"
    required_parameters = ["B", "t"]
    @staticmethod
    def _test(v):
        assert v.B in Positive()
        assert v.t in Positive0()
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

    - `B` - the bound at time t = 0.
    - `tau` - one divided by the time constant for the collapse.
      0 gives constant bounds.

    Example usage:

      | bound = BoundCollapsingExponential(B=1, tau=2.1) # Collapsing with time constant 1/2.1
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

