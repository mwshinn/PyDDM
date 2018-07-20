# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

__all__ = ["InitialCondition", "ICPointSourceCenter", "ICUniform", "ICArbitrary"]

import numpy as np

from .base import Dependence
from paranoid import accepts, returns, requires, ensures, paranoidclass
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

        This function must be redefined in subclasses.

        `x` is a length N ndarray representing the support of the
        initial condition PDF, i.e. the x-domain.  This returns a
        length N ndarray describing the distribution.
        """
        raise NotImplementedError

@paranoidclass
class ICPointSourceCenter(InitialCondition):
    """Initial condition: a dirac delta function in the center of the domain.

    Example usage:

      | ic = ICPointSourceCenter()
    """
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
    @ensures('math.fsum(return) == 1')
    @ensures('list(reversed(return)) == list(return)')
    @ensures('len(set(return)) in [1, 2]')
    def get_IC(self, x, *args, **kwargs):
        pdf = np.zeros(len(x))
        pdf[int((len(x)-1)/2)] = 1. # Initial condition at x=0, center of the channel.
        return pdf

@paranoidclass
class ICUniform(InitialCondition):
    """Initial condition: a uniform distribution.

    Example usage:

      | ic = ICUniform()
    """
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
    @ensures('math.fsum(return) == 1')
    @ensures('list(reversed(return)) == list(return)')
    @ensures('len(set(return)) in [1, 2]')
    def get_IC(self, x, *args, **kwargs):
        pdf = np.zeros(len(x))
        pdf = 1/(len(x))*np.ones((len(x)))
        return pdf

@accepts(NDArray(d=1))
@returns(InitialCondition)
@requires('math.fsum(dist) == 1')
def ICArbitrary(dist):
    """Generate an IC object from an arbitrary distribution.

    `dist` should be a 1 dimensional numpy array which sums to 1.

    Note that ICArbitrary is a function, not an InitialCondition
    object, so it cannot be passed directly.  It returns an instance
    of a an InitialCondition object which can be passed.  So in place
    of, e.g. ICUniform().  In practice, the user should not notice a
    difference, and this function can thus be used in place of an
    InitialCondition object.

    Example usage:

      | import scipy.stats
      | ic = ICPointSourceCenter(dist=scipy.stats.binom.pmf(n=200, p=.4, k=range(0, 201))) # Binomial distribution
      | import numpy as np
      | ic = ICPointSourceCenter(dist=np.asarray([0]*100+[1]+[0]*100)) # Equivalent to ICPointSourceCenter for dx=.01
    """
    class ICArbitrary(InitialCondition):
        """Initial condition from an arbitrary distribution"""
        name = "Arbitrary distribution"
        required_parameters = []
        def get_IC(self, x, _prevdist=dist, *args, **kwargs):
            assert len(x) == len(_prevdist)
            return _prevdist
    return ICArbitrary()
