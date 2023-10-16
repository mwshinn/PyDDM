# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

__all__ = ["InitialCondition", "ICPointSourceCenter", "ICPoint", "ICPointRatio", "ICUniform", "ICRange", "ICGaussian", "ICArbitrary"]

import numpy as np

from .base import Dependence
from paranoid import accepts, returns, requires, ensures, paranoidclass
from paranoid.types import NDArray, Number, Positive, Range, Unchecked
from paranoid.types import Self
import scipy.stats

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
        raise NotImplementedError("IC model %s invalid: must define the get_IC function" % self.__class__.__name__)

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
    @ensures('np.isclose(np.sum(return), 1)')
    @ensures('list(reversed(return)) == list(return)')
    @ensures('len(set(return)) in [1, 2]')
    def get_IC(self, x, *args, **kwargs):
        pdf = np.zeros(len(x))
        pdf[int((len(x)-1)/2)] = 1. # Initial condition at x=0, center of the channel.
        return pdf

@paranoidclass
class ICPoint(InitialCondition):
    """Initial condition: any point.

    Example usage:

      | ic = ICPoint(x0=.2)
    """
    name = "An arbitrary starting point."
    required_parameters = ["x0"]
    @staticmethod
    def _test(v):
        assert v.x0 in Number()
    @staticmethod
    def _generate():
        yield ICPoint(x0=.2)
        yield ICPoint(x0=-.111)
    @accepts(Self, NDArray(d=1), Positive, Unchecked)
    @returns(NDArray(t=Number, d=1))
    @ensures('np.isclose(np.sum(return), 1)')
    @ensures('len(set(return)) == 2')
    def get_IC(self, x, dx, conditions):
        start = np.round(self.get_starting_point(conditions=conditions)/dx)
        # Positive bias for high reward conditions, negative for low reward
        shift_i = int(start + (len(x)-1)/2)
        assert shift_i >= 0 and shift_i < len(x), "Invalid initial conditions: " \
            "Please ensure the value of the parameter x0 falls within the bounds."
        pdf = np.zeros(len(x))
        pdf[shift_i] = 1. # Initial condition at x=self.x0.
        return pdf
    def get_starting_point(self, conditions):
        return self.x0

@paranoidclass
class ICPointRatio(InitialCondition):
    """Initial condition: any point expressed as a ratio between bounds, from -1 to 1.

    Example usage:

      | ic = ICPointRatio(x0=-.2)

    The advantage of ICPointRatio over ICPoint is that, as long as x0 is
    greater than -1 and less than 1, the starting point will always stay within
    the bounds, even when bounds are being fit.
    """
    name = "An arbitrary starting point expressed as a proportion of the distance between the bounds."
    required_parameters = ["x0"]
    @staticmethod
    def _test(v):
        assert v.x0 in Range(-1, 1)
    @staticmethod
    def _generate():
        yield ICPointRatio(x0=.2)
        yield ICPointRatio(x0=-.8)
    def get_IC(self, x, dx, conditions):
        x0 = self.get_starting_point(conditions=conditions)/2 + .5 #rescale to between 0 and 1
        shift_i = int((len(x)-1)*x0)
        assert shift_i >= 0 and shift_i < len(x), "Invalid initial conditions"
        pdf = np.zeros(len(x))
        pdf[shift_i] = 1.
        return pdf
    def get_starting_point(self, conditions):
        return self.x0

@paranoidclass
class ICUniform(InitialCondition):
    """Initial condition: a uniform distribution.

    Example usage:

      | ic = ICUniform()
    """
    name = "Uniform"
    required_parameters = []
    @staticmethod
    def _test(v):
        pass
    @staticmethod
    def _generate():
        yield ICUniform()
    @accepts(Self, NDArray(d=1))
    @returns(NDArray(t=Number, d=1))
    @ensures('np.isclose(np.sum(return), 1)')
    @ensures('list(reversed(return)) == list(return)')
    @ensures('len(set(return)) in [1, 2]')
    def get_IC(self, x, *args, **kwargs):
        pdf = 1/(len(x))*np.ones((len(x)))
        return pdf

@paranoidclass
class ICRange(InitialCondition):
    """Initial condition: a bounded uniform distribution with range from -sz to sz.

    Example usage:

      | ic = ICRange(sz=.3)
    """
    name = "Uniform range"
    required_parameters = ["sz"]
    @staticmethod
    def _test(v):
        assert v.sz >= 0, "sz parameter must be positive"
    @staticmethod
    def _generate():
        yield ICRange(sz=.1)
    @accepts(Self, NDArray(d=1), Positive)
    @requires("max(x) >= self.sz")
    @requires("min(x) <= -self.sz")
    @returns(NDArray(t=Number, d=1))
    @ensures('np.isclose(np.sum(return), 1)')
    @ensures('list(reversed(return)) == list(return)')
    @ensures('len(set(return)) in [1, 2]')
    def get_IC(self, x, dx, *args, **kwargs):
        pdf = np.zeros(len(x))
        center = int((len(x)-1)/2)
        width = self.sz
        # Positive bias for high reward conditions, negative for low reward
        shift_i = int((len(x)-1)/2)
        sz_shift = int(self.sz/dx)
        pdf[(shift_i-sz_shift):(shift_i+sz_shift+1)] = 1
        return pdf/np.sum(pdf)

@paranoidclass
class ICGaussian(InitialCondition):
    """Initial condition: a Gaussian distribution with a specified standard deviation.

    Example usage:

      | ic = ICRange(sz=.3)
    """
    name = "Gaussian"
    required_parameters = ["stdev"]
    @staticmethod
    def _test(v):
        assert v.stdev > 0, "Standard deviation must be positive"
    @staticmethod
    def _generate():
        yield ICGaussian(stdev=.2)
    @accepts(Self, NDArray(d=1, t=Number), Positive)
    @requires("np.all(np.isclose(x+x[::-1], 0))") # Symmetric around 0
    @returns(NDArray(t=Number, d=1))
    @ensures('np.isclose(np.sum(return), 1)')
    @ensures('np.all(np.isclose(return[::-1], return))') # Symmetric
    def get_IC(self, x, dx, *args, **kwargs):
        pdf = scipy.stats.norm(0, self.stdev).pdf(x)
        return pdf/np.sum(pdf)


@accepts(NDArray(d=1))
@returns(InitialCondition)
@requires('np.isclose(np.sum(dist), 1)')
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
      | ic = ICArbitrary(dist=scipy.stats.binom.pmf(n=200, p=.4, k=range(0, 201))) # Binomial distribution
      | import numpy as np
      | ic = ICArbitrary(dist=np.asarray([0]*100+[1]+[0]*100)) # Equivalent to ICPointSourceCenter for dx=.01
    """
    class ICArbitrary(InitialCondition):
        """Initial condition from an arbitrary distribution"""
        name = "Arbitrary distribution"
        required_parameters = []
        def get_IC(self, x, _prevdist=dist, *args, **kwargs):
            assert len(x) == len(_prevdist)
            return _prevdist
    return ICArbitrary()
