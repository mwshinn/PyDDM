# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

__all__ = ["Drift", "DriftConstant", "DriftLinear"]

import numpy as np
from ..tridiag import TriDiagMatrix

from .base import Dependence
from paranoid import *
from .paranoid_types import Conditions

@paranoidclass
class Drift(Dependence):
    """Subclass this to specify how drift rate varies with position and time.

    This abstract class provides the methods which define a dependence
    of drift on x and t.  To subclass it, implement get_drift.  Since
    it inherits from Dependence, subclasses must also assign a `name`
    and `required_parameters` (see documentation for Dependence.)
    """
    depname = "Drift"
    def _uses_t(self):
        return self._uses(self.get_drift, "t")
    def _uses_x(self):
        return self._uses(self.get_drift, "x")
    @accepts(Self, x=NDArray(d=1, t=Number), t=Positive0, dx=Positive, dt=Positive, conditions=Conditions, implicit=Boolean)
    @returns(TriDiagMatrix)
    @ensures("return.shape == (len(x), len(x))")
    def get_matrix(self, x, t, dx, dt, conditions, implicit=False, **kwargs):
        """The drift component of the implicit method diffusion matrix across the domain `x` at time `t`.

        `x` should be a length N ndarray of all positions in the grid.
        `t` should be the time in seconds at which to calculate drift.
        `dt` and `dx` should be the simulations timestep and grid step
        `conditions` should be the conditions at which to calculate drift

        Returns a sparse NxN matrix as a PyDDM TriDiagMatrix object.

        There is generally no need to redefine this method in
        subclasses.
        """
        drift = self.get_drift(x=x, t=t, dx=dx, dt=dt, conditions=conditions, **kwargs)
        D = np.zeros(len(x))
        if np.isscalar(drift):
            UP = 0.5*dt/dx * drift * np.ones(len(x)-1)
            DOWN = -0.5*dt/dx * drift * np.ones(len(x)-1)
        else:
            UP = 0.5*dt/dx * drift[1:]
            DOWN = -0.5*dt/dx * drift[:-1]
        if implicit:
            D[-1] = UP[-1]
            UP[-1] = 0
            D[0] = DOWN[0]
            DOWN[0] = 0
        return TriDiagMatrix(up=UP,
                             down=DOWN,
                             diag=D)
    # Amount of flux from bound/end points to correct and erred
    # response probabilities, due to different parameters.
    @accepts(Self, x_bound=Number, t=Positive0, dx=Positive, dt=Positive, conditions=Conditions)
    @returns(Number)
    def get_flux(self, x_bound, t, dx, dt, conditions, **kwargs):
        """The drift component of flux across the boundary at position `x_bound` at time `t`.

        Flux here is essentially the amount of the mass of the PDF
        that is past the boundary point `x_bound`.

        There is generally no need to redefine this method in
        subclasses.
        """
        return 0.5*dt/dx * np.sign(x_bound) * self.get_drift(x=x_bound, t=t, dx=dx, dt=dt, conditions=conditions, **kwargs)
    def get_drift(self, t, x, conditions, **kwargs):
        """Calculate the instantaneous drift rate.

        This function must be redefined in subclasses.

        It may take several arguments:

        - `t` - The time at which drift should be calculated
        - `x` - The particle position (or 1-dimensional NDArray of
          particle positions) at which drift should be calculated
        - `conditions` - A dictionary describing the task conditions

        It should return a number or an NDArray (the same as `x`)
        indicating the drift rate at that particular time,
        position(s), and task conditions.

        Definitions of this method in subclasses should only have
        arguments for needed variables and should always be followed
        by "**kwargs".  For example, if the function does not depend
        on `t` or `x` but does depend on task conditions, this should
        be:

          | def get_drift(self, conditions, **kwargs):

        Of course, the function would still work properly if `x` were
        included as an argument, but this convention allows PyDDM to
        automatically select the best simulation methods for the
        model.

        If a function depends on `x`, it should return a scalar if `x`
        is a scalar, or an NDArray of the same size as `x` if `x` is
        an NDArray.  If the function does not depend on `x`, it should
        return a scalar.  (The purpose of this is a dramatic speed
        increase by using numpy vectorization.)
        """
        raise NotImplementedError("Drift model %s invalid: must define the get_drift function" % self.__class__.__name__)

@paranoidclass
class DriftConstant(Drift):
    """Drift dependence: drift rate is constant throughout the simulation.

    Only take one parameter: drift, the constant drift rate.

    Note that this is a special case of DriftLinear.

    Example usage:

      | drift = DriftConstant(drift=0.3)
    """
    name = "constant"
    required_parameters = ["drift"]
    @staticmethod
    def _test(v):
        assert v.drift in Number()
    @staticmethod
    def _generate():
        yield DriftConstant(drift=0)
        yield DriftConstant(drift=1)
        yield DriftConstant(drift=-1)
        yield DriftConstant(drift=100)
    @accepts(Self)
    @returns(Number)
    def get_drift(self, **kwargs):
        return self.drift

@paranoidclass
class DriftLinear(Drift):
    """Drift dependence: drift rate varies linearly with position and time.

    Take three parameters:

    - `drift` - The starting drift rate
    - `x` - The coefficient by which drift varies with x
    - `t` - The coefficient by which drift varies with t

    Example usage:

      | drift = DriftLinear(drift=0.5, t=0, x=-1) # Leaky integrator
      | drift = DriftLinear(drift=0.8, t=0, x=0.4) # Unstable integrator
      | drift = DriftLinear(drift=0, t=1, x=0.4) # Urgency function
    """
    name = "linear_xt"
    required_parameters = ["drift", "x", "t"]
    @staticmethod
    def _test(v):
        assert v.drift in Number()
        assert v.x in Number()
        assert v.t in Number()
    @staticmethod
    def _generate():
        yield DriftLinear(drift=0, x=0, t=0)
        yield DriftLinear(drift=1, x=-1, t=1)
        yield DriftLinear(drift=10, x=10, t=10)
        yield DriftLinear(drift=1, x=-10, t=-.5)
    # We allow this function to accept a vector or a scalar for x,
    # because if we use list comprehensions instead of native numpy
    # multiplication in the get_matrix function it slows things down by
    # around 100x.
    @accepts(Self, Or(Number, NDArray(d=1, t=Number)), Positive0)
    @returns(Or(Number, NDArray(d=1, t=Number)))
    @ensures("np.isscalar(x) <--> np.isscalar(return)")
    def get_drift(self, x, t, **kwargs):
        return self.drift + self.x*x + self.t*t
    def _uses_t(self):
        return self.t != 0
    def _uses_x(self):
        return self.x != 0
