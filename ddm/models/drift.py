__ALL__ = ["Drift", "DriftConstant", "DriftLinear"]

import numpy as np
from scipy import sparse
import scipy.sparse.linalg

from .base import Dependence
from paranoid import *

@paranoidclass
class Drift(Dependence):
    """Subclass this to specify how drift rate varies with position and time.

    This abstract class provides the methods which define a dependence
    of drift on x and t.  To subclass it, implement get_matrix and
    get_flux.  All subclasses must include a parameter drift in
    required_parameters, which is the drift rate at the start of the
    simulation.

    Also, since it inherits from Dependence, subclasses must also
    assign a `name` and `required_parameters` (see documentation for
    Dependence.)
    """
    depname = "Drift"
    def _cached_sparse_diags(self, *args, **kwargs):
        """Cache sparse.diags to save 15% runtime.

        This is basically just a wrapper for the sparse.diags function
        to incorporate a size 1 LRU memoization.  Functools
        memoization doesn't work because arguments are not hashable.

        Profiling identified the sparse.diags function as a major
        bottleneck, and the use of this function ameliorates that.
        """
        if "_last_diag_args" not in self.__dict__:
            object.__setattr__(self, "_last_diag_args", [])
            object.__setattr__(self, "_last_diag_kwargs", {})
            object.__setattr__(self, "_last_diag_val", None)
        if np.array_equal(object.__getattribute__(self, "_last_diag_args"), args) and \
           object.__getattribute__(self, "_last_diag_kwargs") == kwargs:
            return object.__getattribute__(self, "_last_diag_val")

        object.__setattr__(self, "_last_diag_args", args)
        object.__setattr__(self, "_last_diag_kwargs", kwargs)
        object.__setattr__(self, "_last_diag_val", sparse.diags(*args, **kwargs))
        return object.__getattribute__(self, "_last_diag_val")
    @accepts(Self, x=NDArray(d=1), t=Positive0, dx=Positive, dt=Positive, conditions=Dict(k=String, v=Number))
    @returns(sparse.spmatrix)
    @ensures("return.shape == (len(x), len(x))")
    def get_matrix(self, x, t, dx, dt, conditions, **kwargs):
        """The drift component of the implicit method diffusion matrix across the domain `x` at time `t`.

        `x` should be a length N ndarray of all positions in the grid.
        `t` should be the time in seconds at which to calculate drift.
        `dt` and `dx` should be the simulations timestep and grid step
        `conditions` should be the conditions at which to calculate drift

        Returns a sparse NxN numpy matrix.
        """
        drift = self.get_drift(x=x, t=t, dx=dx, dt=dt, conditions=conditions, **kwargs)
        if np.isscalar(drift):
            return self._cached_sparse_diags([ 0.5*dt/dx * drift,
                                              -0.5*dt/dx * drift],
                                             [1, -1], shape=(len(x), len(x)), format="csr")
        else:
            return self._cached_sparse_diags([ 0.5*dt/dx * drift[1:],
                                              -0.5*dt/dx * drift[:-1]],
                                             [1, -1], format="csr")
    # Amount of flux from bound/end points to correct and erred
    # response probabilities, due to different parameters.
    @accepts(Self, x_bound=Number, t=Positive0, dx=Positive, dt=Positive, conditions=Dict(k=String, v=Number))
    @returns(Number)
    def get_flux(self, x_bound, t, dx, dt, conditions, **kwargs):
        """The drift component of flux across the boundary at position `x_bound` at time `t`.

        Flux here is essentially the amount of the mass of the PDF
        that is past the boundary point `x_bound`.
        """
        return 0.5*dt/dx * np.sign(x_bound) * self.get_drift(x=x_bound, t=t, dx=dx, dt=dt, conditions=conditions, **kwargs)
    def get_drift(self, t, conditions, **kwargs):
        raise NotImplementedError

@paranoidclass
class DriftConstant(Drift):
    """Drift dependence: drift rate is constant throughout the simulation.

    Only take one parameter: drift, the constant drift rate.

    Note that this is a special case of DriftLinear."""
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
