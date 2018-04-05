__ALL__ = ["Sigma", "SigmaConstant", "SigmaLinear"]

import numpy as np
from scipy import sparse
import scipy.sparse.linalg

from .base import Dependence
from paranoid import *

class Sigma(Dependence):
    """Subclass this to specify how diffusion rate/noise varies with position and time.

    This abstract class provides the methods which define a dependence
    of sigma on x and t.  To subclass it, implement get_matrix and
    get_flux.  All subclasses must include a parameter sigma in
    required_parameters, which is the diffusion rate/noise at the
    start of the simulation.

    Also, since it inherits from Dependence, subclasses must also
    assign a `name` and `required_parameters` (see documentation for
    Dependence.)
    """
    depname = "Sigma"
    def _cached_sparse_diags(self, *args, **kwargs):
        """Cache sparse.diags to save 15% runtime.

        This is basically just a wrapper for the sparse.diags function
        to incorporate a size 1 LRU memoization.  Functools
        memoization doesn't work because arguments are not hashable.

        Profiling identified the sparse.diags function as a major
        bottleneck, and the use of this function ameliorates that.
        """
        if "_last_diag_val" not in self.__dict__:
            object.__setattr__(self, "_last_diag_args", [])
            object.__setattr__(self, "_last_diag_kwargs", {})
            object.__setattr__(self, "_last_diag_val", None)
        if object.__getattribute__(self, "_last_diag_args") == args and \
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
        """The diffusion component of the implicit method diffusion matrix across the domain `x` at time `t`.

        `x` should be a length N ndarray of all positions in the grid.
        `t` should be the time in seconds at which to calculate sigma.
        `dt` and `dx` should be the simulations timestep and grid step
        `conditions` should be the conditions at which to calculate sigma

        Returns a sparse NxN numpy matrix.
        """
        sigma = self.get_sigma(x=x, t=t, dx=dx, dt=dt, conditions=conditions, **kwargs)
        if np.isscalar(sigma):
            return self._cached_sparse_diags([ 1.0*sigma**2 * dt/dx**2,
                                              -0.5*sigma**2 * dt/dx**2,
                                              -0.5*sigma**2 * dt/dx**2],
                                             [0, 1, -1], shape=(len(x), len(x)), format="csr")
        else:
            return self._cached_sparse_diags([ 1.0*sigma**2 * dt/dx**2,
                                              -0.5*sigma[1:]**2 * dt/dx**2,
                                              -0.5*sigma[:-1]**2 * dt/dx**2],
                                             [0, 1, -1], format="csr")
    @accepts(Self, x_bound=Number, t=Positive0, dx=Positive, dt=Positive, conditions=Dict(k=String, v=Number))
    @returns(Positive0)
    def get_flux(self, x_bound, t, dx, dt, conditions, **kwargs):
        """The diffusion component of flux across the boundary at position `x_bound` at time `t`.

        Flux here is essentially the amount of the mass of the PDF
        that is past the boundary point `x_bound` at time `t` (in seconds).
        """
        return 0.5*dt/dx**2 * self.get_sigma(x=x_bound, t=t, dx=dx, dt=dt, conditions=conditions, **kwargs)**2
    def get_sigma(self, conditions, **kwargs):
        raise NotImplementedError

@paranoidclass
class SigmaConstant(Sigma):
    """Simga dependence: diffusion rate/noise is constant throughout the simulation.

    Only take one parameter: sigma, the diffusion rate.

    Note that this is a special case of SigmaLinear."""
    name = "constant"
    required_parameters = ["sigma"]
    @staticmethod
    def _test(v):
        assert v.sigma in Positive()
    @staticmethod
    def _generate():
        yield SigmaConstant(sigma=.001)
        yield SigmaConstant(sigma=1)
        yield SigmaConstant(sigma=100)
    @accepts(Self)
    @returns(Number)
    def get_sigma(self, **kwargs):
        return self.sigma

#TODO While testing, make sure that with x=0 this is the same as
#constant mu.
@paranoidclass
class SigmaLinear(Sigma):
    """Sigma dependence: diffusion rate varies linearly with position and time.

    Take three parameters:

    - `sigma` - The starting diffusion rate/noise
    - `x` - The coefficient by which sigma varies with x
    - `t` - The coefficient by which sigma varies with t
    """
    name = "linear_xt"
    required_parameters = ["sigma", "x", "t"]
    @staticmethod
    def _test(v):
        assert v.sigma in Positive0()
        assert v.x in Number()
        assert v.t in Number()
    @staticmethod
    def _generate():
        yield SigmaLinear(sigma=0, x=0, t=0)
        yield SigmaLinear(sigma=1, x=-1, t=1)
        yield SigmaLinear(sigma=10, x=10, t=10)
        yield SigmaLinear(sigma=1, x=-10, t=-.5)
    @accepts(Self, Number, Number)
    @returns(Positive)
    @requires('self.sigma + self.x*x + self.t*t > 0') # Sigma can't go below zero
    def get_sigma(self, x, t, **kwargs):
        return self.sigma + self.x*x + self.t*t
