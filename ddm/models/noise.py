# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

__all__ = ["Noise", "NoiseConstant", "NoiseLinear"]

import numpy as np
from scipy import sparse
import scipy.sparse.linalg

from .base import Dependence
from paranoid import *

@paranoidclass
class Noise(Dependence):
    """Subclass this to specify how diffusion rate/noise varies with position and time.

    This abstract class provides the methods which define a dependence
    of noise on x and t.  To subclass it, implement get_noise.  Since
    it inherits from Dependence, subclasses must also assign a `name`
    and `required_parameters` (see documentation for Dependence.)
    """
    depname = "Noise"
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
    @accepts(Self, x=NDArray(d=1, t=Number), t=Positive0, dx=Positive, dt=Positive, conditions=Dict(k=String, v=Number))
    @returns(sparse.spmatrix)
    @ensures("return.shape == (len(x), len(x))")
    def get_matrix(self, x, t, dx, dt, conditions, **kwargs):
        """The diffusion component of the implicit method diffusion matrix across the domain `x` at time `t`.

        `x` should be a length N ndarray of all positions in the grid.
        `t` should be the time in seconds at which to calculate noise.
        `dt` and `dx` should be the simulations timestep and grid step
        `conditions` should be the conditions at which to calculate noise

        Returns a sparse NxN numpy matrix.

        There is generally no need to redefine this method in
        subclasses.
        """
        noise = self.get_noise(x=x, t=t, dx=dx, dt=dt, conditions=conditions, **kwargs)
        if np.isscalar(noise):
            return self._cached_sparse_diags([ 1.0*noise**2 * dt/dx**2,
                                              -0.5*noise**2 * dt/dx**2,
                                              -0.5*noise**2 * dt/dx**2],
                                             [0, 1, -1], shape=(len(x), len(x)), format="csr")
        else:
            return self._cached_sparse_diags([ 1.0*noise**2 * dt/dx**2,
                                              -0.5*(0.5*(noise[1:]+noise[:-1]))**2 * dt/dx**2,
                                              -0.5*(0.5*(noise[1:]+noise[:-1]))**2 * dt/dx**2],
                                             [0, 1, -1], format="csr")
    @accepts(Self, x_bound=Number, t=Positive0, dx=Positive, dt=Positive, conditions=Dict(k=String, v=Number))
    @returns(Positive0)
    def get_flux(self, x_bound, t, dx, dt, conditions, **kwargs):
        """The diffusion component of flux across the boundary at position `x_bound` at time `t`.

        Flux here is essentially the amount of the mass of the PDF
        that is past the boundary point `x_bound` at time `t` (in
        seconds).

        Note that under the central scheme we want to use x at
        half-grid from the boundary. This is however cleaner and
        justifiable using forward/backward scheme.

        There is generally no need to redefine this method in
        subclasses.
        """
        return 0.5*dt/dx**2 * self.get_noise(x=x_bound, t=t, dx=dx, dt=dt, conditions=conditions, **kwargs)**2
    def get_noise(self, conditions, **kwargs):
        """Calculate the instantaneous noise (diffusion rate).

        This function must be redefined in subclasses.

        It may take several arguments:

        - `t` - The time at which noise should be calculated
        - `x` - The particle position (or 1-dimensional NDArray of
          particle positions) at which noise should be calculated
        - `conditions` - A dictionary describing the task conditions

        It should return a number or an NDArray (the same as `x`)
        indicating the noise (diffusion rate) at that particular time,
        position(s), and task conditions.

        Definitions of this method in subclasses should only have
        arguments for needed variables and should always be followed
        by "**kwargs".  For example, if the function does not depend
        on `t` or `x` but does depend on task conditions, this should
        be:

          | def get_noise(self, conditions, **kwargs):

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
        raise NotImplementedError

@paranoidclass
class NoiseConstant(Noise):
    """Simga dependence: diffusion rate/noise is constant throughout the simulation.

    Only take one parameter: noise, the diffusion rate.

    Note that this is a special case of NoiseLinear.

    Example usage:

      | noise = NoiseConstant(noise=0.5)
    """
    name = "constant"
    required_parameters = ["noise"]
    @staticmethod
    def _test(v):
        assert v.noise in Positive()
    @staticmethod
    def _generate():
        yield NoiseConstant(noise=.001)
        yield NoiseConstant(noise=.5)
        yield NoiseConstant(noise=1)
        yield NoiseConstant(noise=2)
        yield NoiseConstant(noise=100)
    @accepts(Self)
    @returns(Number)
    def get_noise(self, **kwargs):
        return self.noise

@paranoidclass
class NoiseLinear(Noise):
    """Noise dependence: diffusion rate varies linearly with position and time.

    Take three parameters:

    - `noise` - The starting diffusion rate/noise
    - `x` - The coefficient by which noise varies with x
    - `t` - The coefficient by which noise varies with t

    Example usage:

      | noise = NoiseLinear(noise=0.5, x=0, t=.1) # Noise increases over time
    """
    name = "linear_xt"
    required_parameters = ["noise", "x", "t"]
    @staticmethod
    def _test(v):
        assert v.noise in Positive0()
        assert v.x in Number()
        assert v.t in Number()
    @staticmethod
    def _generate():
        yield NoiseLinear(noise=0, x=0, t=0)
        yield NoiseLinear(noise=1, x=-1, t=1)
        yield NoiseLinear(noise=10, x=10, t=10)
        yield NoiseLinear(noise=1, x=-10, t=-.5)
    @accepts(Self, Or(Number, NDArray(d=1, t=Number)), Positive0)
    @returns(Or(Positive, NDArray(d=1, t=Positive)))
    @requires('self.noise + self.x*x + self.t*t > 0') # Noise can't go below zero
    @ensures("np.isscalar(x) <--> np.isscalar(return)")
    def get_noise(self, x, t, **kwargs):
        return self.noise + self.x*x + self.t*t
