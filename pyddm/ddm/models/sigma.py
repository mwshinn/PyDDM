from scipy import sparse
import scipy.sparse.linalg

from .base import Dependence

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
        """Cache sparse.diags to save 15% runtime."""
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
    def get_matrix(self, x, t, dx, dt, conditions, **kwargs):
        """The diffusion component of the implicit method diffusion matrix across the domain `x` at time `t`.

        `x` should be a length N ndarray.
        `t` should be a float for the time.
        """
        sigma = self.get_sigma(x=x, t=t, dx=dx, dt=dt, conditions=conditions, **kwargs)
        return self._cached_sparse_diags([1.0*sigma**2 * dt/dx**2,
                                          -0.5*sigma**2 * dt/dx**2,
                                          -0.5*sigma**2 * dt/dx**2],
                                         [0, 1, -1], shape=(len(x), len(x)), format="csr")
    def get_flux(self, x_bound, t, dx, dt, conditions, **kwargs):
        """The diffusion component of flux across the boundary at position `x_bound` at time `t`.

        Flux here is essentially the amount of the mass of the PDF
        that is past the boundary point `x_bound` at time `t` (a float).
        """
        return 0.5*dt/dx**2 * self.get_sigma(x=x_bound, t=t, dx=dx, dt=dt, conditions=conditions, **kwargs)**2
    def sigma_base(self, conditions):
        """Return the value of sigma at the beginning of the simulation."""
        assert "sigma" in self.required_parameters, "Sigma must be a required parameter"
        return self.sigma
    def get_sigma(self, conditions, **kwargs):
        raise NotImplementedError

class SigmaConstant(Sigma):
    """Simga dependence: diffusion rate/noise is constant throughout the simulation.

    Only take one parameter: sigma, the diffusion rate.

    Note that this is a special case of SigmaLinear."""
    name = "constant"
    required_parameters = ["sigma"]
    def get_sigma(self, **kwargs):
        return self.sigma

class SigmaLinear(Sigma):
    """Sigma dependence: diffusion rate varies linearly with position and time.

    Take three parameters:

    - `sigma` - The starting diffusion rate/noise
    - `x` - The coefficient by which sigma varies with x
    - `t` - The coefficient by which sigma varies with t
    """
    name = "linear_xt"
    required_parameters = ["sigma", "x", "t"]
    def get_matrix(self, x, t, dx, dt, conditions, **kwargs):
        diagadj = self.sigma + self.x*x + self.t*t
        diagadj[diagadj<0] = 0
        return sparse.diags(1.0*(diagadj)**2 * dt/dx**2, 0) \
             - sparse.diags(0.5*(diagadj[1:])**2 * dt/dx**2, 1) \
             - sparse.diags(0.5*(diagadj[:-1])**2 * dt/dx**2,-1)
    def get_flux(self, x_bound, t, dx, dt, conditions, **kwargs):
        fluxadj = (self.sigma + self.x*x_bound + self.t*t)
        if fluxadj < 0:
            return 0
        return 0.5*dt/dx**2 * fluxadj**2
    def get_sigma(self, x, t, **kwargs):
        return self.sigma + self.x*x + self.t*t
