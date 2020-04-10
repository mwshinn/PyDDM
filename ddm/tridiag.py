# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

# This file implements a diagonal sparse matrix format.  Converting
# between formats for binops is causing problems in the scipy sparse
# matrix routines.

import paranoid.types as pt
import paranoid.decorators as pns
from scipy import sparse
import numpy as np

import scipy.linalg.lapack as lapack

@pns.paranoidclass
class TriDiagMatrix:
    """A tri-tiagonal sparse matrix.

    Note that not all matrix operations are defined for this class,
    and that these operations are only compatible with scalars and
    with other TriDiagonal Matrix objects.
    """
    @staticmethod
    def _test(v):
        assert v.shape[0] == v.shape[1] == len(v.diag)
        assert len(v.up) == len(v.diag) - 1
        assert len(v.down) == len(v.diag) - 1
        assert v.diag in pt.NDArray(d=1)
        assert v.up in pt.NDArray(d=1)
        assert v.down in pt.NDArray(d=1)
        assert v.diag.dtype == np.dtype('float64')
        assert v.up.dtype == np.dtype('float64')
        assert v.down.dtype == np.dtype('float64')
        assert not np.any(np.isnan(v.diag))
        assert not np.any(np.isnan(v.up))
        assert not np.any(np.isnan(v.down))
    @staticmethod
    def _generate():
        yield TriDiagMatrix.eye(1)*1.1
        yield TriDiagMatrix.eye(2)
        yield TriDiagMatrix.eye(101)
        yield TriDiagMatrix(diag=np.asarray([1, 2, 3, 4]),
                            up=np.asarray([5, 3, 1]),
                            down=np.asarray([0, 1, 4]))
        yield TriDiagMatrix(up=np.asarray([4.2, 1.2]),
                            down=np.asarray([7.6, 2.1]))
    @pns.accepts(pt.Self, diag=pt.Or(pt.NDArray(d=1, t=pt.Number), pt.Nothing),
                          up=pt.NDArray(d=1, t=pt.Number),
                          down=pt.NDArray(d=1, t=pt.Number))
    @pns.requires("len(up) == len(down)")
    @pns.requires("diag is not None --> len(diag) == len(up) + 1")
    def __init__(self, diag=None, up=None, down=None):
        """Create a new TriDiagMatirx object from three vectors.

        `diag` is a vector of diagonal elements, or None if the
        diagonal is zero.  `up` and `down` are the above and below
        diagonal elements of the matrix, but may not be None.  The
        length of `up` and `down` should be equal, and should be one
        less than the length of `diag`.
        """
        assert up is not None and down is not None, "Need off-diagonals"
        if diag is None:
            diag = np.zeros(len(up)+1)
        self.diag = diag.astype('float64')
        self.up = up.astype('float64')
        self.down = down.astype('float64')
        self.shape = (len(self.diag), len(self.diag))
    @pns.accepts(pt.Self)
    @pns.returns(sparse.spmatrix)
    def to_scipy_sparse(self):
        """Returns the matrix as a scipy sparse matrix in CSR format."""
        if len(self.up) == 0:
            return sparse.diags([self.diag], [0])
        else:
            return sparse.diags([self.up, self.diag, self.down], [1, 0, -1], format="csr")
    @classmethod
    def eye(cls, size):
        """Return an identity matrix of size `size`."""
        return cls(diag=np.ones(size), up=np.zeros(size-1), down=np.zeros(size-1))
    #@pns.accepts(pt.Self, pt.Integer, pt.Integer) TODO determine domain
    #@pns.returns(pt.Self)
    def splice(self, lower, upper):
        """The splice operator implemented as a function.

        "Lower" and "upper" should be integers less than the size of
        the matrix.

        For a TriDiagonal matrix A, returns the TriDiagonal submatrix
        A[lower:upper,lower:upper].
        """
        while upper < 0:
            upper += len(self.diag)
        return TriDiagMatrix(diag=self.diag[lower:upper], up=self.up[lower:upper-1], down=self.down[lower:upper-1])
    @pns.accepts(pt.Self, pt.Or(pt.Self, pt.NDArray(d=1, t=pt.Number)))
    #@pns.returns(pt.Or(pt.Self, pt.NDArray(d=1, t=pt.Number))) # Bug with Self in Or for @returns
    @pns.requires("self.shape == other.shape or (self.shape[0],) == other.shape")
    def dot(self, other):
        """Performs matrix multipilcation of the matrix with `other`.

        `other` should either be another TriDiagonal matrix, or else a
        vector.

        Returns a Scipy sparse (csr) matrix when dotting with a matrix
        or a Numpy ndarray when dotting with a vector.

        """
        if self.shape == other.shape: # Matrix multiplication
            if self.shape == (1,1):
                return (self * other).to_scipy_sparse()
            downdown = self.down[1:] * other.down[:-1]
            down = self.down * other.diag[:-1] + self.diag[1:] * other.down
            diag = self.diag * other.diag
            diag[:-1] += self.up * other.down
            diag[1:] += self.down * other.up
            up = self.diag[:-1] * other.up + self.up * other.diag[1:]
            upup = self.up[:-1] * other.up[1:]
            # Old numpy versions throw an error if upup or downdown
            # are empty
            if len(upup) != 0:
                return sparse.diags([upup, up, diag, down, downdown], [2, 1, 0, -1, -2], format="csr")
            else:
                return sparse.diags([up, diag, down], [1, 0, -1], format="csr")
        elif (self.shape[0],) == other.shape: # Multiply by a vector
            v = self.diag * other
            if self.shape[0] > 1:
                v[:-1] += self.up * other[1:]
                v[1:] += self.down * other[:-1]
            return v
        else:
            raise ValueError("Incompatible shapes " + str(self.shape) + " and " + str(other.shape))
    @pns.accepts(pt.Self, pt.NDArray(d=1, t=pt.Number))
    @pns.returns(pt.NDArray(d=1, t=pt.Number))
    @pns.requires("len(vec) == self.shape[0]")
    def spsolve(self, vec):
        """For a matrix A, solves the equation "Ax = vec" for x.
        """
        if len(vec) == 1:
            return vec/self.diag
        (_, _, _, x, _) = lapack.dgtsv(self.down, self.diag, self.up, vec)
        return x
    @pns.accepts(pt.Self, pt.Or(pt.Number, pt.Self))
    @pns.requires("not np.isscalar(other) --> self.shape == other.shape")
    @pns.returns(pt.Self)
    def __add__(self, other):
        """Addition, defined only for scalars and TriDiagonal matrices."""
        if np.isscalar(other):
            return TriDiagMatrix(diag=self.diag + other,
                                 up=self.up + other,
                                 down=self.down + other)
        else:
            return TriDiagMatrix(diag=self.diag + other.diag,
                                 up=self.up + other.up,
                                 down=self.down + other.down)
    @pns.accepts(pt.Self, pt.Or(pt.Number, pt.Self))
    @pns.requires("not np.isscalar(other) --> self.shape == other.shape")
    @pns.returns(pt.Self)
    def __sub__(self, other):
        """Subtraction, defined only for scalars and TriDiagonal matrices."""
        if np.isscalar(other):
            return TriDiagMatrix(diag=self.diag - other,
                                 up=self.up - other,
                                 down=self.down - other)
        else:
            return TriDiagMatrix(diag=self.diag - other.diag,
                                 up=self.up - other.up,
                                 down=self.down - other.down)
    @pns.accepts(pt.Self, pt.Or(pt.Number, pt.Self))
    @pns.requires("not np.isscalar(other) --> self.shape == other.shape")
    @pns.returns(pt.Self)
    def __mul__(self, other):
        """Multipilcation, defined only for scalars and TriDiagonal matrices."""
        if np.isscalar(other):
            return TriDiagMatrix(diag=self.diag * other,
                                 up=self.up * other,
                                 down=self.down * other)
        else:
            return TriDiagMatrix(diag=self.diag * other.diag,
                                 up=self.up * other.up,
                                 down=self.down * other.down)
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    @pns.accepts(pt.Self, pt.Or(pt.Number, pt.Self))
    @pns.requires("not np.isscalar(other) --> self.shape == other.shape")
    @pns.returns(pt.Self)
    def __iadd__(self, other):
        """In-place addition, defined only for scalars and TriDiagonal matrices."""
        if np.isscalar(other):
            self.up += other
            self.down += other
            self.diag += other
        else:
            self.up += other.up
            self.down += other.down
            self.diag += other.diag
        return self
    @pns.accepts(pt.Self, pt.Or(pt.Number, pt.Self))
    @pns.requires("not np.isscalar(other) --> self.shape == other.shape")
    @pns.returns(pt.Self)
    def __isub__(self, other):
        """In-place subtraction, defined only for scalars and TriDiagonal matrices."""
        if np.isscalar(other):
            self.up -= other
            self.down -= other
            self.diag -= other
        else:
            self.up -= other.up
            self.down -= other.down
            self.diag -= other.diag
        return self
    @pns.accepts(pt.Self, pt.Or(pt.Number, pt.Self))
    @pns.requires("not np.isscalar(other) --> self.shape == other.shape")
    @pns.returns(pt.Self)
    def __imul__(self, other):
        """In-place multiplication, defined only for scalars and TriDiagonal matrices."""
        if np.isscalar(other):
            self.up *= other
            self.down *= other
            self.diag *= other
        else:
            self.up *= other.up
            self.down *= other.down
            self.diag *= other.diag
        return self
    @pns.accepts(pt.Self, pt.Self)
    @pns.returns(pt.Boolean)
    def __eq__(self, other):
        if self.shape != other.shape:
            return False
        if np.all([np.all(self.up == other.up),
                   np.all(self.down == other.down),
                   np.all(self.diag == other.diag)]):
            return True
           
        
