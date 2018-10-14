# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

# This file implements a diagonal sparse matrix format.  Converting
# between formats for binops is causing problems in the scipy sparse
# matrix routines.

import paranoid.types as pt
from scipy import sparse
import numpy as np

class DiagMatrix:
    def __init__(self, diag=None, up=None, down=None):
        assert up is not None and down is not None, "Need off-diagonals"
        if diag is None:
            diag = np.zeros(len(up)+1)
        self.diag = diag
        self.up = up
        self.down = down
        assert up is None or len(up) == len(diag) - 1
        assert down is None or len(up) == len(diag) - 1
        #assert diag in pt.Or(pt.Nothing(), pt.NDArray(d=1, t=pt.Number))
        #assert up in pt.Or(pt.Nothing(), pt.NDArray(d=1, t=pt.Number))
        #assert down in pt.Or(pt.Nothing(), pt.NDArray(d=1, t=pt.Number))
        self.shape = (len(self.diag), len(self.diag))
    def to_scipy_sparse(self):
        return sparse.diags([self.up, self.diag, self.down], [1, 0, -1], format="csr")
    @classmethod
    def eye(cls, size):
        return cls(diag=np.ones(size), up=np.zeros(size-1), down=np.zeros(size-1))
    def splice(self, lower, upper):
        return DiagMatrix(diag=self.diag[lower:upper], up=self.up[lower:upper-1], down=self.down[lower:upper-1])
    def dot(self, other):
        if self.shape == other.shape: # Matrix multiplication
            downdown = self.down[1:] * other.down[:-1]
            down = self.down * other.diag[:-1] + self.diag[1:] * other.down
            diag = self.diag * other.diag
            diag[:-1] += self.up * other.down
            diag[1:] += self.down * other.up
            up = self.diag[:-1] * other.up + self.up * other.diag[1:]
            upup = self.up[:-1] * other.up[1:]
            return sparse.diags([upup, up, diag, down, downdown], [2, 1, 0, -1, -2], format="csr")
        elif (self.shape[0],) == other.shape: # Multiply by a vector
            v = self.diag * other
            v[:-1] += self.up * other[1:]
            v[1:] += self.down * other[:-1]
            return v
        else:
            raise ValueError("Incompatible shapes " + str(self.shape) + " and " + str(other.shape))
    def __add__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return DiagMatrix(diag=self.diag + other,
                              up=self.up + other,
                              down=self.down + other)
        else:
            return DiagMatrix(diag=self.diag + other.diag,
                              up=self.up + other.up,
                              down=self.down + other.down)
    def __sub__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return DiagMatrix(diag=self.diag - other,
                              up=self.up - other,
                              down=self.down - other)
        else:
            return DiagMatrix(diag=self.diag - other.diag,
                              up=self.up - other.up,
                              down=self.down - other.down)
    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return DiagMatrix(diag=self.diag * other,
                              up=self.up * other,
                              down=self.down * other)
        else:
            return DiagMatrix(diag=self.diag * other.diag,
                              up=self.up * other.up,
                              down=self.down * other.down)
    def __radd__(self, other):
        return self.__add__(other)
    def __rsub__(self, other):
        return self.__sub__(other)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __iadd__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            self.up += other
            self.down += other
            self.diag += other
        else:
            self.up += other.up
            self.down += other.down
            self.diag += other.diag
        return self
    def __isub__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            self.up -= other
            self.down -= other
            self.diag -= other
        else:
            self.up -= other.up
            self.down -= other.down
            self.diag -= other.diag
        return self
    def __imul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            self.up *= other
            self.down *= other
            self.diag *= other
        else:
            self.up *= other.up
            self.down *= other.down
            self.diag *= other.diag
        return self
