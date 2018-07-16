# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

from paranoid.decorators import paranoidclass, accepts, returns
from paranoid.types import Number, Self, String, Unchecked, Dict, Or, Nothing

@paranoidclass
class FitResult:
    """An object to describe the result of a model fit."""
    @staticmethod
    def _generate():
        yield FitResult(method="Test method", loss="Likelihood",
                        value=1.1, prop1="xyz", prop2=-2)
    def _test(v):
        assert v.value in Or(Number, Nothing)
        assert v.method in String()
        assert v.loss in String()
        assert v.properties in Dict(String, Unchecked)
    def __init__(self, method, loss, value, **kwargs):
        """An object for simulation results.

        - `method` - A string describing the fitting method.
        - `loss` - A string describing the loss function
        - `value` - The optimal value of the loss function for this
          model
        - `kwargs` is a dict of any additional properties that should
          be saved for the fit.
        """
        self.value = value
        self.method = method
        self.loss = loss
        self.properties = kwargs
    @accepts(Self)
    @returns(Number)
    def value(self):
        """Returns the objective function value of the fit."""
        if self.value is not None:
            return self.value
        else:
            raise RuntimeError("Error, function has not been fit")
    @accepts(Self, String)
    def property(self, prop):
        """Get extra fit data associated with the fit.

        `prop` should be a string of the name of the property to get.
        If that property doesn't exist, None is returned.

        For example, BIC is computed using likelihood, so
        property("likelihood") is set by the BIC fitting method.
        """

        if prop in self.properties.keys():
            return self.properties[prop]
        else:
            return None

class FitResultEmpty(FitResult):
    """A default Fit object before a model has been fit."""
    def __init__(self):
        self.value = None
        self.properties = {}
        self.method = ""
        self.loss = ""
