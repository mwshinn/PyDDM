# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

from paranoid.decorators import paranoidclass, accepts, returns
from paranoid.types import Number, Self, String, Unchecked, Dict, Nothing, ExtendedReal, Maybe
import numpy as np

@paranoidclass
class FitResult:
    """An object to describe the result of a model fit.

    This keeps track of information related to the fitting procedure.
    It has the following elements:

    - method: the name of the solver used to solve the model,
      e.g. "analytical" or "implicit"
    - fitting_method: the name of the algorithm used to minimize the
      loss function method (e.g. "differential_evolution")
    - loss: the name of the loss function (e.g. "BIC")
    - properties: a dictionary containing any additional values saved
      by the loss function or fitting procedure (e.g. "likelihood" for
      BIC loss function, or "mess" for a message describing the output).

    So, for example, can access FitResult.method to get the name of
    the numerical algorithm used to solve the equation.

    To access the output value of the loss function, use
    FitResult.value().
    """
    @staticmethod
    def _generate():
        yield FitResult(fitting_method="Test method", loss="Likelihood", method="cn",
                        value=1.1, prop1="xyz", prop2=-2)
    @staticmethod
    def _test(v):
        assert v.val in Maybe(Number)
        assert v.method in String()
        assert v.loss in String()
        assert v.properties in Dict(String, Unchecked)
    def __init__(self, fitting_method, method, loss, value, **kwargs):
        """An object for simulation results.

        - `fitting_method` - A string describing the fitting method.
        - `loss` - A string describing the loss function
        - `value` - The optimal value of the loss function for this
          model
        - `method` - The algorithm used to create the correct/error
          PDFs
        - `kwargs` is a dict of any additional properties that should
          be saved for the fit.
        """
        self.val = value
        self.method = method
        self.loss = loss
        self.properties = kwargs
        self.fitting_method = fitting_method
    def __repr__(self):
        components = ["fitting_method=%s" % repr(self.fitting_method),
                      "method=%s" % repr(self.method),
                      "loss=%s" % repr(self.loss),
                      "value=%s" % repr(self.val)]
        components += ["%s=%s" % (k,repr(v)) for k,v in self.properties.items()]
        return type(self).__name__ + "(" + ", ".join(components) + ")"
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False
    @accepts(Self)
    @returns(ExtendedReal)
    def value(self):
        """Returns the objective function value of the fit.

        If there was an error, or if no fit was performed, return
        inf.
        """
        if self.val is not None:
            return self.val
        else:
            return np.inf

class FitResultEmpty(FitResult):
    """A default Fit object before a model has been fit."""
    def __repr__(self):
        return type(self).__name__ + "()"
    def __init__(self):
        self.val = None
        self.properties = {}
        self.method = ""
        self.loss = ""
