# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

__all__ = ["Dependence"]

import paranoid

class Dependence(object): # TODO Base this on ABC
    """An abstract class describing how one variable depends on other variables.

    This is an abstract class which is inherrited by other abstract
    classes only, and has the highest level machinery for describing
    how one variable depends on others.  For example, an abstract
    class that inherits from Dependence might describe how the drift
    rate may change throughout the simulation depending on the value
    of x and t, and then this class would be inherited by a concrete
    class describing an implementation.  For example, the relationship
    between drift rate and x could be linear, exponential, etc., and
    each of these would be a subsubclass of Dependence.

    In order to subclass Dependence, you must set the (static) class
    variable `depname`, which gives an alpha-numeric string describing
    which variable could potentially depend on other variables.

    Each subsubclass of dependence must also define two (static) class
    variables.  First, it must define `name`, which is an
    alpha-numeric plus underscores name of what the algorithm is, and
    also `required_parameters`, a python list of names (strings) for
    the parameters that must be passed to this algorithm.  (This does
    not include globally-relevant variables like dt, it only includes
    variables relevant to a particular instance of the algorithm.)  An
    optional (static) class variable is `default_parameters`, which is
    a dictionary indexed by the parameter names from
    `required_parameters`.  Any parameters referenced here will be
    given a default value.

    Dependence will check to make sure all of the required parameters
    have been supplied, with the exception of those which have default
    versions.  It also provides other convenience and safety features,
    such as allowing tests for equality of derived algorithms and for
    ensuring extra parameters were not assigned.
    """
    @staticmethod
    def _test(v):
        assert hasattr(v, "depname"), "Dependence needs a parameter name"
        assert v.depname in paranoid.types.String(), "depname must be a string"
        assert hasattr(v, "name"), "Dependence classes need a name"
        assert v.name in paranoid.types.String(), "name must be a string"
        assert hasattr(v, "required_parameters"), "Dependence needs a list of required params"
    @classmethod
    def _generate(cls):
        """Generate from subclasses.

        For each class which inherits Dependence, find the subclasses
        of that subclass, and generate from each of those (if the
        _generate function is available).
        """
        # Don't call directly as a Dependence object, it must be
        # inherited.
        if cls is Dependence:
            raise paranoid.NoGeneratorError("Cannot generate directly from Dependence objects")
        # Call the _generate methods of each subclass, e.g. call
        # DriftConstant._generate() if the _generate() function is called
        # from Drift (i.e. cls == Drift).
        subs = cls.__subclasses__()
        for s in subs:
            if hasattr(s, "_generate") and callable(s._generate):
                yield from s._generate()
    def __init__(self, **kwargs):
        """Create a new Dependence object with parameters specified in **kwargs.

        This function will only be called by classes which have been
        inherited from this one.  Errors here are caused by invalid
        subclass declarations.
        """
        # Check to make sure the subclass and subsubclass were implemented correctly
        assert hasattr(self, "depname"), "Dependence needs a parameter name"
        assert hasattr(self, "name"), "Dependence classes need a name"
        assert hasattr(self, "required_parameters"), "Dependence needs a list of required params"
        # Check/set parameters
        if hasattr(self, "default_parameters"):
            args = self.default_parameters.copy()
            args.update(kwargs)
        else:
            args = kwargs
        if not hasattr(self, "required_conditions"):
            object.__setattr__(self, 'required_conditions', [])
        passed_args = sorted(args.keys())
        expected_args = sorted(self.required_parameters)
        assert passed_args == expected_args, "Provided %s arguments, expected %s" % (str(passed_args), str(expected_args))
        for key, value in args.items():
            setattr(self, key, value)

    def __eq__(self, other):
        """Equality is defined as having the same algorithm type and the same parameters."""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __setattr__(self, name, val):
        """Only allow the required parameters to be assigned."""
        if name in self.required_parameters:
            return object.__setattr__(self, name, val) # No super() for python2 compatibility
        raise LookupError
    def __delattr__(self, name):
        """No not allow a required parameter to be deleted."""
        raise LookupError
    def __repr__(self):
        params = ""
        # If it is a sub-sub-class, then print the parameters it was
        # instantiated with
        if self.name:
            for p in self.required_parameters:
                params += str(p) + "=" + getattr(self, p).__repr__()
                if p != self.required_parameters[-1]:
                    params += ", "
        return type(self).__name__ + "(" + params + ")"
    def __str__(self):
        return self.__repr__()
    def __hash__(self):
        return hash(repr(self))


