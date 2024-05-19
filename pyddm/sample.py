# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

import logging
import numpy as np
import itertools

from paranoid.types import NDArray, Number, List, String, Self, Positive, Positive0, Range, Natural0, Unchecked, Dict, Maybe, Nothing, Boolean, Or, Set, String, Tuple
from paranoid.decorators import *
from .models.paranoid_types import Conditions, Choice
from .logger import logger as _logger, deprecation_warning

@paranoidclass
class Sample(object):
    """Describes a sample from some (empirical or simulated) distribution.

    Similarly to Solution, this is a glorified container for three items: a
    list of reaction times for for the two choices (corresponding to upper and
    lower DDM boundaries), and the number of undecided trials.  Each can have
    different properties associated with it, known as "conditions" elsewhere in
    this codebase.  This is to specifiy the experimental parameters of the
    trial, to allow fitting of stimuli by (for example) color or intensity.

    To specify conditions, pass a keyword argument to the constructor.
    The name should be the name of the property, and the value should
    be a tuple of length two or three.  The first element of the tuple
    should be a list of length equal to the number of correct trials,
    and the second should be equal to the number of error trials.  If
    there are any undecided trials, the third argument should
    contain a list of length equal to `undecided`.

    By default, the choice associated with the upper boundary is "correct
    responses" and the lower boundary is "error responses".  To change these,
    set the `choice_names` argument to be a tuple containing two strings, with
    the names of the boundaries.  So the default is ("correct", "error"), but
    could be anything, e.g. ("left", "right"), ("high value" and "low value"),
    etc.  This is sometimes referred to as "accuracy coding" and "stimulus
    coding".  When fitting data, this must match the choice names of the model.

    Optionally, additional data can be associated with each
    independent data point.  These should be passed as keyword
    arguments, where the keyword name is the property and the value is
    a tuple.  The tuple should have either two or three elements: the
    first two should be lists of properties for the correct and error
    reaction times, where the properties correspond to reaction times
    in the correct or error lists.  Optionally, a third list of length
    equal to the number of undecided trials gives a list of conditions
    for these trials.  If multiple properties are passed as keyword
    arguments, the ordering of the undecided properties (in addition
    to those of the correct and error distributions) will correspond
    to one another.

    """
    @classmethod
    def _test(cls, v):
        # Most testing is done in the constructor and the data is read
        # only, so this isn't strictly necessary
        assert type(v) is cls
        assert v.choice_upper in NDArray(d=1, t=Positive0), "choice_upper not a numpy array with elements greater than 0, it is " + str(type(v.choice_upper))
        assert v.choice_lower in NDArray(d=1, t=Positive0), "choice_lower a numpy array with elements greater than 0, it is " + str(type(v.choice_lower))
        assert v.undecided in Natural0(), "undecided not a natural number"
        for k,val in v.conditions.items():
            # Make sure shape and type are correct
            assert k, "Invalid key"
            assert isinstance(val, tuple)
            assert len(val) in [2, 3]
            assert val[0] in NDArray(d=1)
            assert val[1] in NDArray(d=1)
            assert len(val[0]) == len(v.choice_upper)
            assert len(val[1]) == len(v.choice_lower)
            if len(val) == 3:
                assert len(val[2]) == v.undecided
                assert val[2] in NDArray(d=1)
            else:
                assert v.undecided == 0
    @staticmethod
    def _generate():
        aa = lambda x : np.asarray(x)
        yield Sample(aa([.1, .2, .3]), aa([.2, .3, .4]), undecided=0)
        yield Sample(aa([.1, .2, .3]), aa([]), undecided=0)
        yield Sample(aa([]), aa([.2, .3, .4]), undecided=0)
        yield Sample(aa([.1, .2, .3]), aa([.2, .3, .4]), undecided=5)
        
    def __init__(self, choice_upper, choice_lower, undecided=0, choice_names=("correct", "error"), **kwargs):
        assert choice_upper in NDArray(d=1, t=Number), "choice_upper not a numpy array, it is " + str(type(choice_upper))
        assert choice_lower in NDArray(d=1, t=Number), "choice_lower not a numpy array, it is " + str(type(choice_lower))
        assert undecided in Natural0(), "undecided not a natural number"
        # Note that in the original pyddm, choice names were always "correct"
        # or "error".  Now they can be anything, but some parts of the code may
        # still use the "corr" and "err" terminology.
        assert isinstance(choice_names, tuple) and len(choice_names) == 2, "choice_names must be a tuple of length 2"
        self.choice_names = choice_names
        self.choice_upper = choice_upper
        self.choice_lower = choice_lower
        self.undecided = undecided
        # Values should not change
        self.choice_upper.flags.writeable = False
        self.choice_lower.flags.writeable = False
        # Make sure the kwarg parameters/conditions are in the correct
        # format
        for k,v in kwargs.items():
            # Make sure shape and type are correct
            assert k, "Invalid key"
            assert isinstance(v, tuple)
            assert len(v) in [2, 3]
            assert v[0] in NDArray(d=1)
            assert v[1] in NDArray(d=1)
            assert len(v[0]) == len(self.choice_upper)
            assert len(v[1]) == len(self.choice_lower)
            # Make read-only
            v[0].flags.writeable = False
            v[1].flags.writeable = False
            if len(v) == 3:
                assert len(v[2]) == undecided
            else:
                assert undecided == 0
        self.conditions = kwargs
    def __len__(self):
        """The number of samples"""
        return len(self.choice_upper) + len(self.choice_lower) + self.undecided
    def __iter__(self):
        """Iterate through each reaction time, with no regard to whether it was a correct or error trial."""
        return np.concatenate([self.choice_upper, self.choice_lower]).__iter__()
    def __eq__(self, other):
        if len(self.choice_upper) != len(other.choice_upper) or \
           len(self.choice_lower) != len(other.choice_lower) or \
           self.undecided != other.undecided:
            return False
        if self.choice_names != other.choice_names:
            return False
        if not np.allclose(self.choice_upper, other.choice_upper) or \
           not np.allclose(self.choice_lower, other.choice_lower) or \
           self.undecided != other.undecided:
            return False
        for k in self.conditions:
            if k not in other.conditions:
                return False
            if np.issubdtype(self.conditions[k][0].dtype, np.floating) and \
               np.issubdtype(self.conditions[k][0].dtype, np.floating):
                compare_func = np.allclose
            else:
                compare_func = lambda x,y: np.all(x == y)
            if not compare_func(self.conditions[k][0], other.conditions[k][0]) or \
               not compare_func(self.conditions[k][1], other.conditions[k][1]):
                return False
            if len(self.conditions[k]) == 3 and \
               len(other.conditions[k]) == 3 and \
               not compare_func(self.conditions[k][2], other.conditions[k][2]):
                return False
        return True
    def __add__(self, other):
        assert sorted(self.conditions.keys()) == sorted(other.conditions.keys()), "Canot add with unlike conditions"
        assert self.choice_names == other.choice_names, "Cannot add samples with different choice names"
        choice_upper = np.concatenate([self.choice_upper, other.choice_upper])
        choice_lower = np.concatenate([self.choice_lower, other.choice_lower])
        undecided = self.undecided + other.undecided
        conditions = {}
        for k in self.conditions.keys():
            sc = self.conditions
            oc = other.conditions
            bothc = np.concatenate([sc[k][0], oc[k][0]])
            bothe = np.concatenate([sc[k][1], oc[k][1]])
            bothn = np.concatenate([sc[k][2] if len(sc[k]) == 3 else [],
                                    oc[k][2] if len(oc[k]) == 3 else []])
            conditions[k] = (bothc, bothe, bothn)
        return Sample(choice_upper, choice_lower, undecided, choice_names=self.choice_names, **conditions)
    @property
    def corr(self):
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("For backward compatibility with .corr only")
        deprecation_warning(instead="Sample.choice_upper", isfunction=False)
        return self.choice_upper
    @property
    def err(self):
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("For backward compatibility with .corr only")
        deprecation_warning(instead="Sample.choice_lower", isfunction=False)
        return self.choice_lower
    @staticmethod
    @accepts(NDArray(d=2), List(String), Tuple(String, String))
    @returns(Self)
    @requires('data.shape[1] >= 2')
    @requires('set(list(data[:,1])) - {0, 1} == set()')
    @requires('all(data[:,0].astype("float") == data[:,0])')
    @requires('data.shape[1] - 2 == len(column_names)')
    @ensures('len(column_names) == len(return.condition_names())')
    def from_numpy_array(data, column_names, choice_names=("correct", "error")):
        """Generate a Sample object from a numpy array.
        
        `data` should be an n x m array (n rows, m columns) where m>=2. The
        first column should be the response times, and the second column should
        be the choice that the trial corresponds to.  E.g., by default, the
        choices are (1 == correct, 0 == error), but these can be changed by
        passing in a tuple of strings to the `choice_names` variable.
        E.g. ("left", "right") means that (1 == left, 0 == right).

        Any remaining columns in `data` after the first two should be
        conditions.  `column_names` should be a list of length m of strings
        indicating the names of the conditions.  The order of the names should
        correspond to the order of the columns.  This function does not yet
        work with undecided trials.
        """
        c = data[:,1].astype(bool)
        nc = (1-data[:,1]).astype(bool)
        def pt(x): # Pythonic types
            arr = np.asarray(x, dtype=object)
            # The following is somewhat of a hack to get rid of object arrays
            # when a condition is not a number (e.g. string or tuple)
            if len(arr) > 0 and not isinstance(arr[0], (float, int, np.float_, np.int_)):
                return arr
            arr = np.asarray(arr.tolist())
            try:
                if np.all(arr == np.round(arr)):
                    arr = arr.astype(np.int64)
            except TypeError:
                pass
            return arr

        conditions = {k: (pt(data[c,i+2]), pt(data[nc,i+2]), np.asarray([])) for i,k in enumerate(column_names)}
        return Sample(pt(data[c,0]), pt(data[nc,0]), 0, **conditions)
    @staticmethod
    @accepts(Unchecked, String, Maybe(String), Unchecked, Maybe(String)) # TODO change unchecked to pandas
    @returns(Self)
    @requires('df.shape[1] >= 2')
    @requires('rt_column_name in df')
    @requires('choice_column_name in df or correct_column_name in df')
    @requires('not np.any(df.isnull())')
    @requires('len(np.setdiff1d(df[choice_column_name if choice_column_name is not None else correct_column_name], [0, 1])) == 0')
    @requires('all(df[rt_column_name].astype("float") == df[rt_column_name])')
    @ensures('len(df) == len(return)')
    def from_pandas_dataframe(df, rt_column_name, choice_column_name=None, choice_names=("correct", "error"), correct_column_name=None):
        """Generate a Sample object from a pandas dataframe.
        
        `df` should be a pandas array.  `rt_column_name` and
        `choice_column_name` should be strings, and `df` should contain columns
        by these names.

        The column with the name `rt_column_name` should be the response times,
        and the column with the name `choice_column_name` should be the choice
        that the trial corresponds to.  E.g., by default, the choices are (1 ==
        correct, 0 == error), but these can be changed by passing in a tuple of
        strings to the `choice_names` variable.  E.g. ("left", "right") means
        that (1 == left, 0 == right).

        Any remaining columns besides these two should be conditions.  This
        function does not yet work with undecided trials.

        `correct_column_name` is deprecated and included only for backward
        compatibility.

        """
        if len(df) == 0:
            _logger.warning("Empty DataFrame")
        if np.mean(df[rt_column_name]) > 50:
            _logger.warning("RTs should be specified in seconds, not milliseconds")
        for _,col in df.items():
            if len(df) > 0 and isinstance(col.iloc[0], (list, np.ndarray)):
                raise ValueError("Conditions should not be lists or ndarrays.  Please convert to a tuple instead.")
        if choice_column_name is None:
            assert correct_column_name is not None
            assert choice_names == ("correct", "error")
            choice_column_name = correct_column_name
            deprecation_warning("the choice_column_name argument")
        assert np.all(np.isin(df[choice_column_name], [0, 1, True, False])), "Choice must be specified as True/False or 0/1"
        c = df[choice_column_name].astype(bool)
        nc = (1-df[choice_column_name]).astype(bool)
        def pt(x): # Pythonic types
            arr = np.asarray(x, dtype=object)
            # The following is somewhat of a hack to get rid of object arrays
            # when a condition is not a number (e.g. string or tuple)
            if len(arr) > 0 and not isinstance(arr[0], (float, int, np.float_, np.int_)):
                return arr
            arr = np.asarray(arr.tolist())
            try:
                if np.all(arr == np.round(arr)):
                    arr = arr.astype(np.int64)
            except TypeError:
                pass
            return arr

        column_names = [e for e in df.columns if not e in [rt_column_name, choice_column_name]]
        conditions = {k: (pt(df[c][k]), pt(df[nc][k]), np.asarray([])) for k in column_names}
        return Sample(pt(df[c][rt_column_name]), pt(df[nc][rt_column_name]), 0, choice_names=choice_names, **conditions)
    def to_pandas_dataframe(self, rt_column_name='RT', choice_column_name='choice', drop_undecided=False, correct_column_name=None):
        """Convert the sample to a Pandas dataframe.

        `rt_column_name` is the column label for the response time, and
        `choice_column_name` is the column label for the choice (corresponding
        to the upper or lower boundary).

        Because undecided trials do not have an RT or choice, they are cannot
        be added to the data frame.  To ignore them, thereby creating a
        dataframe which is smaller than the sample, set `drop_undecided` to
        True.

        """
        if choice_column_name is None:
            assert correct_column_name == 'correct' or correct_column_name is None
            assert self.choice_names == ("correct", "error")
            choice_column_name = correct_column_name
        import pandas
        all_trials = []
        if self.undecided != 0 and drop_undecided is False:
            raise ValueError("The sample object has undecided trials.  These do not have an RT or a P(correct), so they cannot be converted to a data frame.  Please use the 'drop_undecided' flag when calling this function.")
        conditions = list(self.condition_names())
        columns = [choice_column_name, rt_column_name] + conditions
        for trial in self.items("_top"):
            all_trials.append([1, trial[0]] + [trial[1][c] for c in conditions])
        for trial in self.items("_bottom"):
            all_trials.append([0, trial[0]] + [trial[1][c] for c in conditions])
        return pandas.DataFrame(all_trials, columns=columns)
    def items(self, choice=None, correct=None):
        """Iterate through the reaction times.

        `choice` is whether to iterate through RTs corresponding to the upper
        or lower boundary, given as the name of the choice, e.g. "correct",
        "error", or the choice names specified in the model's choice_names
        parameter.

        `correct` is a deprecated parameter for backward compatibility, please
        use `choice` instead.

        For each iteration, a two-tuple is returned.  The first
        element is the reaction time, the second is a dictionary
        containing the conditions associated with that reaction time.

        If you just want the list of RTs, you can directly iterate
        through "sample.corr" and "sample.err".
        """
        if correct is not None:
            assert choice is None, "Either choice or correct argument must be None"
            assert self.choice_names == ("correct", "error")
            deprecation_warning(instead="Sample.items('correct') or Sample.items('error')")
            use_choice_upper = correct
        else:
            assert choice is not None, "Choice and correct arguments cannot both be None"
            use_choice_upper = (self._choice_name_to_id(choice) == 1)
        return _Sample_Iter_Wraper(self, use_choice_upper=use_choice_upper)
    @accepts(Self)
    @returns(Self)
    def subset(self, **kwargs):
        """Subset the data by filtering based on specified properties.

        Each keyword argument should be the name of a property.  These
        keyword arguments may have one of three values:

        - A list: For each element in the returned subset, the
          specified property is in this list of values.
        - A function: For each element in the returned subset, the
          specified property causes the function to evaluate to True.
        - Anything else: Each element in the returned subset must have
          this value for the specified property.

        Return a sample object representing the filtered sample.
        """
        mask_choice_upper = np.ones(len(self.choice_upper)).astype(bool)
        mask_choice_lower = np.ones(len(self.choice_lower)).astype(bool)
        mask_undec = np.ones(self.undecided).astype(bool)
        for k,v in kwargs.items():
            if hasattr(v, '__call__'):
                mask_choice_upper = np.logical_and(mask_choice_upper, [v(i) for i in self.conditions[k][0]])
                mask_choice_lower = np.logical_and(mask_choice_lower, [v(i) for i in  self.conditions[k][1]])
                mask_undec = np.asarray([], dtype=bool) if self.undecided == 0 else np.logical_and(mask_undec, [v(i) for i in self.conditions[k][2]])
            elif isinstance(v, (list, np.ndarray)):
                mask_choice_upper = np.logical_and(mask_choice_upper, [i in v for i in self.conditions[k][0]])
                mask_choice_lower = np.logical_and(mask_choice_lower, [i in v for i in self.conditions[k][1]])
                mask_undec = np.asarray([], dtype=bool) if self.undecided == 0 else np.logical_and(mask_undec, [i in v for i in self.conditions[k][2]])
            else:
                # Create a zero-dimensional array so this will work with tuples too
                val = np.array(None)
                val[()] = v
                mask_choice_upper = np.logical_and(mask_choice_upper, val == self.conditions[k][0])
                mask_choice_lower = np.logical_and(mask_choice_lower, val == self.conditions[k][1])
                mask_undec = np.asarray([], dtype=bool) if self.undecided == 0 else np.logical_and(mask_undec, val == self.conditions[k][2])
        for k,v in self.conditions.items():
            assert len(v[0]) == len(mask_choice_upper)
            assert len(v[1]) == len(mask_choice_lower)
            assert mask_choice_upper.dtype == bool
            if len(v) == 3:
                assert len(v[2]) == len(mask_undec)
            v[2][mask_undec] if len(v) == 3 else np.asarray([])
        filtered_conditions = {k : (v[0][mask_choice_upper.astype(bool)],
                                    v[1][mask_choice_lower.astype(bool)],
                                    (v[2][mask_undec.astype(bool)] if len(v) == 3 else np.asarray([])))
                               for k,v in self.conditions.items()}
        return Sample(self.choice_upper[mask_choice_upper],
                      self.choice_lower[mask_choice_lower],
                      sum(mask_undec),
                      choice_names=self.choice_names,
                      **filtered_conditions)
    @accepts(Self)
    @returns(List(String))
    def condition_names(self):
        """The names of conditions which hold some non-zero value in this sample."""
        return list(self.conditions.keys())
    @accepts(Self, String)
    @requires('cond in self.condition_names()')
    @returns(List(Unchecked))
    def condition_values(self, cond):
        """The values of a condition that have at least one element in the sample.

        `cond` is the name of the condition from which to get the
        observed values.  Returns a list of these values.
        """
        cs = self.conditions
        cvs = set(cs[cond][0]).union(set(cs[cond][1]))
        if len(cs[cond]) == 3:
            cvs = cvs.union(set(cs[cond][2]))
        return sorted(list(cvs))
        # Saved in case we later come across a bug with sets not working for mutable condition values
        # if len(cs[cond]) == 3:
        #     grouped = itertools.groupby(sorted(list(cs[cond][0])+list(cs[cond][1])+list(cs[cond][2])))
        # elif len(cs[cond]) == 2:
        #     grouped = itertools.groupby(sorted(list(cs[cond][0])+list(cs[cond][1])))
        # return [g for g,_ in grouped]
    @accepts(Self, Maybe(List(String)))
    @returns(List(Conditions))
    def condition_combinations(self, required_conditions=None):
        """Get all values for set conditions and return every combination of them.

        Since PDFs of solved models in general depend on all of the
        conditions, this returns a list of dictionaries.  The keys of
        each dictionary are the names of conditions, and the value is
        a particular value held by at least one element in the sample.
        Each list contains all possible combinations of condition values.

        If `required_conditions` is iterable, only the conditions with
        names found within `required_conditions` will be included.
        """
        cs = self.conditions
        conditions = []
        names = self.condition_names()
        if required_conditions is not None:
            names = [n for n in names if n in required_conditions]
        for c in names:
            undecided = cs[c][2] if len(cs[c]) == 3 else np.asarray([])
            joined = np.concatenate([cs[c][0], cs[c][1], undecided])
            conditions.append(joined)
        alljoined = list(zip(*conditions))
        # Saved in case we later come across a bug with sets not working for mutable condition values
        # combs = [g for g,_ in itertools.groupby(sorted(alljoined))]
        combs = list(set(alljoined))
        if len(combs) == 0: # Generally not needed since iterools.product does this
            return [{}]
        return [dict(zip(names, c)) for c in combs]

    @staticmethod
    @accepts(dt=Positive, T_dur=Positive)
    @returns(NDArray(d=1, t=Positive0))
    #@requires('T_dur/dt < 1e5') # Too large of a number
    def t_domain(dt=.01, T_dur=2):
        """The times that corresponds with pdf/cdf_corr/err parameters (their support)."""
        return np.linspace(0, T_dur, int(T_dur/dt)+1)

    @accepts(Self, Choice)
    @returns(Set([1, 2]))
    def _choice_name_to_id(self, choice):
        """Get an ID from the choice name.

        If the choice corresponds to the upper boundary, return 1.  If it
        corresponds to the lower boundary, return 2.  Otherwise, print an
        error.
        """
        # Do is this way in case someone names their choices "_bottom" and
        # "_top" in reverse.
        if choice in [1, self.choice_names[0]]:
            return 1
        if choice in [0, 2, self.choice_names[1]]:
            return 2
        if choice in ["_top", "top", "top_bound", "upper_bound", "upper"]:
            return 1
        if choice in ["_bottom", "bottom_bound", "lower_bound", "lower", "bottom"]:
            return 2
        raise NotImplementedError("\"choice\" needs to be '"+self.choice_names[0]+"' or '"+self.choice_names[1]+"' to use this function, not '"+choice+"'")

    @accepts(Self, Choice, dt=Positive, T_dur=Positive)
    @returns(NDArray(d=1, t=Positive0))
    #@requires('T_dur/dt < 1e5') # Too large of a number
    @ensures('len(return) == len(self.t_domain(dt=dt, T_dur=T_dur))')
    def pdf(self, choice, dt=.01, T_dur=2):
        """An estimate of the probability density function of sample RTs for a given choice.

        `choice` should be the name of the choice for which to obtain the pdf,
        corresponding to the upper or lower boundary crossings.  E.g.,
        "correct", "error", or the choice names specified in the model's
        choice_names parameter.

        Note that the return value will not sum to one, but both choices plus
        the undecided distribution will collectively sum to one.
        """
        v = self.choice_upper if self._choice_name_to_id(choice) == 1 else self.choice_lower
        return np.histogram(v, bins=int(T_dur/dt)+1, range=(0-dt/2, T_dur+dt/2))[0]/len(self)/dt # dt/2 terms are for continuity correction


    @accepts(Self, dt=Positive, T_dur=Positive)
    @returns(NDArray(d=1, t=Positive0))
    #@requires('T_dur/dt < 1e5') # Too large of a number
    @ensures('len(return) == len(self.t_domain(dt=dt, T_dur=T_dur))')
    def pdf_corr(self, dt=.01, T_dur=2):
        """The correct component of the joint PDF.

        This method is deprecated, use Sample.pdf() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"pdf\" instead.")
        deprecation_warning(instead="Sample.pdf('correct')")
        return np.histogram(self.corr, bins=int(T_dur/dt)+1, range=(0-dt/2, T_dur+dt/2))[0]/len(self)/dt # dt/2 terms are for continuity correction

    @accepts(Self, dt=Positive, T_dur=Positive)
    @returns(NDArray(d=1, t=Positive0))
    #@requires('T_dur/dt < 1e5') # Too large of a number
    @ensures('len(return) == len(self.t_domain(dt=dt, T_dur=T_dur))')
    def pdf_err(self, dt=.01, T_dur=2):
        """The error (incorrect) component of the joint PDF.

        This method is deprecated, use Sample.pdf() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"pdf\" instead.")
        deprecation_warning(instead="Sample.pdf('error')")
        return np.histogram(self.err, bins=int(T_dur/dt)+1, range=(0-dt/2, T_dur+dt/2))[0]/len(self)/dt # dt/2 terms are for continuity correction

    @accepts(Self, Choice, dt=Positive, T_dur=Positive)
    @returns(NDArray(d=1, t=Positive0))
    #@requires('T_dur/dt < 1e5') # Too large of a number
    @ensures('len(return) == len(self.t_domain(dt=dt, T_dur=T_dur))')
    def cdf(self, choice, dt=.01, T_dur=2):
        """An estimate of the cumulative density function of sample RTs for a given choice.

        `choice` should be the name of the choice for which to obtain the cdf,
        corresponding to the upper or lower boundary crossings.  E.g.,
        "correct", "error", or the choice names specified in the model's
        choice_names parameter.


        Note that the return value will not converge to one, but both choices plus
        the undecided distribution will collectively converge to one.

        """
        return np.cumsum(self.pdf(choice, dt=dt, T_dur=T_dur))*dt

    @accepts(Self, dt=Positive, T_dur=Positive)
    @returns(NDArray(d=1, t=Positive0))
    #@requires('T_dur/dt < 1e5') # Too large of a number
    @ensures('len(return) == len(self.t_domain(dt=dt, T_dur=T_dur))')
    def cdf_corr(self, dt=.01, T_dur=2):
        """The correct component of the joint CDF.

        This method is deprecated, use Sample.cdf() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"cdf\" instead.")
        deprecation_warning(instead="Sample.cdf('correct')")
        return np.cumsum(self.pdf_corr(dt=dt, T_dur=T_dur))*dt

    @accepts(Self, dt=Positive, T_dur=Positive)
    @returns(NDArray(d=1, t=Positive0))
    @ensures('len(return) == len(self.t_domain(dt=dt, T_dur=T_dur))')
    def cdf_err(self, dt=.01, T_dur=2):
        """The error (incorrect) component of the joint CDF.

        This method is deprecated, use Sample.cdf() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"cdf\" instead.")
        deprecation_warning(instead="Sample.cdf('error')")
        return np.cumsum(self.pdf_err(dt=dt, T_dur=T_dur))*dt

    @accepts(Self, Choice)
    @returns(Range(0, 1))
    @requires("len(self) > 0")
    def prob(self, choice):
        """Probability of a given choice response.

        `choice` should be the name of the choice for which to obtain the
        probability, corresponding to the upper or lower boundary crossings.
        E.g., "correct", "error", or the choice names specified in the model's
        """
        v = self.choice_upper if self._choice_name_to_id(choice) == 1 else self.choice_lower
        return len(v)/len(self)

    @accepts(Self)
    @returns(Range(0, 1))
    @requires("len(self) > 0")
    def prob_correct(self):
        """The probability of selecting the right response.

        This method is deprecated, use Sample.prob() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"prob\" instead.")
        deprecation_warning(instead="Sample.prob('correct')")
        return len(self.corr)/len(self)

    @accepts(Self)
    @returns(Range(0, 1))
    @requires("len(self) > 0")
    def prob_error(self):
        """The probability of selecting the incorrect (error) response.

        This method is deprecated, use Sample.prob() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"prob\" instead.")
        deprecation_warning(instead="Sample.prob('error')")
        return len(self.err)/len(self)

    @accepts(Self)
    @returns(Range(0, 1))
    @requires("len(self) > 0")
    def prob_undecided(self):
        """The probability of selecting neither response (undecided)."""
        return self.undecided/len(self)

    @accepts(Self, Choice)
    @returns(Range(0, 1))
    @requires("len(self) > 0")
    def prob_forced(self, choice):
        """Probability of a given response if a response is forced.

        `choice` should be the name of the choice for which to obtain the
        probability, corresponding to the upper or lower boundary crossings.
        E.g., "correct", "error", or the choice names specified in the model's

        If a trajectory is undecided, then a response is selected randomly.
        """
        return self.prob(choice) + self.prob_undecided()/2.

    @accepts(Self)
    @returns(Range(0, 1))
    @requires("len(self) > 0")
    def prob_correct_forced(self):
        """The probability of selecting the correct response if a response is forced.

        This method is deprecated, use Sample.prob_forced() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"prob_forced\" instead.")
        deprecation_warning(instead="Sample.prob_forced('correct')")
        return self.prob_correct() + self.prob_undecided()/2.

    @accepts(Self)
    @returns(Range(0, 1))
    @requires("len(self) > 0")
    def prob_error_forced(self):
        """The probability of selecting the incorrect response if a response is forced.

        This method is deprecated, use Sample.prob_forced() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"prob_forced\" instead.")
        deprecation_warning(instead="Sample.prob_forced('error')")
        return self.prob_error() + self.prob_undecided()/2.

    @accepts(Self)
    @requires("len(self.choice_upper) > 0")
    @returns(Positive0)
    def mean_decision_time(self):
        """The mean decision time in the correct trials."""
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  See the mean_rt method.")
        return np.mean(self.choice_upper)

    @accepts(Self)
    @requires("len(self.choice_upper)+len(self.choice_lower) > 0")
    @returns(Positive0)
    def mean_rt(self):
        """The mean decision time in the correct trials."""
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.")
        return np.mean(np.concatenate([self.choice_upper, self.choice_lower]))

class _Sample_Iter_Wraper(object):
    """Provide an iterator for sample objects.

    `sample_obj` is the Sample which we plan to iterate.  `use_choice_upper` should
    be either True (to iterate through upper boundary responses) or False (to
    iterate through lower boundary responses).

    Each step of the iteration returns a two-tuple, where the first
    element is the reaction time, and the second element is a
    dictionary of conditions.
    """
    def __init__(self, sample_obj, use_choice_upper):
        self.sample = sample_obj
        self.i = 0
        self.use_choice_upper = use_choice_upper
        if self.use_choice_upper:
            self.rt = self.sample.choice_upper
            self.ind = 0
        elif not self.use_choice_upper:
            self.rt = self.sample.choice_lower
            self.ind = 1
    def __iter__(self):
        return self
    def __next__(self):
        if self.i == len(self.rt):
            raise StopIteration
        self.i += 1
        return (self.rt[self.i-1], {k : self.sample.conditions[k][self.ind][self.i-1] for k in self.sample.conditions.keys()})
        
