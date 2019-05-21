# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

import numpy as np
import itertools

from paranoid.types import NDArray, Number, List, String, Self, Positive, Positive0, Range, Natural0, Unchecked, Dict, Or, Nothing
from paranoid.decorators import *
from .models.paranoid_types import Conditions

@paranoidclass
class Sample(object):
    """Describes a sample from some (empirical or simulated) distribution.

    Similarly to Solution, this is a glorified container for three
    items: a list of correct reaction times, a list of error reaction
    times, and the number of undecided trials.  Each can have
    different properties associated with it, known as "conditions"
    elsewhere in this codebase.  This is to specifiy the experimental
    parameters of the trial, to allow fitting of stimuli by (for
    example) color or intensity.

    To specify conditions, pass a keyword argument to the constructor.
    The name should be the name of the property, and the value should
    be a tuple of length two or three.  The first element of the tuple
    should be a list of length equal to the number of correct trials,
    and the second should be equal to the number of error trials.  If
    there are any undecided trials, the third argument should
    contain a list of length equal to `undecided`.

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
    @staticmethod
    def _generate():
        aa = lambda x : np.asarray(x)
        yield Sample(aa([.1, .2, .3]), aa([.2, .3, .4]), undecided=0)
        yield Sample(aa([.1, .2, .3]), aa([]), undecided=0)
        yield Sample(aa([]), aa([.2, .3, .4]), undecided=0)
        yield Sample(aa([.1, .2, .3]), aa([.2, .3, .4]), undecided=5)
        
    def __init__(self, sample_corr, sample_err, undecided=0, **kwargs):
        assert sample_corr in NDArray(d=1, t=Number), "sample_corr not a numpy array, it is " + str(type(sample_corr))
        assert sample_err in NDArray(d=1, t=Number), "sample_err not a numpy array, it is " + str(type(sample_err))
        assert undecided in Natural0(), "undecided not a natural number"
        self.corr = sample_corr
        self.err = sample_err
        self.undecided = undecided
        # Values should not change
        self.corr.flags.writeable = False
        self.err.flags.writeable = False
        # Make sure the kwarg parameters/conditions are in the correct
        # format
        for _,v in kwargs.items():
            # Make sure shape and type are correct
            assert isinstance(v, tuple)
            assert len(v) in [2, 3]
            assert v[0] in NDArray(d=1)
            assert v[1] in NDArray(d=1)
            assert len(v[0]) == len(self.corr)
            assert len(v[1]) == len(self.err)
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
        return len(self.corr) + len(self.err) + self.undecided
    def __iter__(self):
        """Iterate through each reaction time, with no regard to whether it was a correct or error trial."""
        return np.concatenate([self.corr, self.err]).__iter__()
    def __eq__(self, other):
        if not np.allclose(self.corr, other.corr) or \
           not np.allclose(self.err, other.err) or \
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
        corr = np.concatenate([self.corr, other.corr])
        err = np.concatenate([self.err, other.err])
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
        return Sample(corr, err, undecided, **conditions)
    @staticmethod
    @accepts(NDArray(d=2, t=Number), List(String))
    @returns(Self)
    @requires('data.shape[1] >= 2')
    @requires('set(list(data[:,1])) - {0, 1} == set()')
    @requires('all(data[:,0].astype("float") == data[:,0])')
    @requires('data.shape[1] - 2 == len(column_names)')
    @ensures('len(column_names) == len(return.condition_names())')
    def from_numpy_array(data, column_names):
        """Generate a Sample object from a numpy array.
        
        `data` should be an n x m array (n rows, m columns) where
        m>=2. The first column should be the response times, and the
        second column should be whether the trial was correct or an
        error (1 == correct, 0 == error).  Any remaining columns
        should be conditions.  `column_names` should be a list of
        length m of strings indicating the names of the conditions.
        The order of the names should correspond to the order of the
        columns.  This function does not yet work with undecided
        trials.
        """
        c = data[:,1].astype(bool)
        nc = (1-data[:,1]).astype(bool)
        def pt(x): # Pythonic types
            arr = np.asarray(x)
            if np.all(arr == np.round(arr)):
                arr = arr.astype(int)
            return arr

        conditions = {k: (pt(data[c,i+2]), pt(data[nc,i+2]), []) for i,k in enumerate(column_names)}
        return Sample(pt(data[c,0]), pt(data[nc,0]), 0, **conditions)
    @staticmethod
    @accepts(Unchecked, String, String) # TODO change unchecked to pandas
    @returns(Self)
    @requires('df.shape[1] >= 2')
    @requires('rt_column_name in df')
    @requires('correct_column_name in df')
    @requires('len(np.setdiff1d(df[correct_column_name], [0, 1])) == 0')
    @requires('all(df[rt_column_name].astype("float") == df[rt_column_name])')
    @ensures('len(df) == len(return)')
    def from_pandas_dataframe(df, rt_column_name, correct_column_name):
        """Generate a Sample object from a pandas dataframe.
        
        `df` should be a pandas array.  `rt_column_name` and
        `correct_column_name` should be strings, and `df` should
        contain columns by these names. The column with the name
        `rt_column_name` should be the response times, and the column
        with the name `correct_column_name` should be whether the
        trial was correct or an error (1 == correct, 0 == error).  Any
        remaining columns should be conditions.  This function does
        not yet work with undecided trials.
        """
        c = df[correct_column_name].astype(bool)
        nc = (1-df[correct_column_name]).astype(bool)
        def pt(x): # Pythonic types
            arr = np.asarray(x)
            if np.all(arr == np.round(arr)):
                arr = arr.astype(int)
            return arr

        column_names = [e for e in df.columns if not e in [rt_column_name, correct_column_name]]
        conditions = {k: (pt(df[c][k]), pt(df[nc][k]), []) for k in column_names}
        return Sample(pt(df[c][rt_column_name]), pt(df[nc][rt_column_name]), 0, **conditions)
    def items(self, correct):
        """Iterate through the reaction times.

        This takes only one argument: a boolean `correct`, true if we
        want to iterate through the correct trials, and false if we
        want to iterate through the error trials.  

        For each iteration, a two-tuple is returned.  The first
        element is the reaction time, the second is a dictionary
        containing the conditions associated with that reaction time.
        """
        return _Sample_Iter_Wraper(self, correct=correct)
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
        mask_corr = np.ones(len(self.corr)).astype(bool)
        mask_err = np.ones(len(self.err)).astype(bool)
        mask_undec = np.ones(self.undecided).astype(bool)
        for k,v in kwargs.items():
            if hasattr(v, '__call__'):
                mask_corr = np.logical_and(mask_corr, [v(i) for i in self.conditions[k][0]])
                mask_err = np.logical_and(mask_err, [v(i) for i in  self.conditions[k][1]])
                mask_undec = [] if self.undecided == 0 else np.logical_and(mask_undec, [v(i) for i in self.conditions[k][2]])
            elif hasattr(v, '__contains__'):
                mask_corr = np.logical_and(mask_corr, [i in v for i in self.conditions[k][0]])
                mask_err = np.logical_and(mask_err, [i in v for i in self.conditions[k][1]])
                mask_undec = [] if self.undecided == 0 else np.logical_and(mask_undec, [i in v for i in self.conditions[k][2]])
            else:
                mask_corr = np.logical_and(mask_corr, [i == v for i in self.conditions[k][0]])
                mask_err = np.logical_and(mask_err, [i == v for i in self.conditions[k][1]])
                mask_undec = [] if self.undecided == 0 else np.logical_and(mask_undec, [i == v for i in self.conditions[k][2]])
        filtered_conditions = {k : (np.asarray(list(itertools.compress(v[0], mask_corr))),
                                    np.asarray(list(itertools.compress(v[1], mask_err))),
                                    (np.asarray(list(itertools.compress(v[2], mask_undec))) if len(v) == 3 else np.asarray([])))
                               for k,v in self.conditions.items()}
        return Sample(np.asarray(list(itertools.compress(self.corr, list(mask_corr)))),
                      np.asarray(list(itertools.compress(self.err, list(mask_err)))),
                      sum(mask_undec),
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
    @accepts(Self, Or(Nothing, List(String)))
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
            conditions.append(list(set(cs[c][0]).union(set(cs[c][1]))))
        combs = []
        for p in itertools.product(*conditions):
            if len(self.subset(**dict(zip(names, p)))) != 0:
                combs.append(dict(zip(names, p)))
        if len(combs) == 0: # Generally not needed since iterools.product does this
            return [{}]
        return combs

    @staticmethod
    @accepts(dt=Positive, T_dur=Positive)
    @returns(NDArray(d=1, t=Positive0))
    #@requires('T_dur/dt < 1e5') # Too large of a number
    def t_domain(dt=.01, T_dur=2):
        """The times that corresponds with pdf/cdf_corr/err parameters (their support)."""
        return np.linspace(0, T_dur, int(T_dur/dt)+1)

    @accepts(Self, dt=Positive, T_dur=Positive)
    @returns(NDArray(d=1, t=Positive0))
    #@requires('T_dur/dt < 1e5') # Too large of a number
    @ensures('len(return) == len(self.t_domain(dt=dt, T_dur=T_dur))')
    def pdf_corr(self, dt=.01, T_dur=2):
        """The correct component of the joint PDF."""
        return np.histogram(self.corr, bins=int(T_dur/dt)+1, range=(0-dt/2, T_dur+dt/2))[0]/len(self)/dt # dt/2 terms are for continuity correction

    @accepts(Self, dt=Positive, T_dur=Positive)
    @returns(NDArray(d=1, t=Positive0))
    #@requires('T_dur/dt < 1e5') # Too large of a number
    @ensures('len(return) == len(self.t_domain(dt=dt, T_dur=T_dur))')
    def pdf_err(self, dt=.01, T_dur=2):
        """The error (incorrect) component of the joint PDF."""
        return np.histogram(self.err, bins=int(T_dur/dt)+1, range=(0-dt/2, T_dur+dt/2))[0]/len(self)/dt # dt/2 terms are for continuity correction

    @accepts(Self, dt=Positive, T_dur=Positive)
    @returns(NDArray(d=1, t=Positive0))
    #@requires('T_dur/dt < 1e5') # Too large of a number
    @ensures('len(return) == len(self.t_domain(dt=dt, T_dur=T_dur))')
    def cdf_corr(self, dt=.01, T_dur=2):
        """The correct component of the joint CDF."""
        return np.cumsum(self.pdf_corr(dt=dt, T_dur=T_dur))*dt

    @accepts(Self, dt=Positive, T_dur=Positive)
    @returns(NDArray(d=1, t=Positive0))
    @ensures('len(return) == len(self.t_domain(dt=dt, T_dur=T_dur))')
    def cdf_err(self, dt=.01, T_dur=2):
        """The error (incorrect) component of the joint CDF."""
        return np.cumsum(self.pdf_err(dt=dt, T_dur=T_dur))*dt

    @accepts(Self)
    @returns(Range(0, 1))
    @requires("len(self) > 0")
    def prob_correct(self):
        """The probability of selecting the right response."""
        return len(self.corr)/len(self)

    @accepts(Self)
    @returns(Range(0, 1))
    @requires("len(self) > 0")
    def prob_error(self):
        """The probability of selecting the incorrect (error) response."""
        return len(self.err)/len(self)

    @accepts(Self)
    @returns(Range(0, 1))
    @requires("len(self) > 0")
    def prob_undecided(self):
        """The probability of selecting neither response (undecided)."""
        return self.undecided/len(self)

    @accepts(Self)
    @returns(Range(0, 1))
    @requires("len(self) > 0")
    def prob_correct_forced(self):
        """The probability of selecting the correct response if a response is forced."""
        return self.prob_correct() + self.prob_undecided()/2.

    @accepts(Self)
    @returns(Range(0, 1))
    @requires("len(self) > 0")
    def prob_error_forced(self):
        """The probability of selecting the incorrect response if a response is forced."""
        return self.prob_error() + self.prob_undecided()/2.

class _Sample_Iter_Wraper(object):
    """Provide an iterator for sample objects.

    `sample_obj` is the Sample which we plan to iterate.  `correct`
    should be either True (to iterate through correct responses) or
    False (to iterate through error responses).

    Each step of the iteration returns a two-tuple, where the first
    element is the reaction time, and the second element is a
    dictionary of conditions.
    """
    def __init__(self, sample_obj, correct):
        self.sample = sample_obj
        self.i = 0
        self.correct = correct
    def __iter__(self):
        return self
    def __next__(self):
        if self.i == len(self.sample):
            raise StopIteration
        self.i += 1
        if self.correct:
            rt = self.sample.corr
            ind = 0
        elif not self.correct:
            rt = self.sample.err
            ind = 1
        return (rt[self.i-1], {k : self.sample.conditions[k][ind][self.i-1] for k in self.sample.conditions.keys()})
        
