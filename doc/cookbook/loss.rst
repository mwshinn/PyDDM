Loss functions
==============

Fitting with an alternative loss function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fitting methods can be modified by changing the loss function or by
changing the algorithm used to optimize the loss function.  The
default loss function is likelihood (via
:class:`~.models.loss.LossLikelihood`).  Squared error (via
:class:`~.models.loss.LossSquaredError`) and BIC (via
:class:`~.models.loss.LossBIC`) are also available.

As an example, to fit the "Simple example" from the quickstart guide,
do::

  fit_adjust_model(samp, model_fit,
                   method="differential_evolution",
                   lossfunction=LossSquaredError)

Custom loss functions may be defined by extending
:class:`~.models.loss.LossFunction`.  The ``loss`` function must be
defined and, given a model, returns the goodness of fit to the sample,
accessible under ``self.sample``. The ``setup`` function may
optionally be defined, which is run once and may be used, for example,
to do computations on the sample don't need to be run at each
evaluation.

.. _loss-means:

Fitting using mean RT and P(correct)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is an example of how to define a loss function as the sum of two
terms: the mean response time of correct trials, and the probability
of choosing the correct target.  While unprincipled, it is a simple
example that in practice gives similar fits to the analytical
expression in Roitman and Shadlen (2002):

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start LossByMeans
   :end-before: # End LossByMeans


.. _loss-undecided:

Fitting with undecided trials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to correct and incorrect trials, some trials may go beyond
the time allowed for a decision.  The effect of these trials is
usually minor due to the design of task paradigms, but PyDDM is
capable of using these values within its fitting procedures.

Currently, the functions which import Sample objects from numpy arrays
do not support undecided trials; thus, to include undecided trials in
a sample, they must be passed directly to the Sample constructor in a
more complicated form.

To construct a sample with undecided trials, first create a Numpy
array of correct RTs and incorrect RTs in units of seconds, and count
the number of undecided trials.  Then, for each task conditions,
create a tuple containing three elements.  The first element should be
a Numpy array with the task condition value for each associated
correct RT, the second should be the same but for error trials, and
the final element should be a Numpy array in no particular order with
a number of elements equal to the undecided trials, with one
corresponding to each undecided trial.

Consider the following example with "reward" as the task condition. We
suppose there is one correct trial with a reward of 3 and an RT of
0.3s, one error with a reward of 2 and an RT of 0.5s, and two
undecided trials with rewards of 1 and 2::

  sample = Sample(np.asarray([0.3]), np.asarray([0.5]), 2,
                  reward=(np.asarray([3]), np.asarray([2]), np.asarray([1, 2])))
                                                                   
A sample created using this method can be used the same way as one
created using :meth:`~.Sample.from_numpy_array` or
:meth:`~.Sample.from_pandas_dataframe`.

