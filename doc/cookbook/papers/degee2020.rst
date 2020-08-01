De Gee et al (2020) - Pupil-linked phasic arousal predicts a reduction of choice bias across species and decision domains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Read the paper <https://elifesciences.org/articles/54014>`_

:download:`Download all code referenced here (degee2020.py) <../../downloads/degee2020.py>`

Here, we would like to fit models which incorporate an external
signal, such as pupil dilation, into biases on drift rate, starting
position, or urgency signal.  Pupil size can be broken up into an
arbitrary number of bins, leading to an arbitrary granularity.  Thus,
models must be constructed based on an arbitrary granularity, which
means that they must accept an arbitrary number of parameters,
depending on the given granularity (number of bins).  To do this, we
create "factory" functions which create Drift, Noise, Bound, and
Overlay (for non-decision time) objects for a given number of bins.

Note that the following is "stimulus coded", meaning that the
"correct" column actually codes for the choices subjects made, and an
additional column "stimulus" is expected that code for which stimulus
was presented.

First, we need a function to get the names of the parameters which we
will pass to the model.  The number of these parameters depends on the
granularity.

.. literalinclude:: ../../downloads/degee2020.py
   :language: python
   :start-after: # Start get_param_names
   :end-before: # End get_param_names

Now, we create functions which generate the relevant Drift, Noise,
Bound, and Overlay objects.

.. literalinclude:: ../../downloads/degee2020.py
   :language: python
   :start-after: # Start maker_functions
   :end-before: # End maker_functions

Finally, we stitch these together into one single model.

.. literalinclude:: ../../downloads/degee2020.py
   :language: python
   :start-after: # Start make_model
   :end-before: # End make_model
