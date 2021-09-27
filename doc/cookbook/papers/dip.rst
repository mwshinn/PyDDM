Shinn et al. (2021) - Transient neuronal suppression for exploitation  of new sensory evidence
===================================================================================================

`Read the paper <https://www.biorxiv.org/content/10.1101/2020.11.29.403089v1>`_

:download:`Download all code referenced here (shinn2021.py) <../../downloads/shinn2021.py>`

`Run this model online interactively <https://colab.research.google.com/github/mwshinn/PyDDM/blob/master/doc/notebooks/shinn2021.ipynb>`_

Model definition
~~~~~~~~~~~~~~~~

Let's import the libraries we'll need:

.. literalinclude:: ../../downloads/shinn2021.py
   :language: python
   :start-after: # BEGIN import
   :end-before: # END import

Included below are the three GDDMs included in the paper.  For convenience of
implementation, each is given a "`diptype`" code: the "pause model" is
`diptype=1`, the "reset model" is `diptype=2`, and the "motor suppression model"
is `diptype=3`.  For no dip, we use the convention `diptype=-1`.

First, we define several helper functions which we will use throughout in the
model:

.. literalinclude:: ../../downloads/shinn2021.py
   :language: python
   :start-after: # BEGIN functions
   :end-before: # END functions

Now we define the drift rate:

.. literalinclude:: ../../downloads/shinn2021.py
   :language: python
   :start-after: # BEGIN drift
   :end-before: # END drift

And the noise, which corresponds to the drift rate we just defined:

.. literalinclude:: ../../downloads/shinn2021.py
   :language: python
   :start-after: # BEGIN noise
   :end-before: # END noise

And the starting position:

.. literalinclude:: ../../downloads/shinn2021.py
   :language: python
   :start-after: # BEGIN ic
   :end-before: # END ic

For motor suppression, we use an increasing bound as an equivalent formulation
of a motor decision variable.  It increases smoothly (according to a Beta(3,3)
function) to avoid numerical transients:

.. literalinclude:: ../../downloads/shinn2021.py
   :language: python
   :start-after: # BEGIN bound
   :end-before: # END bound

Finally, we have an overlay which simulates the detection probability, i.e. the
fraction of trials on which the dip is actually exhibited.  This overlay
achieves this by eliminating the dip mechanism (setting `diptype=-1`),
re-simulating the model, and then blending the resulting histogram with the
actual model's simulation with the given ratio.  This also implements a
trial-by-trial method, whereby the choice between the two models is
probabilistic:

.. literalinclude:: ../../downloads/shinn2021.py
   :language: python
   :start-after: # BEGIN overlay
   :end-before: # END overlay

Running the model
~~~~~~~~~~~~~~~~~

Now that we have defined all of the pieces, let's test the model with the GUI:

.. literalinclude:: ../../downloads/shinn2021.py
   :language: python
   :start-after: # BEGIN demo
   :end-before: # END demo

Optionally, we can run the model in parallel with 4 CPUs using:

  ddm.set_N_cpus(4)

Finally, plot the model in a GUI interface::

  ddm.plot.model_gui(model=m, conditions={"coherence": [50, 53, 60, 70],
                                          "presample": [0, 400, 800],
                                          "highreward": [0, 1]})

Or, if running a Jupyter notebook::

  ddm.plot.model_gui_jupyter(model=m, conditions={"coherence": [50, 53, 60, 70],
                                                  "presample": [0, 400, 800],
                                                  "highreward": [0, 1]})

Voila, look at that!  It's a dip!
