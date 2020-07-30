Shinn et al. (2020) - Confluence of timing and reward biases in perceptual decision-making dynamics
===================================================================================================

`Read the paper <https://www.biorxiv.org/content/10.1101/865501v1>`_

:download:`Download all code referenced here (shinn2020.py) <../../downloads/shinn2020.py>`

`Run this model online interactively <https://colab.research.google.com/github/mwshinn/PyDDM/blob/master/doc/notebooks/shinn2020.ipynb>`_

Model definition
----------------

Several combinations of models were compared and tested.  These are
based on the following drift and noise classes.  Note that all
function decorators and `_test` methods are Paranoid Scientist
annotations to check for model correctness, and can be removed without
a change in functionality.

.. literalinclude:: ../../downloads/shinn2020.py
   :language: python
   :start-after: # BEGIN driftnoise
   :end-before: # END driftnoise


These utilize the following utility functions:

.. literalinclude:: ../../downloads/shinn2020.py
   :language: python
   :start-after: # BEGIN utility_functions
   :end-before: # END utility_functions

Models also utilize the following reward and timing functions

.. literalinclude:: ../../downloads/shinn2020.py
   :language: python
   :start-after: # BEGIN rewardtiming
   :end-before: # END rewardtiming

The above are for the color matching task with reward bias.  The
versions for the color matching task with timing blocks are identical,
except have two copies of each parameter, one for each block.

Interactive demo
----------------

Here is a demonstration of these using the model GUI.  To run this,
set the "URGENCY" variable to be either "collapse" or "gain":

.. literalinclude:: ../../downloads/shinn2020.py
   :language: python
   :start-after: # BEGIN demo
   :end-before: # END demo
   :dedent: 8
