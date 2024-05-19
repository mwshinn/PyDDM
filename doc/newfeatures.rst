New features in PyDDM
=====================

PyDDM has changed a lot since our `2020 eLife paper
<https://elifesciences.org/articles/56938>`_.  Here are some of the new
features:

Models are easier to define
---------------------------

Previously, defining a model required building custom classes or using one of
many special classes built in to PyDDM.  Now, you only need to define a function
with simple, consistent syntax.

For example, here is the **new way** to define a model where drift rate is the
product of evidence strength and a fittable parameter::

    import pyddm
    model = pyddm.gddm(drift=lambda mu,evidence : mu*evidence,
                       conditions=["evidence"],
                       parameters={"mu": (-4, 4)})

Compared to the **old way**::

    import pyddm
    class DriftEvidence(pyddm.models.Drift):
        name = "Drift depends linearly on evidence"
        required_parameters = ["mu"]
        required_conditions = ["evidence"]
        def get_drift(self, conditions, **kwargs):
            return self.mu * conditions['evidence']
    model = pyddm.Model(drift=DriftEvidence(mu=pyddm.Fittable(minval=-4, maxval=4)),
                        overlay=pyddm.OverlayUniformMixture(umixturecoef=.02))

:doc:`Learn more about defining models with the gddm function <quickstart>`

Simulations are faster
----------------------

Most simulations should be 10-100x faster than we reported in the eLife paper.
Some simulations will be up to 1000x faster, since PyDDM now autodetects more
situations where it can use the analytical solver.  (And the analytical solver
itself is also faster!)

Stimulus coding and accuracy coding
-----------------------------------

Previously, the top bound in PyDDM always meant "correct" and the bottom bound
always meant "error".  This so-called "accuracy coding" was baked into PyDDM.
Now, bounds can represent anything you like.  For instance, for the upper bound
to be a left choice and the bottom bound to be a right choice::

    sample = pyddm.Sample.from_pandas_dataframe(..., choice_names=("left", "right"))
    model = pyddm.gddm(..., choice_names=("left", "right"))

:ref:`Learn more about stimulus and accuracy coding <howto-stimulus-coding>`

Expanded capabilities of the Model GUI
--------------------------------------

Previously, the model GUI could be used to plot just the RT distribution.  Now,
it can also also plot the psychometric and chronometric functions, as well as
the bound shape.  It can also be expanded to other functions as well.
Additionally, the model GUI is now compatible with Jupyter notebooks and Google
Colab.

:doc:`Learn more about the model GUI <modelgui>`

Better error messages
---------------------

When things go wrong, PyDDM is better able to tell you what you need to fix.
