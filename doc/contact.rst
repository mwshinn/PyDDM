Contact
-------

Before you ask for help
~~~~~~~~~~~~~~~~~~~~~~~

- **Make sure you have read the quickstart guide.** :doc:`Most common questions
  are answered in the quickstart guide. <quickstart>`  Many other questions are
  answered in the :doc:`FAQs <faqs>`.
- **If you used ChatGPT instead of reading the quickstart guide, please do not
  ask for help.** Instead, read the :doc:`quickstart guide <quickstart>`.  LLMs
  write PyDDM code in a way that is difficult to understand and debug.  For
  instance, ChatGPT often spits out 50 line code blocks for very simple models
  that should only be 3-4 lines of code.


How should I ask for help?
~~~~~~~~~~~~~~~~~~~~~~~~~~

The most important thing to report is a "minimal reproducible example".  This is
code which reproduces the problem with as little code as possible.  Your minimal
example must:

- **Include all code necessary to reproduce the behaviour.** Someone else should
  be able to copy and paste your code and it should run *without further
  modifications*.  This includes the relevant import statements.
- **Be as minimal as possible.** Please remove every aspect of the code and the
  model which does not cause the error.  For example, if you have a drift
  function and a collapsing boundary function in your model, but the error
  persists when you set a constant boundary, then simplify the model by setting
  a constant boundary in your minimal example.  Likewise, remove as many of the
  conditions and parameters from your model as possible.  For example, if you
  have a complicated drift function, simplify the function by removing as many
  terms as possible, and remove as many parameters and conditions as you can from
  your model by setting them to constants.  You want the SMALLEST and SIMPLEST
  piece of code possible that will still cause the error.  If your model is made
  any smaller or simpler, the error should disappear.
- **Do not include model fitting.** Model fitting simply runs your model with
  many different combinations of parameters and evaluates the loss function on
  the result.  Please find the specific parameters which cause the issue and
  call the model with ``model.solve(conditions=...)`` in your minimal example.
- **Do not rely on external data files (if possible).** If you require data for
  your minimal example (most should not if you don't include fitting), construct
  a Sample object by hand using a pandas dataframe in your minimal example,
  including as few data points as possible.  If a Sample object does not
  reproduce the error and you must include a data file, minimise the data file
  by removing as many of the data points as possible.  An ideal minimal data
  file should contain only one or two data points.

When you submit the error, include the following:

- **The expected behaviour.** For example, what you are trying to do and what
  you are trying to model.
- **The observed behaviour.**  This includes the text (NOT a screenshot) of the
  *full* error message (if applicable).  This should be in a Github code block
  to make it readable.  You may also include plots (if applicable) which may
  help understand the problem.
- **The PyDDM version** (``print(pyddm.__version__)``).  Also include whether you
  installed it from pip or directly from Github.
- **Your "minimal reproducible example".** Please read the above carefully about
  how to construct a minimal reproducible example.  Put it in a Github code
  block to make it readable.  Do NOT use screenshots of code.  Use the "preview"
  feature on Github to make sure it is readable.

If you do not include all of these things and a minimal reproducible example in
the format described above, it will be difficult or impossible to help you.
Though in many cases, the process of constructing this minimal reproducible
example will result in solving the problem without the need for help!


Bugs
~~~~

Please report bugs in PyDDM to <https://github.com/mwshinn/pyddm/issues>.  This
includes any problems with the documentation.  PRs for bugs are greatly
appreciated.  If your report is not about a problem in PyDDM, please do not
report this as a bug.  Instead, you can ask these questions at
<https://github.com/mwshinn/PyDDM/discussions>.

Feature requests are currently not being accepted due to limited
resources.  If you implement a new feature in PyDDM, please do the
following before submitting a PR on Github:

- Make sure your code is clean and well commented
- If appropriate, update the official documentation in the docs/
  directory
- Ensure there are Paranoid Scientist verification conditions to your
  code
- Write unit tests and optionally integration tests for your new
  feature (runtests.sh)
- Ensure all existing tests pass (``runtests.sh`` returns without
  error)

For all other questions or comments, contact m.shinn@ucl.ac.uk.
