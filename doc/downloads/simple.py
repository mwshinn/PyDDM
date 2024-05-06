# Simple demonstration of PyDDM.

import pyddm

# Construct a model, show it, and then solve it.

# Start ModelDef
model = pyddm.gddm(drift=0.5, noise=1.0, bound=0.6, starting_position=0.3, nondecision=0.2)
# End ModelDef

# Start ShowModel
model.show()
# End ShowModel

# Start ModelSolve
sol = model.solve()
# End ModelSolve

# Perform parameter recovery on the same model but with free parameters.  First,
# construct the model with free parameters, and visualize it with the model GUI.
# Then, sample some artificial data from our previous model, and then fit the
# new model to this artificial data.  Then, we can check to make sure that the
# parameters are similar to the ones used to generate the data.


# Start ModelFittableDef
model_to_fit = pyddm.gddm(drift="d", noise=1.0, bound="B", nondecision=0.2, starting_position="x0",
                                parameters={"d": (-2,2), "B": (0.3, 2), "x0": (-.8, .8)})
model.show()
# End ModelFittableDef

# Start ModelFittableAltDef
model_to_fit = pyddm.gddm(drift=lambda d : d, noise=1.0, bound=lambda B : B, nondecision=0.2, starting_position=lambda x0 : x0,
                                parameters={"d": (-2,2), "B": (0.3, 2), "x0": (-.8, .8)})
# End ModelFittableAltDef

# Start ModelFittableAlt2Def
def drift_function(d):
    return d
def another_func(B):
    return B
third_function = lambda x0: x0
model_to_fit = pyddm.gddm(drift=drift_function, noise=1.0, bound=another_func, nondecision=0.2, starting_position=third_function,
                                parameters={"d": (-2,2), "B": (0.3, 2), "x0": (-.8, .8)})
# End ModelFittableAlt2Def

# Start ModelGui
import pyddm.plot
pyddm.plot.model_gui(model_to_fit)
# End ModelGui

# Start Resample
samp_simulated = sol.resample(10000)
# End Resample

# Start Fit
model_to_fit.fit(samp_simulated, lossfunction=pyddm.LossBIC, verbose=False)
model_to_fit.show()
# End Fit

# Start Parameters
model_to_fit.parameters()
# End Parameters

# Start Lossval
model_to_fit.get_fit_result().value()
# End Lossval

# Start Plot
# Plot the model fit to the PDFs and save the file.
import pyddm.plot
import matplotlib.pyplot as plt
pyddm.plot.plot_fit_diagnostics(model=model_fit, sample=samp)
plt.savefig("simple-fit.png")
plt.show()
# End Plot

# Start Probs
print(sol.prob("correct"))
print(sol.pdf("error"))
# End Probs

