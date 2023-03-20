# Simple demonstration of PyDDM.

# Create a simple model with constant drift, noise, and bounds.
from pyddm import Model
from pyddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayNonDecision, ICPointSourceCenter
from pyddm.functions import fit_adjust_model, display_model

model = Model(name='Simple model',
              drift=DriftConstant(drift=2.2),
              noise=NoiseConstant(noise=1.5),
              bound=BoundConstant(B=1.1),
              overlay=OverlayNonDecision(nondectime=.1),
              dx=.001, dt=.01, T_dur=2)

# Solve the model, i.e. simulate the differential equations to
# generate a probability distribution solution.
display_model(model)
sol = model.solve()

# Now, sample from the model solution to create a new generated
# sample.
samp = sol.resample(1000)

# Fit a model identical to the one described above on the newly
# generated data so show that parameters can be recovered.
from pyddm import Fittable, Fitted
from pyddm.models import LossRobustBIC
from pyddm.functions import fit_adjust_model
model_fit = Model(name='Simple model (fitted)',
                  drift=DriftConstant(drift=Fittable(minval=0, maxval=4)),
                  noise=NoiseConstant(noise=Fittable(minval=.5, maxval=4)),
                  bound=BoundConstant(B=1.1),
                  overlay=OverlayNonDecision(nondectime=Fittable(minval=0, maxval=1)),
                  dx=.001, dt=.01, T_dur=2)

fit_adjust_model(samp, model_fit,
                 fitting_method="differential_evolution",
                 lossfunction=LossRobustBIC, verbose=False)

display_model(model_fit)
model_fit.parameters()
model_fit.get_fit_result().value()

# Plot the model fit to the PDFs and save the file.
import pyddm.plot
import matplotlib.pyplot as plt
pyddm.plot.plot_fit_diagnostics(model=model_fit, sample=samp)
plt.savefig("simple-fit.png")
plt.show()

print(sol.prob("correct"))
print(sol.pdf("error"))

# Save the model
with open("model.txt", "w") as f:
    f.write(repr(model_fit))

# Load the model
from pyddm import FitResult
with open("model.txt", "r") as f:
    model_loaded = eval(f.read())
