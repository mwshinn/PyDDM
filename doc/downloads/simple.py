# Simple demonstration of PyDDM.

# Create a simple model with constant drift, noise, and bounds.
from ddm import Model
from ddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayNonDecision
from ddm.functions import fit_adjust_model, display_model

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
from ddm import Fittable
from ddm.models import LossBIC
from ddm.functions import fit_adjust_model
model_fit = Model(name='Simple model (fitted)',
                  drift=DriftConstant(drift=Fittable(minval=0, maxval=4)),
                  noise=NoiseConstant(noise=Fittable(minval=.5, maxval=4)),
                  bound=BoundConstant(B=1.1),
                  overlay=OverlayNonDecision(nondectime=Fittable(minval=0, maxval=1)),
                  dx=.001, dt=.01, T_dur=2)

fit_adjust_model(samp, model_fit,
                 method="differential_evolution",
                 lossfunction=LossBIC)

display_model(model_fit)

# Plot the model fit to the PDFs and save the file.
import ddm.plot
import matplotlib.pyplot as plt
ddm.plot.plot_fit_diagnostics(model=model_fit, sample=samp)
plt.savefig("simple-fit.png")
plt.show()

print(sol.prob_correct())
print(sol.pdf_err())
