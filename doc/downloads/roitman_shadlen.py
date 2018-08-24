# Fitting the Roitman and Shadlen (2002) dataset.  See
# https://shadlenlab.columbia.edu/resources/RoitmanDataCode.html to
# download the dataset.

# Read the dataset into a Pandas DataFrame
from ddm import Sample
import pandas
with open("roitman_rts.csv", "r") as f:
    df_rt = pandas.read_csv(f)

df_rt = df_rt[df_rt["monkey"] == 1] # Only monkey 1
  
# Remove short and long RTs, as in 10.1523/JNEUROSCI.4684-04.2005.
# This is not strictly necessary, but is performed here for
# compatibility with this study.
df_rt = df_rt[df_rt["rt"] > .1] # Remove trials less than 100ms
df_rt = df_rt[df_rt["rt"] < 1.65] # Remove trials greater than 1650ms
  
# Create a sample object from our data.  This is the standard input
# format for fitting procedures.  Since RT and correct/error are
# both mandatory columns, their names are specified by command line
# arguments.
roitman_sample = Sample.from_pandas_dataframe(df_rt, rt_column_name="rt", correct_column_name="correct")



# For demonstration purposes, repeat the above using a numpy matrix.
from ddm import Sample
import numpy as np
with open("roitman_rts.csv", "r") as f:
    M = np.loadtxt(f, delimiter=",", skiprows=1)

# RT data must be the first column and correct/error must be the
# second column.
rt = M[:,1].copy() # Use .copy() because np returns a view
corr = M[:,3].copy()
monkey = M[:,0].copy()
M[:,0] = rt
M[:,1] = corr
M[:,3] = monkey

# Only monkey 1
M = M[M[:,3]==1,:]

# As before, remove longest and shortest RTs
M = M[M[:,0]>.1,:]
M = M[M[:,0]<1.65,:]
  
conditions = ["coh", "monkey", "trgchoice"]
roitman_sample2 = Sample.from_numpy_array(M, conditions)

# As we can see, these two approches are equivalent.
assert roitman_sample == roitman_sample2



# Define a Drift component which determines drift rate from trial coherence.
import ddm.models
class DriftCoherence(ddm.models.Drift):
    name = "Drift depends linearly on coherence"
    required_parameters = ["driftcoh"] # <-- Parameters we want to include in the model
    required_conditions = ["coh"] # <-- Task parameters ("conditions"). Should be the same name as in the sample.
    
    # We must always define the get_drift function, which is used to compute the instantaneous value of drift.
    def get_drift(self, conditions, **kwargs):
        return self.driftcoh * conditions['coh']


# Define a model which uses our new DriftCoherence defined above.
from ddm import Model, Fittable
from ddm.functions import fit_adjust_model, display_model
from ddm.models import NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture
model_rs = Model(name='Roitman data, drift varies with coherence',
                 drift=DriftCoherence(driftcoh=Fittable(minval=0, maxval=20)),
                 noise=NoiseConstant(noise=1),
                 bound=BoundConstant(B=Fittable(minval=.1, maxval=1.5)),
                 # Since we can only have one overlay, we use
                 # OverlayChain to string together multiple overlays.
                 # They are applied sequentially in order.  OverlayNonDecision
                 # implements a non-decision time by shifting the
                 # resulting distribution of response times by
                 # `nondectime` seconds.
                 overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=.4)),
                                                OverlayPoissonMixture(pmixturecoef=.02,
                                                                      rate=1)]),
                 dx=.001, dt=.01, T_dur=2)

# Fitting this will also be fast because PyDDM can automatically
# determine that DriftCoherence will allow an analytical solution.
fit_model_rs = fit_adjust_model(sample=roitman_sample, model=model_rs)
display_model(fit_model_rs)


# Plot the model fit to the PDFs and save the file.
import ddm.plot
import matplotlib.pyplot as plt
ddm.plot.plot_fit_diagnostics(model=fit_model_rs, sample=roitman_sample)
plt.savefig("roitman-fit.png")
plt.show()


# To get an intuition for how parameters affect the fit, play with the
# parameters and task conditions in a GUI.
ddm.plot.model_gui(model=fit_model_rs, sample=roitman_sample)


# Let's try to improve the model fit.

# We define a model which allows us to use a coherence-dependent leaky integrator.
class DriftCoherenceLeak(ddm.models.Drift):
    name = "Leaky drift depends linearly on coherence"
    required_parameters = ["driftcoh", "leak"] # <-- Parameters we want to include in the model
    required_conditions = ["coh"] # <-- Task parameters ("conditions"). Should be the same name as in the sample.
    
    # We must always define the get_drift function, which is used to compute the instantaneous value of drift.
    def get_drift(self, x, conditions, **kwargs):
        return self.driftcoh * conditions['coh'] + self.leak * x

# Now define the model using a leaky 
from ddm.models import BoundCollapsingExponential
model_leak = Model(name='Roitman data, leaky drift varies with coherence',
                   drift=DriftCoherenceLeak(driftcoh=Fittable(minval=0, maxval=20),
                                            leak=Fittable(minval=-10, maxval=10)),
                   noise=NoiseConstant(noise=1),
                   bound=BoundCollapsingExponential(B=Fittable(minval=0.5, maxval=3),
                                                    tau=Fittable(minval=.0001, maxval=5)),
                   # Since we can only have one overlay, we use
                   # OverlayChain to string together multiple overlays.
                   # They are applied sequentially in order.  OverlayDelay
                   # implements a non-decision time by shifting the
                   # resulting distribution of response times by
                   # `delaytime` seconds.
                   overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=.4)),
                                                  OverlayPoissonMixture(pmixturecoef=.02,
                                                                        rate=1)]),
                   dx=.01, dt=.01, T_dur=2)

# Fitting this will also be fast because we can automatically
# determine that DriftCoherence will allow an analytical solution.
fit_model_leak = fit_adjust_model(sample=roitman_sample, model=model_leak)
ddm.plot.plot_fit_diagnostics(model=fit_model_leak, sample=roitman_sample)
plt.savefig("leak-collapse-fit.png")
# If the fitting step is too slow, you can use this pre-fit model:
#     Model(name='Roitman data, leaky drift varies with coherence',
#           drift=DriftCoherenceLeak(driftcoh=10.49091, leak=-.482),
#           noise=NoiseConstant(noise=1),
#           bound=BoundCollapsingExponential(B=1.811, tau=1.992),
#           overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=.211),
#                                          OverlayPoissonMixture(pmixturecoef=0.02, rate=1)]),
#           dx=0.01, dt=0.01, T_dur=2)

# Or, in a form which you can copy and paste:
#     Model(name='Roitman data, leaky drift varies with coherence', drift=DriftCoherenceLeak(driftcoh=10.49091, leak=-.482), noise=NoiseConstant(noise=1), bound=BoundCollapsingExponential(B=1.811, tau=1.992), overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=.211), OverlayPoissonMixture(pmixturecoef=0.02, rate=1)]), dx=0.01, dt=0.01, T_dur=2)

