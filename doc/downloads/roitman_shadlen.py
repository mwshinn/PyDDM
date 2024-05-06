# Fitting the Roitman and Shadlen (2002) dataset.  See
# https://shadlenlab.columbia.edu/resources/RoitmanDataCode.html to
# download the dataset.

import pyddm

# Start Load
# Read the dataset into a Pandas DataFrame
from pyddm import Sample
import pandas
with open("roitman_rts.csv", "r") as f:
    df_rt = pandas.read_csv(f)

df_rt = df_rt[df_rt["monkey"] == 1] # Only monkey 1
  
# Remove short and long RTs, as in 10.1523/JNEUROSCI.4684-04.2005.
# This is not strictly necessary, but is performed here for
# compatibility with this study.
df_rt = df_rt[df_rt["rt"] > .1] # Remove trials less than 100ms
df_rt = df_rt[df_rt["rt"] < 1.65] # Remove trials greater than 1650ms
# End Load

# Create a sample object from our data.  This is the standard input
# format for fitting procedures.  Since RT and correct/error are
# both mandatory columns, their names are specified by command line
# arguments.
# Start Sample
roitman_sample = Sample.from_pandas_dataframe(df_rt, rt_column_name="rt", choice_column_name="correct")
# End Sample


# Start Numpy Load
# For demonstration purposes, repeat the above using a numpy matrix.
from pyddm import Sample
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
# End Numpy Load

# As we can see, these two approches are equivalent.
# Start Test
assert roitman_sample == roitman_sample2
# End Test


# Start Model
m = pyddm.gddm(drift=lambda coh, driftcoh : driftcoh*coh,
                     noise=1,
                     bound="b",
                     nondecision="ndtime",
                     parameters={"driftcoh": (-20,20), "b": (.4, 3), "ndtime": (0, .5)},
                     conditions=["coh"])
# pyddm.plot.model_gui(m) # ...or...
pyddm.plot.model_gui_jupyter(m, sample=roitman_sample)
# End Model



# Start display model
# Fitting this will also be fast because PyDDM can automatically
# determine that DriftCoherence will allow an analytical solution.
m.fit(sample=roitman_sample, verbose=False) # Set verbose=True to see fitting progress
m.show()
print("Parameters:", m.parameters())
# End display model

# Start Plot
# Plot the model fit to the PDFs and save the file.
import pyddm.plot
import matplotlib.pyplot as plt
pyddm.plot.plot_fit_diagnostics(model=m, sample=roitman_sample)
plt.savefig("roitman-fit.png")
plt.show()
# End Plot

# Start Gui
# pyddm.plot.model_gui(m) # ...or...
pyddm.plot.model_gui_jupyter(m, sample=roitman_sample)
# End Gui


# Start leak model
model_leak = pyddm.gddm(
    drift=lambda driftcoh,leak,coh,x : driftcoh*coh - leak*x,
    bound=lambda bound_base,invtau,t : bound_base * np.exp(-t*invtau),
    nondecision="ndtime",
    parameters={"driftcoh": (-20,20),
                "leak": (-5, 5),
                "bound_base": (.5, 10),
                "ndtime": (0, .5),
                "invtau": (.1, 10)},
    conditions=["coh"])
# End leak model

# Start leak model show
# Fit, plot, and show the result
model_leak.fit(sample=roitman_sample, verbose=False)
pyddm.plot.plot_fit_diagnostics(model=model_leak, sample=roitman_sample)
plt.savefig("leak-collapse-fit.png")
model_leak.show()
# End leak model show
