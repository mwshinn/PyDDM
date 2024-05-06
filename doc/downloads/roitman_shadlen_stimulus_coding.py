# Fitting the Roitman and Shadlen (2002) dataset with stimulus coding.  See
# https://shadlenlab.columbia.edu/resources/RoitmanDataCode.html to
# download the dataset.

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
  
# Adjust the dataset for stimulus coding
df_rt['choice'] = df_rt['trgchoice'] % 2
df_rt['coh'] = df_rt['coh'] * ((df_rt["choice"] == df_rt["correct"])*2-1)

# Create a sample object from our data.  This is the standard input
# format for fitting procedures.  Since RT and correct/error are
# both mandatory columns, their names are specified by command line
# arguments.
roitman_sample = Sample.from_pandas_dataframe(df_rt, rt_column_name="rt", choice_column_name="choice", choice_names=("target 1", "target 2"))


# Define a model which uses our new DriftCoherence defined above.
from pyddm import gddm
from pyddm.models import NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture
model_rs = gddm(name='Roitman data, drift varies with coherence',
                drift=lambda driftcoh,coh: driftcoh*coh,
                noise=1,
                bound="B",
                nondecision_time="nondectime",
                uniform_mixture_coef=.02,
                choice_names=("target 1", "target 2"),
                dx=.001, dt=.01, T_dur=2,
                parameters={"driftcoh": (-20, 20), "B": (.1, 1.5), "nondectime": (0, .4)},
                conditions=["coh"])

# Fitting this will also be fast because PyDDM can automatically
# determine that DriftCoherence will allow an analytical solution.
model_rs.fit(sample=roitman_sample, verbose=False)
fit_model_rs.show()
fit_model_rs.parameters()


# To get an intuition for how parameters affect the fit, play with the
# parameters and task conditions in a GUI.
pyddm.plot.model_gui(model=fit_model_rs, sample=roitman_sample)
