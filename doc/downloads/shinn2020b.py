import ddm
import pandas
import matplotlib.pyplot as plt
import fit_roitman_models as mods

subj_dfs = []
# 0s delay first
for subj in range(1, 12):
    for session in range(1, 11):
        subj_df = pandas.read_csv(f"humanmonkey/Data/0s_Delay/Sub{subj:02}_Ses{session:02}.txt", sep="\t", header=6)
        subj_df['subject'] = subj
        subj_df['session'] = session
        subj_df['delay'] = 0
        subj_dfs.append(subj_df)

# 1s delay next
for subj in range(12, 22):
    for session in range(1, 11):
        subj_df = pandas.read_csv(f"humanmonkey/Data/1s_Delay/Sub{subj:02}_Ses{session:02}.txt", sep="\t", header=6)
        subj_df['subject'] = subj
        subj_df['session'] = session
        subj_df['delay'] = 1
        subj_dfs.append(subj_df)

df = pandas.concat(subj_dfs)

# Do the preprocessing described in Evans and Hawkins (2019): remove
# RTs greater than 7s, remove the first block of each subject, and
# remove the first session of what is presumably subject 6 (who had
# the lowest accuracy on the first session).
df = df.query('RT <= 7000 and blkNum > 1 and not (subject == 6 and session == 1)')


# Let's get a baseline fit of the GDDM.  We'll pool all the data.

# First, get ready to convert it to a sample object
df['coh' ] = df['percentCoherence']/100
df['RTsec'] = df['RT']/1000

# We take only the delay == 0 trials, since these are presumably
# better fit by a non-collapsing bounds model and are "typical" RT
# data.
sample = ddm.Sample.from_pandas_dataframe(df.query('delay == 0')[['coh', 'RTsec', 'correct']], 'RTsec', 'correct')


m = ddm.Model(drift=mods.DriftNLCoherenceLeak(driftcoh=ddm.Fittable(minval=0, maxval=15, default=9),
                                              leak=ddm.Fittable(minval=-5, maxval=5, default=-1), # Both leaky and unstable integration
                                              power=ddm.Fittable(minval=.5, maxval=1.5, default=1), # Exponent for coherence -> drift nonlinearity
                                              maxcoh=.4), # We don't need to fit this, just use the maximum coherence in the dataset
              noise=ddm.NoiseConstant(noise=1),
              bound=ddm.BoundCollapsingExponential(B=ddm.Fittable(minval=0.4, maxval=1.5, default=.7),
                                                   tau=ddm.Fittable(minval=.0001, maxval=5, default=.05)),
              overlay=ddm.OverlayChain(overlays=[ddm.OverlayNonDecision(nondectime=ddm.Fittable(minval=.1, maxval=.5, default=.4)),
                                                 ddm.OverlayUniformMixture(umixturecoef=.05)]),
              dx=.01, dt=.01, T_dur=7)

ddm.set_N_cpus(3)
fit_model = ddm.fit_adjust_model(sample, m)
