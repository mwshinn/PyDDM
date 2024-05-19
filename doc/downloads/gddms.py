# Some example GDDMs from the PyDDM tutorial, specified through the gddm function

# Start condition model
import pyddm
m = pyddm.gddm(
    drift=lambda drift_rate_scale,coh : drift_rate_scale*coh,
    parameters={"drift_rate_scale": (-6,6)},
    conditions=["coh"])
# End condition model

# Start condition_noparams model
m_to_sim = pyddm.gddm(drift=lambda coh : 3.8*coh, conditions=["coh"])
# End condition_noparams model

# Start solve multiple conditions
samp_coh3 = m_to_sim.solve(conditions={"coh": .5}).sample(2000) # 2000 trials with coh=.5
samp_coh6 = m_to_sim.solve(conditions={"coh": 1.0}).sample(1000) # 1000 trials with coh=1.0
samp_coh0 = m_to_sim.solve(conditions={"coh": 0}).sample(400) # 400 trials with coh=0
sample = samp_coh3 + samp_coh6 + samp_coh0 # This preserves information about the conditions
# End solve multiple conditions

# Start fit condition model
m.fit(sample, verbose=False)
print(m.parameters())
# End fit condition model

# Start conditions model gui
import pyddm.plot
pyddm.plot.model_gui(m, conditions={"coh": [0, .3, .6]})
pyddm.plot.model_gui_jupyter(m, conditions={"coh": [0, .3, .6]})
# End conditions model gui

# Start drift bounds gddm
import pyddm
m = pyddm.gddm(
    drift=lambda drift_rate,t : drift_rate*np.exp(t),
    bound=lambda bound_height,t : np.max(bound_height-t, 0),
    parameters={"drift_rate": (-2,2), "bound_height": (.5, 2)})
# pyddm.plot.model_gui(m) # ...or...
pyddm.plot.model_gui_jupyter(m)
# End drift bounds gddm

# Start leaky gddm
import pyddm
m = pyddm.gddm(
    drift=lambda drift_rate,leak,x : drift_rate - x*leak,
    parameters={"drift_rate": (-2,2), "bound_height": (.5, 2)})
# pyddm.plot.model_gui(m) # ...or...
pyddm.plot.model_gui_jupyter(m)
# End leaky gddm
