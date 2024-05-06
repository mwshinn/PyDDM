# Some example GDDMs from the PyDDM tutorial, specified through the gddm function

# Start condition model
import pyddm
m = pyddm.gddm(
    drift=lambda drift_rate_scale,coh : drift_rate_scale*coh,
    parameters={"drift_rate_scale": (-2,2)},
    conditions=["coh"])
# End condition model

# Start solve multiple conditions
samp_coh3 = m.solve(conditions={"coh": .3}).resample(1000) # 1000 trials with coh=.3
samp_coh6 = m.solve(conditions={"coh": .6}).resample(500) # 500 trials with coh=.6
samp_coh0 = m.solve(conditions={"coh": 0}).resample(100) # 100 trials with coh=.6
sample = samp_coh3 + samp_coh6 + samp_coh0 # This preserves information about the conditions
# End solve multiple conditions

# Start conditions model gui
import pyddm.plot
# To manually specify conditions
pyddm.plot.model_gui(m, conditions={"coh": [0, .3, .6]})
pyddm.plot.model_gui_jupyter(m, conditions={"coh": [0, .3, .6]})

# To use a sample object (the one we just generated)
pyddm.plot.model_gui(m, sample)
pyddm.plot.model_gui_jupyter(m, sample)
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
