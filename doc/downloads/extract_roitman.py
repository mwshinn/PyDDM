import numpy as np
import scipy.io
from glob import glob
import pandas
import os

if "b_rt" not in glob("*") or "n_rt" not in glob("*"):
    exit("Error, please download the data from "
         "https://shadlenlab.columbia.edu/resources/RoitmanDataCode.html "
         "and copy this script into the extracted zip file (i.e. such "
         "that 'n_rt' and 'b_rt' are in the same directory as this script)")

# Load each session as a separate pandas dataframe and save them in
# the dfs list.
dfs = []
for fn in glob("*_rt/*.mat"):
    # These magic numbers and names come from the structure of the
    # .mat file.
    d = scipy.io.loadmat(fn, squeeze_me=True, chars_as_strings=True)['data']
    labels = list(d[0].flatten()[0][7][0])
    data = d[1]
    # The byteswap stuff is due to loading from matlab... see:
    # https://stackoverflow.com/questions/30283836/creating-pandas-dataframe-from-numpy-array-leads-to-strange-errors
    df = pandas.DataFrame(d[1].byteswap().newbyteorder(), columns=labels)
    df['session'] = os.path.basename(fn).split(".")[0]
    df['monkey'] = 1 if 'b_rt' in fn else 2
    dfs.append(df)

# Check to make sure the columns are all the same
for df in dfs:
    assert np.all(df.columns == dfs[0].columns)

# Make the master dataframe
expdata = pandas.concat(dfs)

# Compute reaction time in seconds.
expdata['rt'] = (expdata['sac'] - expdata['ston'])/1000

# Convert coherence to a proportion
expdata['coh'] = expdata['coh']/1000

# To keep things clean, just keep the data that we care about for this
# analysis.
taskdata = expdata[['monkey','rt','coh','correct', 'trgchoice']].dropna()

# Integrity check
assert set(taskdata['monkey']) == {1, 2}
assert set(taskdata['correct']) == {0, 1}
assert set(taskdata['trgchoice']) == {1, 2}
assert np.all(taskdata['rt'] > 0)
assert np.all(0 <= taskdata['coh']) and np.all(taskdata['coh'] <= 1000)

# Save to csv
taskdata.to_csv('roitman_rts.csv', index=False)
