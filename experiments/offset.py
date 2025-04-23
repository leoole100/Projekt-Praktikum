"""
get's the pixel to nm offset from a andor solis file
"""
#%%
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

#%%
# get file path
p = glob("../measurement/*/*grating=600nm*")[-1]

# get wavelengths
wl = np.loadtxt(p)[:, 0]

# get offset
wl = wl-600
px = np.arange(len(wl))

np.savetxt("offset.csv", wl)

# %% look at a different grating position
p = glob("../measurement/*/*grating=1200nm*")

#%%

p = np.polyfit(px, wl, 3)
fit = np.polyval(p, px)

plt.plot(px, wl-fit)
plt.xlabel("residual")

# %%