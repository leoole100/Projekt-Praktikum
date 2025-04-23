"""
get's the pixel to nm scale from a andor solis file
"""
#%%
import numpy as np
from glob import glob

#%%
# get file path
p = glob("../measurement/*/*grating=600nm*")[-1]

# get wavelengths
wl = np.loadtxt(p)[:, 1]

# get offset
wl = wl-600

# %%

# get the pixel number
center_px = np.argmin(wl**2)
print(center_px)
# (np.arange(len(wl))-center_px)