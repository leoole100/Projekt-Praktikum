#%%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
import xarray as xr
from glob import glob

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# %%
p = sorted(glob("../measurement/2025-05-05/003 thermal*.asc"))

d = np.loadtxt(p[0])

wl = d[:,0]
d = d[:,1] - np.min(d[:,1])
d /= 2 * np.diff(wl).mean()

plt.plot(wl, d)
mask = (wl >= 0) & (wl <= 800)
max_value = np.max(d[mask])
plt.ylim(0, max_value*1.1)
plt.ylabel("counts / s / nm")
plt.xlabel("wavelength (nm)")
plt.savefig("figures/spectrum.pdf")
plt.show()
# %%
