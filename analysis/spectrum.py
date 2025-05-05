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
p = sorted(glob("../measurement/2025-05-02/*.asc"))

d = np.loadtxt(p[1])
bkg = np.loadtxt(p[2])

wl = d[:,0]
d = d[:,1] - bkg[:,1]
d /= 2

plt.plot(wl, d)
plt.ylabel("counts / s")
plt.xlabel("wavelength (nm)")
plt.savefig("figures/spectrum.pdf")
plt.show()
# %%
