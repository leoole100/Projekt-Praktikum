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

#%%
def spectrum(p):
	a = np.loadtxt(p)
	d = xr.DataArray(a[:,1], {"wavelength":a[:,0]})
	d.attrs["path"] = p
	return d

ocean_optics = {p:spectrum(p) for p in glob("../measurement/2025-04-04/004*.txt")}

our_spectrometer = {p:spectrum(p) for p in glob("../measurement/2025-04-09/001*")}
bkg = spectrum("../measurement/2025-04-09/001 bkg.asc")
our_spectrometer = {p:i-bkg for p,i in our_spectrometer.items()}

norm = lambda s: s/s.sel(wavelength=slice(850, 1000)).max()

plt.figure()
# for i in ocean_optics:
# 	norm(i).plot(color="b", label="OceanOptics")


for p,i in our_spectrometer.items():
	norm(i).plot(label=p)

plt.legend()