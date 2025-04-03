#%%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
import xarray as xr

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# %%
def to_xarray(a):
	return xr.DataArray(a[:,1], {"wavelength":a[:,0]})

reference = to_xarray(np.loadtxt("../measurement/2025-04-03/001.txt"))
spectrum = to_xarray(np.loadtxt("../measurement/2025-04-03/002.asc"))
spectrum = spectrum.rolling(wavelength=100).mean() # filter the spectrum
spectrum -= 23895 # recorded background

eff = (spectrum/reference.interp(wavelength=spectrum.wavelength))
eff = eff / eff.sel(wavelength=slice(300, None)).max() # normalize on peak

# import the manufactures curve
man = to_xarray(np.loadtxt("../measurement/2025-04-03/QEcurve.dat")).sortby("wavelength")

norm = lambda a: a/a.sel(wavelength=slice(400, 600)).max()

ax1 = plt.subplot(2, 1, 1)
norm(reference).plot(label="Ocean Optics")
norm(spectrum).plot(label="Andor Solis")
plt.ylabel("Intensity")
plt.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
man.plot(label="Manufacturer", color="C2")
(eff*100).plot(label="Measured", color="C3")
plt.ylabel("Efficiency (%)")
plt.ylim(0, 100)
plt.legend()
ax1.label_outer()

plt.gcf().savefig("figures/efficiency.pdf")