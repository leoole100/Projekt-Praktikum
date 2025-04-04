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

reference = to_xarray(np.loadtxt("../measurement/2025-04-04/001.txt"))
spectrum = to_xarray(np.loadtxt("../measurement/2025-04-04/002.asc"))
# spectrum -= 1190 # recorded background
spectrum -= spectrum.min()
spectrum = spectrum.sel(wavelength=slice(220, 1200))
# spectrum["wavelength"] = spectrum["wavelength"]/1.92

norm = lambda a: a/a.sel(wavelength=slice(400, 600)).max()
eff = norm(spectrum)/norm(reference).interp(wavelength=spectrum.wavelength)

# import the camera curve
man = to_xarray(np.loadtxt("../measurement/2025-04-03/QEcurve.dat")).sortby("wavelength")

norm(reference).plot(label="Ocean Optics")
norm(spectrum).plot(label="Andor Solis")
plt.legend()
plt.xlabel(r"$\lambda$ (nm)")
plt.savefig("figures/efficiency_spectrum.pdf")
plt.show()

fig = plt.figure()
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0, 0])
norm(reference).plot(label="Ocean Optics", ax=ax1)
norm(spectrum).plot(label="Andor Solis", ax=ax1)
ax1.set_ylabel("Intensity")
ax1.legend()

ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
man.plot(label="Camera", color="C2", ax=ax2)
(eff*100).plot(color="C1", ax=ax2)
ax2.set_ylabel("Efficiency (%)")
ax2.set_xlabel(r"$\lambda$ (nm)")
ax2.set_ylim(0, 100)
ax2.legend()
ax1.label_outer()

fig.savefig("figures/efficiency.pdf")
# %%
