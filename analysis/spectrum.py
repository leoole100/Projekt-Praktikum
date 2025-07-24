#%%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
import xarray as xr
from glob import glob
import scipy as sp

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# %%
p = sorted(glob("../measurement/2025-05-05/003 thermal*.asc"))

d = np.loadtxt(p[0])

wl = d[:,0]
d = d[:,1] - np.min(d[:,1])
d /= 2 # convert to counts / s
d = np.diff(wl) * d[:-1] # convert to counts / s / nm
wl = wl[:-1] + np.diff(wl)/2 # center the wavelength bins

plt.plot(wl, d)
mask = (wl >= 0) & (wl <= 800)
max_value = np.max(d[mask])
plt.ylim(0, max_value*1.1)
plt.ylabel("counts / s / nm")
plt.xlabel("wavelength (nm)")
plt.savefig("figures/spectrum.pdf")
plt.show()

# %%
# correct the efficiency
# import the camera curve
camera = np.loadtxt("../measurement/2025-04-03/QEcurve.dat")
def blaze(l, lc=500, k=.75): # https://adsabs.harvard.edu/full/1984AJ.....89..899B
	return np.sin(np.pi*k*(1-lc/l))**2/(np.pi*k*(1-lc/l))**2
combined = blaze(camera[:,0]) * camera[:,1]
combined *= 0.6 # low pass filter for < 900 nm
combined *= 0.053 # collection efficiency
combined *= 0.04 # fiber collecting efficiency

plt.figure()
sorted_indices = np.argsort(camera[1:, 0])
plt.plot(camera[1:, 0][sorted_indices], combined[1:][sorted_indices])
plt.xlabel("wavelength (nm)")
plt.ylabel("combined efficiency (%)")
plt.title("Combined System Efficiency")
plt.show()

efficiency = sp.interpolate.interp1d(camera[:,0], combined, bounds_error=False, fill_value=0)

eff_mask = efficiency(wl) > 0.2*np.max(efficiency(wl))
corrected_counts = d[eff_mask] / (efficiency(wl[eff_mask]) / 100)

plt.plot(wl[eff_mask], corrected_counts)
plt.ylabel("corrected counts / s / nm")

plt.show()

# Calculate energy per wavelength (w / nm)
h = 6.62607015e-34  # Planck constant (J·s)
c = 2.99792458e8    # Speed of light (m/s)

# Convert wavelength from nm to m
wl_m = wl[eff_mask] * 1e-9

# Energy per photon: E = h * c / λ
energy_per_photon = h * c / wl_m  # in Joules

# Power per nm: counts * energy per photon
w_per_nm = corrected_counts * energy_per_photon

# plot the model
import sys
import os
# sys.path.append(os.path.abspath(".."))
# %%
# from model.model import Model
# from model.streak import Streak
# m = Model()
# m.E = 50e-6
# m.fwhm = 250e-15
# s = Streak(m)
# s()
# s.T_e.max()
# plt.plot(s.l/1e-9, s.b.sum(axis=1)/1e15*4e-15)

plt.plot(wl[eff_mask], w_per_nm)
plt.ylabel("corrected power (W / nm)")
plt.xlabel("wavelength (nm)")
plt.ylim(0, None)
plt.show()

# Integrate total power over the measured wavelength range
total_power = np.trapz(w_per_nm, wl[eff_mask])
print(f"Total power: {total_power:.3e} W")

# %%
