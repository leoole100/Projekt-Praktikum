#%%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
import xarray as xr
from glob import glob

import os
from scipy.stats import norm

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#%%

paths = glob("../measurement/2025-05-05/001*")

spectra = {p: np.loadtxt(p) for p in paths}

spectra = {p: np.column_stack((s[:, 0], s[:, 1] - np.min(s[:, 1]))) for p, s in spectra.items()}

pw = list(filter(lambda x: x.find("white") != -1, paths))[0]
white = spectra.pop(pw)

def absorbance(s, min_counts=20):
	filtered = np.where(s[:, 1] < min_counts, np.nan, s[:, 1])
	return filtered / white[:, 1][:len(filtered)]

# plot the approximate laser shape
wavelengths = np.linspace(1000, 1064, 100)
laser = norm.pdf(wavelengths, loc=1032, scale=10)
laser /= np.max(laser) 
plt.plot(wavelengths, laser, label="Laser", linestyle="--", color="k")

for path, spectrum in spectra.items():
	plt.plot(spectrum[:, 0], absorbance(spectrum), label=os.path.basename(path).rsplit(".")[0])

plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.ylim(0, 1)
plt.legend()
plt.savefig("./figures/filter.pdf")
plt.show()

# %%
