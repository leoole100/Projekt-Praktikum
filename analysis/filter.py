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
# plt.plot(wavelengths, laser, label="Laser", linestyle="--", color="k")

# for path, spectrum in spectra.items():
	# plt.plot(spectrum[:, 0], absorbance(spectrum), label=os.path.basename(path).rsplit(".")[0][4:])

plt.plot(spectra['../measurement/2025-05-05/001 KG3.asc'][:, 0], absorbance(spectra['../measurement/2025-05-05/001 KG3.asc']), label="KG3 Filter")
plt.plot(spectra['../measurement/2025-05-05/001 HR.asc'][:, 0], absorbance(spectra['../measurement/2025-05-05/001 HR.asc']), label="HR Filter")

camera = np.loadtxt("../measurement/2025-04-03/QEcurve.dat")
camera = xr.DataArray(camera[:,1], {"wavelength":camera[:,0]}).sortby("wavelength")
plt.plot(camera.wavelength, camera/100, label="Camera")


def blaze(l, lc=500, k=.75): # https://adsabs.harvard.edu/full/1984AJ.....89..899B
	return np.sin(np.pi*k*(1-lc/l))**2/(np.pi*k*(1-lc/l))**2
plt.plot(camera["wavelength"], blaze(camera["wavelength"])*0.95, label="Grating")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Efficiency")
plt.ylim(0, 1)
plt.legend(loc="upper left", frameon=True)
plt.savefig("./figures/filter.pdf")
plt.show()

# %%
#%%



plt.plot()
def blaze(l, lc=500, k=.75): # https://adsabs.harvard.edu/full/1984AJ.....89..899B
	return np.sin(np.pi*k*(1-lc/l))**2/(np.pi*k*(1-lc/l))**2
plt.plot(man["wavelength"], blaze(man["wavelength"])*100, label="Blaze")
(blaze(man["wavelength"])*man).plot(label="Combined", color="k")
plt.legend()
plt.xlabel(r"$\lambda$ (nm)")
plt.ylabel(r"Relative Efficiency (%)")
plt.gcf().savefig("./figures/expected.pdf")