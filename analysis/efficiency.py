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
def to_xarray(a):
	return xr.DataArray(a[:,1], {"wavelength":a[:,0]})

reference = to_xarray(np.loadtxt("../measurement/2025-04-04/001.txt"))
spectrum = to_xarray(np.loadtxt("../measurement/2025-04-04/002.asc"))
# spectrum -= 1190 # recorded background
spectrum -= spectrum.min()
spectrum = spectrum.sel(wavelength=slice(220, 1200))
# spectrum["wavelength"] = spectrum["wavelength"]/1.92

norm = lambda a: a/a.sel(wavelength=slice(400, 600)).max()
norm = lambda a: a
eff = norm(spectrum)/norm(reference).interp(wavelength=spectrum.wavelength)

# import the camera curve
man = to_xarray(np.loadtxt("../measurement/2025-04-03/QEcurve.dat")).sortby("wavelength")

norm(reference).plot(label="Ocean Optics")
norm(spectrum).plot(label="Andor Solis")
plt.legend()
plt.xlabel(r"$\lambda$ (nm)")
plt.yscale("log")
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
ax2.set_ylim(0, 200)
ax2.legend()
ax1.label_outer()

fig.savefig("figures/efficiency.pdf")

#%%
man.plot(label="Camera")
def blaze(l, lc=500, k=.75): # https://adsabs.harvard.edu/full/1984AJ.....89..899B
	return np.sin(np.pi*k*(1-lc/l))**2/(np.pi*k*(1-lc/l))**2
plt.plot(man["wavelength"], blaze(man["wavelength"])*100, label="Blaze")
(blaze(man["wavelength"])*man).plot(label="Combined", color="k")
plt.legend()
plt.xlabel(r"$\lambda$ (nm)")
plt.ylabel(r"Relative Efficiency (%)")
plt.gcf().savefig("./figures/expected.pdf")
# %%

paths = list(sorted(glob("../measurement/2025-04-04/004*")))[:-1]
references = [to_xarray(np.loadtxt(p)) for p in paths]
spectrum = to_xarray(np.loadtxt("../measurement/2025-04-04/004e.asc"))
spectrum -= spectrum.min()

def scale(target, reference):
	t, r = target.interp(wavelength=reference.wavelength), reference
	scale = (r * t).sum() / (t**2).sum()
	return target*scale

label = [
	"USB2000+",
	"USB2000+XR1-ES",
	"USB2000+",
	"USB4000",
]

for r,l in zip(references,label): 
	scale(r, spectrum).plot(label=l)
spectrum.plot(color="k", label="Acton & Andor")
plt.ylim(5e2, None)
plt.xlim(200, 1250)
# plt.yscale("log")
plt.xlabel(r"Wavelength (nm)")
plt.ylabel("Intensity Scaled")
plt.gca().yaxis.set_tick_params(labelbottom=False)
plt.legend()

plt.gcf().savefig("figures/efficiency_different.pdf")
# %%
