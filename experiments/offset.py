"""
Gets the pixel-to-nm offset from Andor Solis files for a spectrometer system.
"""
#%%
import numpy as np
from glob import glob
from scipy.interpolate import CubicSpline

# %%
paths = glob("../measurement/*/*grating=*nm*")
positions = [int(path.split("grating=")[-1].split("nm")[0]) for path in paths]

sorted_indices = np.argsort(positions)
paths = [paths[i] for i in sorted_indices]
positions = [positions[i] for i in sorted_indices]

wls = [np.loadtxt(p)[:, 0] for p in paths]
offsets = np.array([wl-p for (p, wl) in zip(positions, wls)])
spl = CubicSpline(positions, offsets)

def offset(position:float):
	"""Returns pixel-wise offset (in nm) for a given monochromator center wavelength."""
	return spl(position)

# %%
if __name__ == "__main__":
	import matplotlib.pyplot as plt
	import matplotlib as mpl
	plt.style.use("../style.mplstyle")
	all_offsets = np.concatenate(offsets)
	all_pixels = np.concatenate([np.arange(len(o)) for o in offsets])

	linear_fit = np.polyfit(all_pixels, all_offsets, 1)
	linear_trend = np.polyval(linear_fit, all_pixels)

	non_linear_part = all_offsets - linear_trend

	for p, o, c in zip(positions, offsets, mpl.colormaps['plasma'](np.linspace(0,1, len(positions)))):
		start_idx = sum(len(offsets[i]) for i in range(positions.index(p)))
		end_idx = start_idx + len(o)
		plt.plot(non_linear_part[start_idx:end_idx], label=f"{p} nm", color=c)
	plt.legend()
	plt.ylabel("nonlinear offset in nm")
	plt.xlabel("pixel number")
	plt.savefig("figures/spectrometer offset.pdf")
	plt.show()