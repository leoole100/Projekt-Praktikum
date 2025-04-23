"""
get's the pixel to nm offset from a andor solis file
"""
#%%
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# %%
paths = glob("../measurement/*/*grating=*nm*")
positions = [int(path.split("grating=")[-1].split("nm")[0]) for path in paths]
wls = [np.loadtxt(p)[:, 0] for p in paths]

offsets = {p:wl-p for (p, wl) in zip(positions, wls)}
offsets = dict(sorted(offsets.items()))

px = np.arange(len(offsets[600]))

fits = {}
for position, offset in offsets.items():
	fits[position] = np.polyfit(px, offset, 2)

fit_params = {position: fit for position, fit in fits.items()}


#%%

for p,wl in offsets.items():
	plt.plot(wl, label=p)
plt.legend()
plt.show()

# %%