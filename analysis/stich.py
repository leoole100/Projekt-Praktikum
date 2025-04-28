#%%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
import yaml
from glob import glob

# %%
# import the data
p = glob("../measurement/2025-04-28/001 *.yaml")
data = []  
for file in p:
	with open(file, 'r') as f:
		data.append(yaml.safe_load(f))
data = sorted(data, key=lambda x: x["metadata"]["monochromator"]["center"])

# get the wavelength range
step = min(np.min(np.diff(d["spectrum"]["wavelength"])) for d in data)
wl = np.arange(
	min(np.min(d["spectrum"]["wavelength"]) for d in data),
	max(np.max(d["spectrum"]["wavelength"]) for d in data) + step/2,
	step
)

interpolated = []
for d in data:
    w = d["spectrum"]["wavelength"]
    i = d["spectrum"]["counts"]

    # Create a cubic interpolation function
    interp = sp.interpolate.interp1d(
        w, i,
        kind='cubic',
        bounds_error=False,
        fill_value=np.nan
    )

    i_interp = interp(wl)
    interpolated.append(i_interp)

interpolated = np.vstack(interpolated)
counts_mean = np.nanmean(interpolated, axis=0)
counts_std = np.nanstd(interpolated, axis=0)

plt.figure()
plt.plot(wl, counts_mean, color="k", label="Combined")
plt.fill_between(
    wl,
    counts_mean - counts_std,
    counts_mean + counts_std,
    color='gray',
	label="std"
)
for d in data:
	s = d["spectrum"]
	plt.plot(
		s["wavelength"], s["counts"] - sp.interpolate.interp1d(wl,counts_mean)(s["wavelength"]), 
		# label=f"{round(d["metadata"]["monochromator"]["center"])} nm",
		alpha=0.7
	)
plt.legend()
plt.xlabel(r"$\lambda$ (nm)")
plt.ylabel("difference from mean")
plt.savefig("figures/stich.pdf")
plt.show()
