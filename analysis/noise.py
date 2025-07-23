# %%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
import xarray as xr
from glob import glob
import yaml
import os
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
import matplotlib.cm as cm

# Change to script directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# %%

# Load data
paths = sorted(glob("../measurement/2025-07-23/005 *.npy"))
times = np.array([float(p.split(" ")[-1][:-5]) for p in paths])  # exposure times
counts = np.array([np.load(p) for p in paths])  # shape: (times, runs, 2, pixels)
wl = counts[0, 0, 0]  # wavelength array, shape: (pixels,)
counts = counts[:, :, 1, :]  # shape: (times, runs, pixels)

# Compute mean and variance across runs
mean = counts.mean(1)          # shape: (times, pixels)
var = counts.std(1)**2         # shape: (times, pixels)

# Build mask for valid data points
mask = (mean > 0) & (mean < 1.6e4) & (var < 2e3)

# Plot signal vs variance per pixel
plt.figure()
colors = cm.viridis(np.linspace(0, 1, mean.shape[1]))
for i in range(mean.shape[1]):
    m = mask[:, i]
    if np.any(m):
        plt.plot(mean[m, i], var[m, i], marker='o', linestyle='-', color=colors[i], alpha=0.1)
plt.xlabel("Signal")
plt.ylabel("Variance")
plt.grid(True)
plt.show()

# --- Relative QE estimation ---
n_times, n_pixels = mean.shape
slopes = np.full(n_pixels, np.nan)

# Fit var = a * t per pixel
for i in range(n_pixels):
    y = var[:, i]
    x = times
    m = mask[:, i]
    if np.sum(m) >= 3:
        slope, *_ = linregress(x[m], y[m])
        if slope > 0:
            slopes[i] = slope

# Choose a reference signal S_k from the last exposure time
k = -1  # use longest exposure time
signal_ref = mean[k, :]  # shape: (pixels,)

# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    qe_rel = (slopes * times[k]) / signal_ref

# Normalize and smooth
qe_rel /= np.nanmax(qe_rel)
qe_smooth = gaussian_filter1d(qe_rel, sigma=20)

# get the camera manufacturer curve in
expected = np.load("expected_efficiency.npy")
expected = xr.DataArray(expected[1], {"wavelength":expected[0]}).sortby("wavelength")
expected = expected.sel(wavelength=slice(wl.min(), wl.max()))

# Plot estimated QE
plt.plot(wl, qe_rel, label="Estimated QE", alpha=0.4)
plt.plot(wl, qe_smooth, label="Smoothed QE", linewidth=2, color='black')
plt.plot(expected.wavelength, expected, label="expected")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Relative QE")
plt.legend()
plt.show()
