#%%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
import xarray as xr
from glob import glob
import yaml
import os
from scipy.optimize import curve_fit

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# %%
with open("../measurement/2025-07-09/dark_noise.yaml", "r") as f:
    result = yaml.load(f, Loader=yaml.UnsafeLoader)

# %%
# --- Constants and Model ---
k = 8.617333262145e-5  # Boltzmann constant in eV/K

def dark_noise_model(T, t, I0, E, B):
    # T in Celsius, convert to Kelvin
    T_K = T + 273.15
    return I0 * np.exp(-E / (k * T_K)) * t + B
    # return np.sqrt(I0 * np.exp(-E / (k * T_K)) * t + B**2)

def dark_noise_model_fit(X, I0, E, B):
    T, t = X
    return dark_noise_model(T, t, I0, E, B)

# --- Prepare Data ---
# temperature sweep
temps = np.array(list(result["temperature_sweep"].keys()))
# counts_temp = np.array(list(result["temperature_sweep"].values())).mean(axis=(1, 2))
counts_temp = np.array(list(result["temperature_sweep"].values())).std(axis=(1, 2))
exposure_temp = 10

# exposure sweep
exposures = np.array(list(result["exposure_sweep"].keys()))
# counts_exposure = np.array(list(result["exposure_sweep"].values())).mean(axis=(1, 2))
counts_exposure = np.array(list(result["exposure_sweep"].values())).std(axis=(1, 2))
temperature_exp = -20

# Prepare data for fitting
T_fit = np.concatenate([temps, np.full_like(exposures, temperature_exp)])
t_fit = np.concatenate([np.full_like(temps, exposure_temp), exposures])
counts_fit = np.concatenate([counts_temp, counts_exposure])

# --- Fit Model ---
p0 = [5e17, 0.9, 1.5]  # Initial guess for I0, E, B
popt, pcov = curve_fit(
    dark_noise_model_fit,
    (T_fit, t_fit),
    counts_fit,
    p0 = p0,
    maxfev=10000,
)
I0_fit, E_fit, B_fit = p0
I0_fit, E_fit, B_fit = popt
perr = np.sqrt(np.diag(pcov))
print(f"Fit results: I0={I0_fit:.2e} ± {perr[0]:.2e}, E={E_fit:.3f} ± {perr[1]:.3f} eV, B={B_fit:.2f} ± {perr[2]:.2f}")

# --- Plotting ---
fig, axes = plt.subplots(1, 2, sharey=True)

# Temperature sweep plot
ax = axes[0]
ax.set_ylabel(r"Dark noise std (e$^-$)")
# ax.set_yscale("log")
ax.set_xlabel("Temperature (°C)")
ax.plot(temps, counts_temp, "o", label="Data")
T_plot = np.linspace(temps.min(), temps.max(), 100)
counts_model_temp = dark_noise_model(T_plot, exposure_temp, I0_fit, E_fit, B_fit)
ax.plot(T_plot, counts_model_temp, "-", label="Fit")
ax.axvline(temperature_exp, color="k", linestyle="--", linewidth=1)
ax.axhline(B_fit, color="k", linewidth=1)
# ax.legend()

# Exposure sweep plot
ax = axes[1]
# ax.set_xscale("log")
ax.set_xlabel("Exposure time (s)")
ax.tick_params(axis='y', left=False, labelleft=False)
ax.plot(exposures, counts_exposure, "o", label="Data")
t_plot = np.linspace(exposures.min(), exposures.max(), 1000)
counts_model_exp = dark_noise_model(temperature_exp, t_plot, I0_fit, E_fit, B_fit)
ax.plot(t_plot, counts_model_exp, "-", label="Fit")
# ax.set_xscale("log")
ax.axvline(exposure_temp, color="k", linestyle="--", linewidth=1, label="Cross Section")
ax.axhline(B_fit, color="k", linewidth=1, label=r"$\sigma_{read}$")
ax.legend()

# plt.ylim(1e0, 1e2)
plt.tight_layout()
plt.savefig("figures/dark_noise.pdf")
plt.show()
# %%
# --- Plot mean noise counts ---

fig, axes = plt.subplots(1, 2, sharey=True)

# Temperature sweep mean plot
ax = axes[0]
ax.set_ylabel("Dark noise mean (counts)")
ax.set_xlabel("Temperature (°C)")
counts_temp_mean = np.array(list(result["temperature_sweep"].values())).mean(axis=(1, 2))
ax.plot(temps, counts_temp_mean, "o", label="Mean Data")
counts_model_temp_mean = dark_noise_model(T_plot, exposure_temp, I0_fit, E_fit, B_fit)
# ax.legend()

# Exposure sweep mean plot
ax = axes[1]
ax.set_xlabel("Exposure time (s)")
ax.tick_params(axis='y', left=False, labelleft=False)
counts_exposure_mean = np.array(list(result["exposure_sweep"].values())).mean(axis=(1, 2))
ax.plot(exposures, counts_exposure_mean, "o", label="Mean Data")
counts_model_exp_mean = dark_noise_model(temperature_exp, t_plot, I0_fit, E_fit, B_fit)
ax.legend()

plt.tight_layout()
plt.show()
