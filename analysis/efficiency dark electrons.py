# %%
from curses import noqiflush
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
from networkx import power
import numpy as np
from math import pi
from scipy.constants import h, c, k
from labellines import labelLines
from glob import glob
import scipy as sp
from scipy.interpolate import interp1d
from uncertainties import ufloat
from uncertainties.unumpy import uarray
import uncertainties.unumpy as unp

from model import run_monte_carlo_simulation, SimulationParameters

import os

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# ===== MEASUREMENT PROCESSING FUNCTIONS =====

def load_spectrum_data(filepath):
    """Load and preprocess raw spectrum data"""
    data = np.loadtxt(filepath)
    wl = data[:, 0]
    counts = data[:, 1] - np.min(data[:, 1])  # Remove baseline
    
    counts *= 10  # Gain correction to photons
    counts = uarray(counts, np.sqrt(counts))

    counts /= 2  # Convert to counts/s
    
    # Convert to spectral density (counts/s/nm)
    spectral_counts = np.diff(wl) * counts[:-1]
    wl_centers = wl[:-1] + np.diff(wl)/2
    
    return wl_centers, spectral_counts

def load_efficiency_curve(filepath, k=0.75):
    """Load and calculate combined system efficiency"""
    camera_data = np.loadtxt(filepath)
    wl_cam = camera_data[:, 0]
    qe_cam = camera_data[:, 1]
    
    def blaze_function(wavelength, center=500, k=k):
        """Grating blaze function"""
        return np.sin(np.pi*k*(1-center/wavelength))**2 / (np.pi*k*(1-center/wavelength))**2
    
    # Calculate combined efficiency
    blaze_eff = blaze_function(wl_cam)
    combined_eff = blaze_eff * qe_cam
    combined_eff *= 0.6   # Low pass filter efficiency
    
    # Create interpolation function
    efficiency_func = sp.interpolate.interp1d(wl_cam, combined_eff, 
                                            bounds_error=False, fill_value=0)
    return efficiency_func

def counts_to_power_density(wavelength, counts):
    # Convert to power (W/nm)
    wavelength = wavelength * 1e-9  # Convert to meters
    energy_per_photon = h * c / wavelength  # Joules per photon
    power_spectrum = counts * energy_per_photon  # W/nm
    
    return power_spectrum

def apply_efficiency_correction(wavelength, counts, efficiency_func):
    # Get efficiency values
    efficiency = efficiency_func(wavelength)
    
    # Create mask for reliable efficiency data
    eff_mask = efficiency > 0.3 * np.max(efficiency)
    wl_masked = wavelength[eff_mask]
    counts_masked = counts[eff_mask]
    eff_masked = efficiency[eff_mask]
    
    # Correct for efficiency
    corrected_counts = counts_masked / (eff_masked / 100)
    
    # Convert to power (W/nm)
    wl_m = wl_masked * 1e-9  # Convert to meters
    energy_per_photon = h * c / wl_m  # Joules per photon
    power_spectrum = corrected_counts * energy_per_photon  # W/nm
    
    return wl_masked, power_spectrum

def mask_harmonics(wavelength, power, harmonic_wavelengths, mask_width=6):
    """Mask out harmonic peaks"""
    power_masked = power.copy()
    
    for harmonic_wl in harmonic_wavelengths:
        harmonic_mask = (wavelength < harmonic_wl - mask_width) | \
                       (wavelength > harmonic_wl + mask_width)
        power_masked = np.where(harmonic_mask, power_masked, np.nan)
    
    return power_masked

# ===== MODEL CALCULATION =====

mc_results = run_monte_carlo_simulation(sim_params=SimulationParameters(n_monte_carlo=10))

# ===== MEASUREMENT PROCESSING =====

# File paths
data_files = sorted(glob("../measurement/2025-05-05/003 thermal*.asc"))
efficiency_file = "../measurement/2025-04-03/QEcurve.dat"

wavelength_meas, counts = load_spectrum_data(data_files[0])
counts[wavelength_meas>900]=np.nan
counts = mask_harmonics(wavelength_meas, counts, [1032/2, 1032* 2/3, 1032/3], mask_width=10)
power_meas = counts_to_power_density(wavelength_meas, counts)

# l = plt.plot(wavelength_meas, unp.nominal_values(counts))
# plt.fill_between(wavelength_meas,
#     (unp.nominal_values(counts) - unp.std_devs(counts)),
#     (unp.nominal_values(counts) + unp.std_devs(counts)),
#     color = l[0].get_color(), alpha=0.5
# )
# plt.show()
#%%

# ===== Efficiency =====
def norm(x): return x/np.nanmax(x)

fig, axs = plt.subplots(2, 1, sharex=True)

# plot the model
power_spectrum_model_nW = mc_results['spectrum_mean'] * 1e-9  # Convert to W/nm
wavelength_model = mc_results['wavelength']

l = axs[0].plot(wavelength_model, power_spectrum_model_nW, label="model")[0]
    
# Uncertainty bands
axs[0].fill_between(wavelength_model,
                mc_results['spectrum_percentiles'][0] * 1e-9,
                mc_results['spectrum_percentiles'][1] * 1e-9,
                alpha=0.3, color=l.get_color())


model = interp1d(wavelength_model, power_spectrum_model_nW, bounds_error=False)
mask = np.logical_and(wavelength_meas>500, wavelength_meas<700)
# mask = wavelength_meas>500
scale = np.nanmedian(model(wavelength_meas[mask])/unp.nominal_values(power_meas[mask]))

l = axs[0].plot(wavelength_meas, unp.nominal_values(power_meas)*scale, label=fr"measured $\times {scale:g}$")
axs[0].fill_between(wavelength_meas,
    (unp.nominal_values(power_meas) - unp.std_devs(power_meas))*scale,
    (unp.nominal_values(power_meas) + unp.std_devs(power_meas))*scale,
    color = l[0].get_color(), alpha=0.3
)
axs[0].legend()
axs[0].set_ylabel("Power Density\n (W/nm)")
axs[0].set_ylim(0, None)


l = axs[1].plot(wavelength_model, norm(load_efficiency_curve(efficiency_file)(wavelength_model)), label="expected")
axs[1].fill_between(wavelength_model,
    norm(load_efficiency_curve(efficiency_file, k=0.5)(wavelength_model)),    
    norm(load_efficiency_curve(efficiency_file, k=1)(wavelength_model)),
    color = l[0].get_color(), alpha=0.3
)

modelStd = interp1d(wavelength_model, mc_results['spectrum_std'] * 1e-9, bounds_error=False)
eff_meas =  unp.nominal_values(power_meas) / model(wavelength_meas)
eff_meas_lower =  unp.nominal_values(power_meas) / (model(wavelength_meas) + modelStd(wavelength_meas))
eff_meas_upper =  unp.nominal_values(power_meas) / (model(wavelength_meas) - modelStd(wavelength_meas))
scale = np.nanmax(eff_meas)
l = axs[1].plot(wavelength_meas, eff_meas/scale, label="measured")
axs[1].fill_between(wavelength_meas,
    eff_meas_lower/np.nanmax(eff_meas_lower), eff_meas_upper/np.nanmax(eff_meas_upper),
    color = l[0].get_color(), alpha=0.3
)

axs[1].set_xlim(200, 1000)
# axs[1].set_ylim(0, 1.4)
axs[1].legend()
axs[1].set_xlabel("Wavelength (nm)")
axs[1].set_ylabel("effective\n Efficiency")
plt.savefig("figures/efficiency de.pdf")
plt.show()
# plt.yscale("log")
# %%
