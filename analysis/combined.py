# %%
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

from model import HotElectronSim, planck

import os

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# ===== MEASUREMENT PROCESSING FUNCTIONS =====

def spectrum_data(filepath):
    """Load and preprocess raw spectrum data"""
    data = np.loadtxt(filepath)
    wl = data[:, 0]
    counts = data[:, 1] - np.min(data[:, 1])  # Remove baseline
    
    counts *= 10  # Gain correction to photons
    # counts = uarray(counts, np.sqrt(counts))

    counts /= 2  # Convert to counts/s
    
    # Convert to spectral density (counts/s/nm)
    spectral_counts = np.diff(wl) * counts[:-1]
    wl_centers = wl[:-1] + np.diff(wl)/2

    spectral_counts /= 0.6 # filter transmission
    spectral_counts /= 0.5 # collection part r^2 / 2f^2 for half complete half sphere

    return wl_centers, spectral_counts

def efficiency_curve(k=1.3):
    """Load and calculate combined system efficiency"""
    camera_data = np.loadtxt("../measurement/2025-04-03/QEcurve.dat")
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
    efficiency_func = sp.interpolate.interp1d(
        wl_cam, combined_eff,bounds_error=False, fill_value=0
    )
    return efficiency_func

def counts_to_power_density(wavelength, counts):
    # Convert to power (W/nm)
    energy_per_photon = 1240 / wavelength  # Joules per photon
    return counts * energy_per_photon  # W/nm

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


# File paths
data_files = sorted(glob("../measurement/2025-05-05/003 thermal*.asc"))

# ========= Raw ========= 
wavelength_meas, raw_power = spectrum_data(data_files[0])
raw_power = counts_to_power_density(wavelength_meas, raw_power)
wavelength_meas = wavelength_meas[200:1800]
raw_power = raw_power[200:1800]
raw_power /= raw_power.max()
plt.plot(
    wavelength_meas, raw_power,
    label="raw"
)

# ====== Corrected ====== 
power = raw_power/efficiency_curve()(wavelength_meas)
power /= power[100:500].max()
plt.plot(
    wavelength_meas, power,
    label="corrected"
)

# ========= Sim =========
sim = HotElectronSim(
    F_exc=400
)
sim_spec = sim.spectrum()
sim_spec /= sim_spec.max()

plt.plot(
    sim.wavelength_nm, sim_spec,
    label="sim"
)

# ==== Sim Corrected ====
sim_corrected = sim_spec*efficiency_curve()(sim.wavelength_nm)
sim_corrected /= sim_corrected.max()
plt.plot(
    sim.wavelength_nm, sim_corrected,
    label="sim corr"
)

def ev_nm(x): return 1240/x
plt.gca().secondary_xaxis('top', functions=(ev_nm, ev_nm)).\
set_xlabel("Photon Energy (eV)")

plt.ylim(0, None)
plt.yticks([0])
plt.xlabel("wavelength (nm)")
plt.legend()
plt.show()