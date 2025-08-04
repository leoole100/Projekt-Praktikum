# %%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
from glob import glob
import scipy as sp
from scipy.constants import h, c
import os

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def load_spectrum_data(filepath):
    """Load and preprocess raw spectrum data"""
    data = np.loadtxt(filepath)
    wl = data[:, 0]
    counts = data[:, 1] - np.min(data[:, 1])  # Remove baseline
    
    # Convert to counts/s and apply gain correction
    counts /= 2  # Convert to counts/s
    counts *= 10  # Gain correction to photons
    
    # Convert to spectral density (counts/s/nm)
    spectral_counts = np.diff(wl) * counts[:-1]
    wl_centers = wl[:-1] + np.diff(wl)/2
    
    return wl_centers, spectral_counts

def load_efficiency_curve(filepath):
    """Load and calculate combined system efficiency"""
    camera_data = np.loadtxt(filepath)
    wl_cam = camera_data[:, 0]
    qe_cam = camera_data[:, 1]
    
    def blaze_function(wavelength, center=500, k=0.75):
        """Grating blaze function"""
        return np.sin(np.pi*k*(1-center/wavelength))**2 / (np.pi*k*(1-center/wavelength))**2
    
    # Calculate combined efficiency
    blaze_eff = blaze_function(wl_cam)
    combined_eff = blaze_eff * qe_cam
    combined_eff *= 0.6   # Low pass filter efficiency
    combined_eff *= 0.053 # Collection efficiency  
    combined_eff *= 0.04  # Fiber coupling efficiency
    
    # Create interpolation function
    efficiency_func = sp.interpolate.interp1d(wl_cam, combined_eff, 
                                            bounds_error=False, fill_value=0)
    return efficiency_func

def apply_efficiency_correction(wavelength, counts, efficiency_func):
    """Apply efficiency correction and convert to power units"""
    # Get efficiency values
    efficiency = efficiency_func(wavelength)
    
    # Create mask for reliable efficiency data
    eff_mask = efficiency > 0.2 * np.max(efficiency)
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

# Main processing pipeline
def process_thermal_spectrum(data_file, efficiency_file):
    """Complete processing pipeline for thermal spectrum data"""
    
    # Load raw data
    wavelength, raw_counts = load_spectrum_data(data_file)
    
    # Load efficiency correction
    efficiency_func = load_efficiency_curve(efficiency_file)
    
    # Apply corrections
    wl_corrected, power_corrected = apply_efficiency_correction(
        wavelength, raw_counts, efficiency_func)
    
    # Mask harmonics (e.g., laser harmonics)
    power_masked = mask_harmonics(wl_corrected, power_corrected, [686])
    
    return wl_corrected, power_corrected, power_masked

# Execute processing
if __name__ == "__main__":
    # File paths
    data_files = sorted(glob("../measurement/2025-05-05/003 thermal*.asc"))
    efficiency_file = "../measurement/2025-04-03/QEcurve.dat"
    
    # Process data
    wavelength, power_spectrum, power_masked = process_thermal_spectrum(
        data_files[0], efficiency_file)
    
    # Plot results
    # Masked spectrum (harmonics removed)
    plt.plot(wavelength, power_masked)
    plt.ylabel("Power (W/nm)")
    plt.xlabel("Wavelength (nm)")
    
    plt.tight_layout()
    plt.show()
