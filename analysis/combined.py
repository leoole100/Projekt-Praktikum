# %%
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
import numpy as np
from math import pi
from scipy.constants import h, c, k
from labellines import labelLines
from glob import glob
import scipy as sp

import os

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# ===== MODEL PARAMETERS =====

# Laser Pulse Parameters
P_avg = 0.3                    # Average power (W)
f_rep = 40e3                   # Repetition rate (Hz)
E_pulse = P_avg / f_rep        # Pulse energy (J)
tau_fwhm = 250e-15             # FWHM pulse duration (s)
sigma = tau_fwhm / 2.355       # Gaussian std dev (s)
spot_diameter = 50e-6          # Laser spot diameter (m)
r = spot_diameter / 2          # Laser spot radius (m)

# Volume and Material Constants
d = 200e-9                     # Optical absorption depth (m)
V = pi * r**2 * d              # Excited volume (m³)
A = pi * r**2                  # Surface area (m²)
M = 0.012                      # Molar mass (kg/mol)
rho = 2260                     # Density (kg/m³)
Vm = M / rho                   # Molar volume (m³/mol)

# Coupling and Environment
g = 0.3e-12                    # Electron-lattice coupling time (s)
T_room = 294                   # Room temperature (K)

# ===== MODEL FUNCTIONS =====

def gauss_pulse(t, sigma, t0=0):
    """Gaussian laser pulse"""
    return np.exp(-((t - t0)**2) / (2 * sigma**2))

def c_e(T):
    """Electronic heat capacity"""
    return 1e-6 * 12.8 * T * (1 + 1.16e-3 * T + 2.6e-7 * T**2)

def planck_spectrum(wavelength, T):
    """Planck's law for blackbody radiation [W/(m³·sr)]"""
    wavelength_m = wavelength * 1e-9
    B = (2 * h * c**2 / wavelength_m**5) / (np.exp(h * c / (wavelength_m * k * T)) - 1)
    return B

# ===== MEASUREMENT PROCESSING FUNCTIONS =====

def load_spectrum_data(filepath):
    """Load and preprocess raw spectrum data"""
    data = np.loadtxt(filepath)
    wl = data[:, 0]
    counts = data[:, 1] - np.min(data[:, 1])  # Remove baseline
    
    # Convert to counts/s and apply gain correction
    counts /= 2  # Convert to counts/s
    counts *= 10  # Gain correction to photons
    
    counts /= 0.6 # low pass filter efficiency
    counts *= 8 # collection angle 4 f^2 / r^2


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
    
    # Create interpolation function
    efficiency_func = sp.interpolate.interp1d(wl_cam, combined_eff, 
                                            bounds_error=False, fill_value=0)
    return efficiency_func

def apply_efficiency_correction(wavelength, counts, efficiency_func):
    """Apply efficiency correction and convert to power units"""
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

# ===== MODEL CALCULATION =====

# Time evolution calculation
t = np.linspace(-0.5, 1, 1000) * 1e-12
dt = t[1] - t[0]

# Initialize temperature array
T_e = np.full(len(t), T_room, dtype=np.float64)

# Laser source array (normalized to W/m³)
S = gauss_pulse(t, sigma)
S = S * E_pulse / (V * np.sum(S * dt))  # S has units W/m³

# Time evolution using forward Euler
for n in range(len(t) - 1):
    T_e[n + 1] = T_e[n]
    # Laser heating term
    T_e[n + 1] += dt * Vm / c_e(T_e[n]) * S[n]
    # Cooling term  
    T_e[n + 1] += -dt * (T_e[n] - T_room) / g

# Spectrum calculation
wavelength_model = np.linspace(200, 1000, 500)  # nm
solid_angle = 2 * pi  # steradians (hemisphere)

power_spectrum_model = np.zeros_like(wavelength_model)

for i, wl in enumerate(wavelength_model):
    # Integrate over the hot electron cooling time
    for j, temp in enumerate(T_e):
        # Power radiated = Planck function × Area × solid angle × repetition rate
        radiance = planck_spectrum(wl, temp)  # W/(m³·sr)
        power_density = radiance * A * solid_angle  # W/m (power per unit wavelength)
        power_spectrum_model[i] += power_density * dt * f_rep  # Average power W/m

# Convert to more convenient units
power_spectrum_model_nW = power_spectrum_model * 1e-9  # Convert to W/nm

# ===== MEASUREMENT PROCESSING =====

# File paths
data_files = sorted(glob("../measurement/2025-05-05/003 thermal*.asc"))
efficiency_file = "../measurement/2025-04-03/QEcurve.dat"

# Process measurement data
try:
    wavelength_meas, power_spectrum_meas, power_masked_meas = process_thermal_spectrum(
        data_files[0], efficiency_file)
    measurement_available = True
    print(f"Loaded measurement data from: {data_files[0]}")
except (FileNotFoundError, IndexError) as e:
    print(f"Warning: Could not load measurement data - {e}")
    measurement_available = False

# ===== PLOTTING =====

# Convert time to ps for plotting
t_ps = t / 1e-12

# Temperature dynamics plot
fig = plt.figure()

# Plot model spectrum
plt.plot(wavelength_model, power_spectrum_model_nW, 'k-', label='Theoretical Model')


from scipy.interpolate import interp1d
model = interp1d(wavelength_model, power_spectrum_model_nW)
scale = np.nanmedian(model(wavelength_meas)/power_masked_meas)

# Plot measurement if available
if measurement_available:
    plt.plot(wavelength_meas, scale*power_masked_meas, label=rf'Measurement $\times {scale:.0f}$')

plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Average Radiated Power (W/nm)')
plt.ylim(0, None)


plt.tight_layout()
plt.savefig("figures/model_vs_measurement.pdf")
plt.show()

# ===== RESULTS =====

# Calculate and print model results
max_temp = np.max(T_e)
max_temp_time = t_ps[np.argmax(T_e)]
fluence = E_pulse / (pi * r**2)
peak_wavelength_model = wavelength_model[np.argmax(power_spectrum_model_nW)]
total_power_model = np.trapz(power_spectrum_model, wavelength_model * 1e-9)

print("\n===== MODEL RESULTS =====")
print(f"Pulse energy: {E_pulse*1e6:.1f} μJ")
print(f"Fluence: {fluence:.2f} J/m²")
print(f"Maximum electron temperature: {max_temp:.0f} K")
print(f"Time of maximum temperature: {max_temp_time:.3f} ps")
print(f"Temperature rise: {max_temp - T_room:.0f} K")
print(f"Peak wavelength of thermal radiation: {peak_wavelength_model:.0f} nm")
print(f"Total average radiated power: {total_power_model*1e12:.1f} pW")
print(f"Peak spectral power: {np.max(power_spectrum_model_nW):.2f} nW/nm")

# Print measurement results if available
if measurement_available:
    # Remove NaN values for analysis
    valid_mask = ~np.isnan(power_masked_meas)
    if np.any(valid_mask):
        peak_wavelength_meas = wavelength_meas[valid_mask][np.argmax(power_masked_meas[valid_mask])]
        total_power_meas = np.trapz(power_masked_meas[valid_mask], wavelength_meas[valid_mask] * 1e-9)
        peak_power_meas = np.nanmax(power_masked_meas)
        
        print("\n===== MEASUREMENT RESULTS =====")
        print(f"Peak wavelength: {peak_wavelength_meas:.0f} nm")
        print(f"Total measured power: {total_power_meas*1e12:.1f} pW")
        print(f"Peak spectral power: {peak_power_meas:.2f} nW/nm")
        
        print(f"\n===== COMPARISON =====")
        print(f"Peak wavelength difference: {peak_wavelength_model - peak_wavelength_meas:.0f} nm")
        print(f"Power ratio (model/measurement): {total_power_model/total_power_meas:.2f}")
else:
    print("\nMeasurement data not available for comparison.")