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

wavelength_meas, counts = load_spectrum_data(data_files[0])
counts[wavelength_meas>900]=np.nan
counts = mask_harmonics(wavelength_meas, counts, [1032/2, 1032* 2/3, 1032/3], mask_width=10)
power_meas = counts_to_power_density(wavelength_meas, counts)

l = plt.plot(wavelength_meas, unp.nominal_values(counts))
plt.fill_between(wavelength_meas,
    (unp.nominal_values(counts) - unp.std_devs(counts)),
    (unp.nominal_values(counts) + unp.std_devs(counts)),
    color = l[0].get_color(), alpha=0.5
)
plt.show()

# ===== Efficiency =====
def norm(x): return x/np.nanmax(x)

fig, axs = plt.subplots(2, 1, sharex=True)

model = interp1d(wavelength_model, power_spectrum_model_nW, bounds_error=False)
mask = np.logical_and(wavelength_meas>500, wavelength_meas<700)
scale = np.nanmedian(model(wavelength_meas[mask])/unp.nominal_values(power_meas[mask]))
l = axs[0].plot(wavelength_meas, unp.nominal_values(power_meas)*scale, label=fr"measured $\times {scale:g}$")
axs[0].fill_between(wavelength_meas,
    (unp.nominal_values(power_meas) - unp.std_devs(power_meas))*scale,
    (unp.nominal_values(power_meas) + unp.std_devs(power_meas))*scale,
    color = l[0].get_color(), alpha=0.5
)
axs[0].plot(wavelength_model, power_spectrum_model_nW, label="model")
axs[0].legend()
axs[0].set_ylabel("Power Density\n (W/nm)")


eff_meas =  power_meas / model(wavelength_meas)
l = axs[1].plot(wavelength_meas, unp.nominal_values(norm(eff_meas)), label="measured")
axs[1].fill_between(wavelength_meas,
    unp.nominal_values(norm(eff_meas)) - unp.std_devs(norm(eff_meas)),
    unp.nominal_values(norm(eff_meas)) + unp.std_devs(norm(eff_meas)),
    color = l[0].get_color(), alpha=0.5
)

l = axs[1].plot(wavelength_model, norm(load_efficiency_curve(efficiency_file)(wavelength_model)), label="expected")
axs[1].fill_between(wavelength_model,
    norm(load_efficiency_curve(efficiency_file, k=0.5)(wavelength_model)),    
    norm(load_efficiency_curve(efficiency_file, k=1)(wavelength_model)),
    color = l[0].get_color(), alpha=0.5
)

axs[1].legend()
axs[1].set_xlabel("Wavelength (nm)")
axs[1].set_ylabel("effective\n Efficiency")
plt.savefig("figures/efficiency de.pdf")
plt.show()
# plt.yscale("log")
# %%
