# %%
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
import numpy as np
from math import pi
from scipy.constants import h, c, k
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from glob import glob
import scipy as sp
import os

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# ===== BASE PARAMETERS =====
BASE_PARAMS = {
    'P_avg': 0.3,                    # Average power (W)
    'f_rep': 40e3,                   # Repetition rate (Hz)  
    'tau_fwhm': 250e-15,             # FWHM pulse duration (s)
    'spot_diameter': 50e-6,         # Laser spot diameter (m)
    'd': 200e-9,                     # Optical absorption depth (m)
    'M': 0.012,                      # Molar mass (kg/mol)
    'rho': 2260,                     # Density (kg/m³)
    'g': 0.3e-12,                    # Electron-lattice coupling time (s)
    'T_room': 294,                   # Room temperature (K)
}

# Global variables
_iteration_count = 0

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

def calculate_model_spectrum(params, wavelength_grid, fast_mode=True):
    """Calculate model spectrum"""
    # Extract parameters
    P_avg = params.get('P_avg', BASE_PARAMS['P_avg'])
    f_rep = params.get('f_rep', BASE_PARAMS['f_rep'])
    tau_fwhm = params.get('tau_fwhm', BASE_PARAMS['tau_fwhm'])
    spot_diameter = params.get('spot_diameter', BASE_PARAMS['spot_diameter'])
    d = params.get('d', BASE_PARAMS['d'])
    M = params.get('M', BASE_PARAMS['M'])
    rho = params.get('rho', BASE_PARAMS['rho'])
    g = params.get('g', BASE_PARAMS['g'])
    T_room = params.get('T_room', BASE_PARAMS['T_room'])
    
    # Derived parameters
    E_pulse = P_avg / f_rep
    sigma = tau_fwhm / 2.355
    r = spot_diameter / 2
    V = pi * r**2 * d
    A = pi * r**2
    Vm = M / rho
    
    # Time grid - adjust resolution based on mode
    n_time = 150 if fast_mode else 500
    t = np.linspace(-0.5, 1, n_time) * 1e-12
    dt = t[1] - t[0]
    
    # Temperature evolution
    T_e = np.full(len(t), T_room, dtype=np.float64)
    S = gauss_pulse(t, sigma) * E_pulse / (V * np.sum(gauss_pulse(t, sigma)) * dt)
    
    for n in range(len(t) - 1):
        heating = dt * Vm / c_e(T_e[n]) * S[n]
        cooling = -dt * (T_e[n] - T_room) / g
        T_e[n + 1] = T_e[n] + heating + cooling
    
    # Spectrum calculation
    solid_angle = 2 * pi
    power_spectrum = np.zeros_like(wavelength_grid)
    
    # Sample temperature points for speed in fast mode
    if fast_mode:
        sample_indices = slice(0, len(T_e), 3)
        T_sampled = T_e[sample_indices]
        weight = 3
    else:
        T_sampled = T_e
        weight = 1
    
    for i, wl in enumerate(wavelength_grid):
        radiance = planck_spectrum(wl, T_sampled)
        power_density = radiance * A * solid_angle
        power_spectrum[i] = np.sum(power_density) * dt * f_rep * weight
    
    return power_spectrum * 1e-9, T_e

# ===== MEASUREMENT PROCESSING =====

def load_spectrum_data(filepath):
    """Load and preprocess raw spectrum data"""
    data = np.loadtxt(filepath)
    wl = data[:, 0]
    counts = data[:, 1] - np.min(data[:, 1]) # remove bias
    counts /= 2  # Convert to counts/s
    counts *= 10  # Gain correction
    spectral_counts = np.diff(wl) * counts[:-1]
    wl_centers = wl[:-1] + np.diff(wl)/2
    return wl_centers, spectral_counts

def load_efficiency_curve(filepath):
    """Load and calculate combined system efficiency"""
    camera_data = np.loadtxt(filepath)
    wl_cam = camera_data[:, 0]
    qe_cam = camera_data[:, 1]
    
    def blaze_function(wavelength, center=500, k=0.75):
        return np.sin(np.pi*k*(1-center/wavelength))**2 / (np.pi*k*(1-center/wavelength))**2
    
    blaze_eff = blaze_function(wl_cam)
    combined_eff = blaze_eff * qe_cam 
    combined_eff *= 0.6 # filter
    
    return sp.interpolate.interp1d(wl_cam, combined_eff, bounds_error=False, fill_value=0)

def apply_efficiency_correction(wavelength, counts, efficiency_func):
    """Apply efficiency correction and convert to power units"""
    efficiency = efficiency_func(wavelength)
    eff_mask = efficiency > 0.4 * np.max(efficiency)
    wl_masked = wavelength[eff_mask]
    counts_masked = counts[eff_mask]
    eff_masked = efficiency[eff_mask]
    
    corrected_counts = counts_masked / (eff_masked / 100)
    wl_m = wl_masked * 1e-9
    energy_per_photon = h * c / wl_m
    power_spectrum = corrected_counts * energy_per_photon
    
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
    """Complete processing pipeline"""
    wavelength, raw_counts = load_spectrum_data(data_file)
    efficiency_func = load_efficiency_curve(efficiency_file)
    wl_corrected, power_corrected = apply_efficiency_correction(wavelength, raw_counts, efficiency_func)
    power_masked = mask_harmonics(wl_corrected, power_corrected, [686])
    return wl_corrected, power_corrected, power_masked

# ===== FITTING FUNCTION =====

def fit_model_to_measurement(wl_meas, power_meas, fit_params=['g', 'tau_fwhm']):
    """Fit model parameters to measurement data"""
    print(f"Fitting parameters: {fit_params}")
    
    # Setup wavelength grid based on measurement
    valid_mask = ~np.isnan(power_meas)
    wl_min = np.min(wl_meas[valid_mask])
    wl_max = np.max(wl_meas[valid_mask])
    wl_range = wl_max - wl_min
    wl_grid = np.linspace(wl_min - 0.1*wl_range, wl_max + 0.1*wl_range, 80)
    
    # Interpolate measurement to grid
    meas_interp = interp1d(wl_meas[valid_mask], power_meas[valid_mask], 
                          bounds_error=False, fill_value=0)
    power_meas_interp = meas_interp(wl_grid)
    
    # Parameter bounds
    initial_values = []
    for param in fit_params:
        initial_values.append(BASE_PARAMS[param])
    
    def objective(params_array):     
        # Update parameters
        params = BASE_PARAMS.copy()
        for i, name in enumerate(fit_params):
            params[name] = params_array[i]
        
        try:
            # Calculate model spectrum
            power_model, _ = calculate_model_spectrum(params, wl_grid, fast_mode=True)
            
            # Normalize and compare
            model_rms = np.sqrt(np.mean(power_model**2))
            meas_rms = np.sqrt(np.mean(power_meas_interp**2))
            
            model_normalized = power_model * (meas_rms / model_rms)
            chi_squared = np.sum((model_normalized - power_meas_interp)**2)
            
            return chi_squared
            
        except Exception:
            return 1e10
    
    # Optimize
    result = minimize(objective, initial_values, options={"maxiter":10000, "disp":True}, method='Nelder-Mead', tol=1e-20)
    
    # Create optimized parameters
    optimized_params = BASE_PARAMS.copy()
    for i, param in enumerate(fit_params):
        optimized_params[param] = result.x[i]
    
    print(f"Optimization completed after {result.nit} iterations")
    print(f"Final χ² = {result.fun:.3e}")
    
    return optimized_params, result

# ===== MAIN EXECUTION =====

# Load measurement data
data_files = sorted(glob("../measurement/2025-05-05/003 thermal*.asc"))
efficiency_file = "../measurement/2025-04-03/QEcurve.dat"

try:
    wavelength_meas, power_spectrum_meas, power_masked_meas = process_thermal_spectrum(
        data_files[0], efficiency_file)
    measurement_available = True
    print(f"Loaded measurement data from: {data_files[0]}")
except (FileNotFoundError, IndexError) as e:
    print(f"Warning: Could not load measurement data - {e}")
    measurement_available = False

# Setup wavelength grid for display
if measurement_available:
    valid_mask = ~np.isnan(power_masked_meas)
    wl_min = np.min(wavelength_meas[valid_mask])
    wl_max = np.max(wavelength_meas[valid_mask])
    wl_range = wl_max - wl_min
    wavelength_display = np.linspace(wl_min - 0.1*wl_range, wl_max + 0.1*wl_range, 300)
else:
    wavelength_display = np.linspace(200, 1000, 400)

# Calculate initial model
power_spectrum_initial, T_e_initial = calculate_model_spectrum(
    BASE_PARAMS, wavelength_display, fast_mode=False)

# Perform fitting if measurement available
print("\nStarting parameter optimization...")

# Choose parameters to fit
fit_params_list = ['g', 'tau_fwhm', 'd', 'spot_diameter']

optimized_params, fit_result = fit_model_to_measurement(
    wavelength_meas, power_masked_meas, fit_params_list)

# Calculate optimized model
power_spectrum_optimized, T_e_optimized = calculate_model_spectrum(
    optimized_params, wavelength_display, fast_mode=False)

# Print parameter changes
print("\nParameter changes:")
for param in fit_params_list:
    initial_val = BASE_PARAMS[param]
    final_val = optimized_params[param]
    change_percent = (final_val - initial_val) / initial_val * 100
    
    if param == 'g':
        print(f"  {param}: {initial_val*1e12:.2f} → {final_val*1e12:.2f} ps ({change_percent:+.1f}%)")
    elif param == 'tau_fwhm':
        print(f"  {param}: {initial_val*1e15:.0f} → {final_val*1e15:.0f} fs ({change_percent:+.1f}%)")
    else:
        print(f"  {param}: {initial_val:.2e} → {final_val:.2e} ({change_percent:+.1f}%)")

# ===== PLOTTING =====
plt.figure()


plt.plot(wavelength_meas, power_masked_meas, "k", label=f'corrected Measurement')

meas_interp = interp1d(wavelength_meas, power_masked_meas, fill_value=np.nan, bounds_error=False)


scale = np.nanmedian(power_spectrum_initial/meas_interp(wavelength_display))
plt.plot(wavelength_display, power_spectrum_initial/scale, label=f'initial Model / {scale:g}')

scale = np.nanmedian(power_spectrum_optimized/meas_interp(wavelength_display))
plt.plot(wavelength_display, power_spectrum_optimized/scale, label=f'optimized Model / {scale:g}')

plt.ylim(0, 1.5*np.nanmax(power_masked_meas))
plt.xlabel('Wavelength (nm)')
plt.ylabel('Average Radiated Power (W/nm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simple_fitting.pdf")
plt.show()

# ===== RESULTS =====

max_temp_initial = np.max(T_e_initial)
peak_wavelength_initial = wavelength_display[np.argmax(power_spectrum_initial)]

print(f"\n===== RESULTS =====")
print(f"Initial model peak temperature: {max_temp_initial:.0f} K")
print(f"Initial model peak wavelength: {peak_wavelength_initial:.0f} nm")

if measurement_available and 'T_e_optimized' in locals():
    max_temp_optimized = np.max(T_e_optimized)
    peak_wavelength_optimized = wavelength_display[np.argmax(power_spectrum_optimized)]
    print(f"Optimized model peak temperature: {max_temp_optimized:.0f} K")
    print(f"Optimized model peak wavelength: {peak_wavelength_optimized:.0f} nm")

print(f"\nTo fit different parameters, modify 'fit_params_list':")
print(f"Current: {fit_params_list}")
print("Available: ['g', 'tau_fwhm', 'P_avg', 'd', 'spot_diameter']")
# %%
