#%%
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
import numpy as np
from math import pi
from scipy.constants import h, c, k
from labellines import labelLines
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import warnings
from tqdm import tqdm
from scipy.stats import skewnorm, truncnorm

import os

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Suppress overflow warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class LaserParameters:
    """Laser pulse parameters with uncertainties"""
    P_avg: float = 0.3              # Average power (W)
    P_avg_uncertainty: float = 0.05 # Power uncertainty (W)
    f_rep: float = 40e3             # Repetition rate (Hz)
    f_rep_uncertainty: float = 0   # Frequency uncertainty (Hz)
    tau_fwhm: float = 250e-15       # FWHM pulse duration (s)
    tau_uncertainty: float = 50e-15  # Pulse duration uncertainty (s)
    spot_diameter: float = 70e-6     # Laser spot diameter (m)
    spot_uncertainty: float = 50e-6   # Spot size uncertainty (m)
    
    @property
    def E_pulse(self) -> float:
        """Pulse energy (J)"""
        return self.P_avg / self.f_rep
    
    @property
    def sigma(self) -> float:
        """Gaussian standard deviation (s)"""
        return self.tau_fwhm / 2.355
    
    @property
    def spot_radius(self) -> float:
        """Laser spot radius (m)"""
        return self.spot_diameter / 2

@dataclass
class MaterialParameters:
    """Material properties with uncertainties"""
    d: float = 250e-9               # Optical absorption depth (m)
    d_uncertainty: float = 100e-9    # Depth uncertainty (m)
    M: float = 0.012                # Molar mass (kg/mol)
    rho: float = 2260               # Density (kg/m³)
    rho_uncertainty: float = 100    # Density uncertainty (kg/m³)
    g: float = 0.25e-12             # Electron-lattice coupling time (s)
    g_uncertainty: float = 0.05e-12  # Coupling time uncertainty (s)
    T_room: float = 294             # Room temperature (K)
    T_room_uncertainty: float = 0   # Temperature uncertainty (K)
    
    @property
    def Vm(self) -> float:
        """Molar volume (m³/mol)"""
        return self.M / self.rho

@dataclass
class SimulationParameters:
    """Simulation control parameters"""
    t_min: float = -0.5e-12         # Start time (s)
    t_max: float = 1e-12            # End time (s)
    n_points: int = 1000            # Number of time points
    wavelength_min: float = 200     # Minimum wavelength (nm)
    wavelength_max: float = 1000    # Maximum wavelength (nm)
    n_wavelengths: int = 500        # Number of wavelength points
    n_monte_carlo: int = 100       # Number of Monte Carlo samples

def sample_parameters(laser_params: LaserParameters, 
                     material_params: MaterialParameters) -> Tuple[LaserParameters, MaterialParameters]:
    """Sample parameters from their uncertainty distributions"""
    
    # Sample laser parameters
    sampled_laser = LaserParameters(
        P_avg=np.random.normal(laser_params.P_avg, laser_params.P_avg_uncertainty),
        f_rep=np.random.normal(laser_params.f_rep, laser_params.f_rep_uncertainty),
        tau_fwhm=np.random.normal(laser_params.tau_fwhm, laser_params.tau_uncertainty),
        # Asymmetric spot size: 70 +50/-0 μm
        spot_diameter=sample_asymmetric_parameter(
            central_value=laser_params.spot_diameter,
            upper_uncertainty=laser_params.spot_uncertainty,
            lower_uncertainty=0,
            method='skewed_normal'
        )
    )
    
    # Sample material parameters
    sampled_material = MaterialParameters(
        d=np.random.normal(material_params.d, material_params.d_uncertainty),
        rho=np.random.normal(material_params.rho, material_params.rho_uncertainty),
        g=np.random.normal(material_params.g, material_params.g_uncertainty),
        T_room=np.random.normal(material_params.T_room, material_params.T_room_uncertainty)
    )
    
    return sampled_laser, sampled_material

def sample_asymmetric_parameter(central_value: float, 
                               upper_uncertainty: float, 
                               lower_uncertainty: float,
                               method: str = 'skewed_normal') -> float:
    """
    Sample from an asymmetric distribution
    
    Args:
        central_value: Central value (e.g., 70e-6)
        upper_uncertainty: Upper uncertainty (e.g., 50e-6) 
        lower_uncertainty: Lower uncertainty (e.g., 0)
        method: 'skewed_normal', 'truncated_normal', or 'piecewise'
    
    Returns:
        Sampled value
    """
    
    if method == 'skewed_normal':
        # Use scipy's skewed normal distribution
        from scipy.stats import skewnorm
        
        # Calculate skewness parameter from asymmetry
        if lower_uncertainty == 0:
            # Special case: no lower uncertainty (hard boundary)
            skewness = 5  # Strong positive skew
            scale = upper_uncertainty / 2
        else:
            # General asymmetric case
            asymmetry = (upper_uncertainty - lower_uncertainty) / (upper_uncertainty + lower_uncertainty)
            skewness = asymmetry * 5  # Scale factor for skewness
            scale = (upper_uncertainty + lower_uncertainty) / 4
        
        sample = skewnorm.rvs(a=skewness, loc=central_value, scale=scale)
        
        # Ensure physical bounds
        min_value = central_value - lower_uncertainty
        max_value = central_value + upper_uncertainty
        sample = np.clip(sample, min_value, max_value)
        
        return sample
    
    elif method == 'truncated_normal':
        # Truncated normal distribution
        from scipy.stats import truncnorm
        
        # Average the uncertainties for the scale
        avg_uncertainty = (upper_uncertainty + lower_uncertainty) / 2
        if avg_uncertainty == 0:
            avg_uncertainty = upper_uncertainty / 2
        
        # Define bounds in standard deviations
        lower_bound = (central_value - lower_uncertainty - central_value) / avg_uncertainty
        upper_bound = (central_value + upper_uncertainty - central_value) / avg_uncertainty
        
        sample = truncnorm.rvs(lower_bound, upper_bound, 
                              loc=central_value, scale=avg_uncertainty)
        return sample
    
    elif method == 'piecewise':
        # Piecewise approach: different distributions above/below central value
        if np.random.random() < 0.5:
            # Sample below central value
            if lower_uncertainty > 0:
                return central_value - np.random.exponential(lower_uncertainty / 2)
            else:
                return central_value  # No variation below
        else:
            # Sample above central value
            return central_value + np.random.exponential(upper_uncertainty / 2)
    
    else:
        raise ValueError(f"Unknown method: {method}")

def gauss_pulse(t: np.ndarray, sigma: float, t0: float = 0) -> np.ndarray:
    """Gaussian laser pulse shape"""
    return np.exp(-((t - t0)**2) / (2 * sigma**2))

def electronic_heat_capacity(T: np.ndarray) -> np.ndarray:
    """Electronic heat capacity as function of temperature"""
    return 1e-6 * 12.8 * T * (1 + 1.16e-3 * T + 2.6e-7 * T**2)

def planck_spectrum(wavelength_nm: np.ndarray, T: float) -> np.ndarray:
    """
    Planck's law for blackbody radiation
    
    Args:
        wavelength_nm: Wavelength array in nanometers
        T: Temperature in Kelvin
    
    Returns:
        Spectral radiance in W/(m³·sr)
    """
    wavelength_m = wavelength_nm * 1e-9
    
    # Handle division by zero and overflow
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        exponent = h * c / (wavelength_m * k * T)
        # Clip exponent to prevent overflow
        exponent = np.clip(exponent, 0, 700)
        
        B = (2 * h * c**2 / wavelength_m**5) / (np.exp(exponent) - 1)
        
        # Handle any remaining infinities or NaNs
        B = np.where(np.isfinite(B), B, 0)
    
    return B

def solve_temperature_evolution(t: np.ndarray, 
                               laser_params: LaserParameters,
                               material_params: MaterialParameters) -> np.ndarray:
    """
    Solve the two-temperature model for electron temperature evolution
    
    Args:
        t: Time array (s)
        laser_params: Laser parameters
        material_params: Material parameters
    
    Returns:
        Temperature array T_e(t) in Kelvin
    """
    dt = t[1] - t[0]
    
    # Geometric parameters
    r = laser_params.spot_radius
    V = pi * r**2 * material_params.d  # Excited volume
    
    # Initialize temperature
    T_e = np.full(len(t), material_params.T_room, dtype=np.float64)
    
    # Laser source term
    S = gauss_pulse(t, laser_params.sigma)
    S = S * laser_params.E_pulse / (V * np.sum(S * dt))  # Normalize to W/m³
    
    # Time evolution using forward Euler
    for n in range(len(t) - 1):
        # Laser heating term
        heating = dt * material_params.Vm / electronic_heat_capacity(T_e[n]) * S[n]
        
        # Cooling term
        cooling = -dt * (T_e[n] - material_params.T_room) / material_params.g
        
        T_e[n + 1] = T_e[n] + heating + cooling
        
        # Ensure temperature doesn't go below room temperature
        T_e[n + 1] = max(T_e[n + 1], material_params.T_room)
    
    return T_e

def calculate_thermal_spectrum(wavelength: np.ndarray,
                              T_e: np.ndarray,
                              t: np.ndarray,
                              laser_params: LaserParameters,
                              material_params: MaterialParameters) -> np.ndarray:
    """
    Calculate the time-integrated thermal emission spectrum
    
    Args:
        wavelength: Wavelength array (nm)
        T_e: Temperature evolution array (K)
        t: Time array (s)
        laser_params: Laser parameters
        material_params: Material parameters
    
    Returns:
        Power spectrum in W/nm
    """
    dt = t[1] - t[0]
    r = laser_params.spot_radius
    A = pi * r**2  # Surface area
    solid_angle = 2 * pi  # Hemisphere
    
    power_spectrum = np.zeros_like(wavelength)
    
    # Vectorized approach - much faster
    for j, temp in enumerate(T_e):
        if temp > material_params.T_room:  # Only consider hot electrons
            # Calculate Planck spectrum for all wavelengths at once
            radiance = planck_spectrum(wavelength, temp)  # W/(m³·sr) - vectorized
            power_density = radiance * A * solid_angle  # W/m
            power_spectrum += power_density * dt * laser_params.f_rep
    
    return power_spectrum

def run_single_simulation(laser_params: LaserParameters,
                         material_params: MaterialParameters,
                         sim_params: SimulationParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a single hot electron simulation
    
    Returns:
        (time_array, temperature_array, spectrum_array)
    """
    # Create time and wavelength arrays
    t = np.linspace(sim_params.t_min, sim_params.t_max, sim_params.n_points)
    wavelength = np.linspace(sim_params.wavelength_min, sim_params.wavelength_max, sim_params.n_wavelengths)
    
    # Solve temperature evolution
    T_e = solve_temperature_evolution(t, laser_params, material_params)
    
    # Calculate spectrum
    spectrum = calculate_thermal_spectrum(wavelength, T_e, t, laser_params, material_params)
    
    return t, T_e, spectrum

def run_monte_carlo_simulation(laser_params: LaserParameters = LaserParameters(),
                              material_params: MaterialParameters = MaterialParameters(),
                              sim_params: SimulationParameters = SimulationParameters()) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation with parameter uncertainties
    
    Returns:
        Dictionary containing statistics of results
    """
    # Storage for results
    temperature_evolutions = []
    spectra = []
    peak_temperatures = []
    
    # Create base arrays
    t = np.linspace(sim_params.t_min, sim_params.t_max, sim_params.n_points)
    wavelength = np.linspace(sim_params.wavelength_min, sim_params.wavelength_max, sim_params.n_wavelengths)
    
    for i in tqdm(range(sim_params.n_monte_carlo), desc="Monte Carlo progress"):
        # Sample parameters
        sampled_laser, sampled_material = sample_parameters(laser_params, material_params)
        
        # Run simulation
        _, T_e, spectrum = run_single_simulation(sampled_laser, sampled_material, sim_params)
        
        # Store results
        temperature_evolutions.append(T_e)
        spectra.append(spectrum)
        peak_temperatures.append(np.max(T_e))
    
    # Convert to arrays for statistical analysis
    temperature_evolutions = np.array(temperature_evolutions)
    spectra = np.array(spectra)
    peak_temperatures = np.array(peak_temperatures)
    
    # Calculate statistics including 1σ equivalent (15.85% and 84.15%)
    results = {
        'time': t,
        'wavelength': wavelength,
        'temperature_mean': np.mean(temperature_evolutions, axis=0),
        'temperature_std': np.std(temperature_evolutions, axis=0),
        'temperature_percentiles': np.percentile(temperature_evolutions, [15.85, 84.15], axis=0),
        'spectrum_mean': np.mean(spectra, axis=0),
        'spectrum_std': np.std(spectra, axis=0),
        'spectrum_percentiles': np.percentile(spectra, [15.85, 84.15], axis=0),
        'peak_temp_mean': np.mean(peak_temperatures),
        'peak_temp_std': np.std(peak_temperatures),
        'peak_temp_distribution': peak_temperatures,
        'n_simulations': sim_params.n_monte_carlo
    }
    
    return results



# Example usage
if __name__ == "__main__":
    # Define parameters
    laser_params = LaserParameters()
    material_params = MaterialParameters()
    sim_params = SimulationParameters(n_monte_carlo=100)
    
    # Run deterministic simulation
    t, T_e, spectrum = run_single_simulation(laser_params, material_params, sim_params)
    
    # Run Monte Carlo simulation
    mc_results = run_monte_carlo_simulation(laser_params, material_params, sim_params)
    
    # Print statistics
    print(f"\nResults Summary:")
    print(f"Peak temperature: {mc_results['peak_temp_mean']:.0f} ± {mc_results['peak_temp_std']:.0f} K")
    print(f"Peak spectrum power: {np.max(mc_results['spectrum_mean'])*1e9:.2f} ± {np.max(mc_results['spectrum_std'])*1e9:.2f} W/nm")

    # ===== PLOTTING =====
    
    # Convert time to ps for plotting
    t_ps = mc_results['time'] / 1e-12
    
    # Temperature dynamics plot with uncertainty
    fig = plt.figure()
    
    ax1 = plt.subplot(2, 1, 1)
    # Laser pulse
    laser_pulse = gauss_pulse(mc_results['time'], laser_params.sigma)
    V = pi * laser_params.spot_radius**2 * material_params.d
    laser_pulse = laser_pulse * laser_params.E_pulse / (V * np.trapz(laser_pulse, mc_results['time']))
    ax1.plot(t_ps, laser_pulse * V, '-', label='Laser Power')
    ax1.set_ylabel('Laser Power (W)')
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.set_ylim(0, None)
    ax1.set_xlim(t_ps.min(), t_ps.max())
    
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    # Mean temperature
    l = ax2.plot(t_ps, mc_results['temperature_mean'])[0]
    
    # Uncertainty bands
    # ax2.fill_between(t_ps, 
    #                 mc_results['temperature_percentiles'][0], 
    #                 mc_results['temperature_percentiles'][1],
    #                 alpha=0.3, color=l.get_color())
    
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Temperature (K)')
    
    plt.savefig("figures/model te.pdf")
    plt.show()


    # Blackbody spectrum plot with uncertainty and reference temperatures
    plt.figure()
    wavelength = mc_results['wavelength']
    
    # Main spectrum with uncertainty
    spectrum_mean_nW = mc_results['spectrum_mean'] * 1e-9  # Convert to W/nm
    # %%
    l = plt.plot(wavelength, spectrum_mean_nW)[0]
    
    # Uncertainty bands
    # plt.fill_between(wavelength,
    #                 mc_results['spectrum_percentiles'][0] * 1e-9,
    #                 mc_results['spectrum_percentiles'][1] * 1e-9,
    #                 alpha=0.3, color=l.get_color())
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Average Radiated Power (W/nm)')
    plt.ylim(0, None)
    
    # Calculate effective radiating time for duty cycle
    T_hot = mc_results['temperature_mean'] - material_params.T_room
    mean_time = np.sum(t_ps * T_hot) / np.sum(T_hot)  # Weighted mean time
    variance_time = np.sum(T_hot * (t_ps - mean_time)**2) / np.sum(T_hot)  # Weighted variance
    std_time = np.sqrt(variance_time) * 1e-12  # Convert back to seconds
    duty_cycle = std_time * laser_params.f_rep
    
    # Plot reference temperature curves
    reference_temps = np.arange(7000,10001, 1000)  # K
    thermal_lines = []
    
    for temp in reference_temps:
        # Calculate Planck spectrum for this temperature - vectorized
        radiance = planck_spectrum(wavelength, temp)  # W/(m³·sr) - all wavelengths at once
        # Power from volume into hemisphere, per wavelength bin
        A = pi * laser_params.spot_radius**2
        power_per_nm = radiance * A * 2 * pi * 1e-9  # W/nm
        # Apply duty cycle to get average power
        thermal_spectrum = power_per_nm * duty_cycle
        
        # Plot thermal spectrum line
        line = plt.plot(wavelength, thermal_spectrum, 
                       color="gray", linewidth=1, alpha=0.7,
                       label=f'{temp} K')[0]
        thermal_lines.append(line)
    
    labelLines(thermal_lines, fontsize=8)
    
    plt.autoscale()
    plt.ylim(0, None)
    plt.xlim(wavelength.min(), wavelength.max())
    plt.savefig("figures/model spectrum.pdf")
    plt.show()