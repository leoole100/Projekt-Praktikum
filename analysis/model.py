# %%
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
import numpy as np
from math import pi
from scipy.constants import h, c, k
from labellines import labelLines

import os

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# ===== PARAMETERS =====

# Laser Pulse Parameters
P_avg = 0.1                    # Average power (W)
f_rep = 40e3                   # Repetition rate (Hz)
E_pulse = P_avg / f_rep        # Pulse energy (J)
tau_fwhm = 250e-15             # FWHM pulse duration (s)
sigma = tau_fwhm / 2.355       # Gaussian std dev (s)
spot_diameter = 70e-6          # Laser spot diameter (m)
r = spot_diameter / 2          # Laser spot radius (m)

# Volume and Material Constants
d = 250e-9                     # Optical absorption depth (m)
V = pi * r**2 * d              # Excited volume (m³)
A = pi * r**2                  # Surface area (m²)
M = 0.012                      # Molar mass (kg/mol)
rho = 2260                     # Density (kg/m³)
Vm = M / rho                   # Molar volume (m³/mol)

# Coupling and Environment
g = 0.3e-12                    # Electron-lattice coupling time (s)
T_room = 294                   # Room temperature (K)

# ===== FUNCTIONS =====

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

# ===== TIME EVOLUTION CALCULATION =====

# Time vector
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

# ===== SPECTRUM CALCULATION =====

# Calculate integrated blackbody spectrum
wavelength = np.linspace(200, 1000, 500)  # nm
solid_angle = 2 * pi  # steradians (hemisphere)

power_spectrum = np.zeros_like(wavelength)

for i, wl in enumerate(wavelength):
    # Integrate over the hot electron cooling time
    for j, temp in enumerate(T_e):
        # Power radiated = Planck function × Area × solid angle × repetition rate
        radiance = planck_spectrum(wl, temp)  # W/(m³·sr)
        power_density = radiance * A * solid_angle  # W/m (power per unit wavelength)
        power_spectrum[i] += power_density * dt * f_rep  # Average power W/m

# Convert to more convenient units
power_spectrum_nW = power_spectrum * 1e-9  # Convert to W/nm

# ===== PLOTTING =====

# Convert time to ps for plotting
t_ps = t / 1e-12

# Temperature dynamics plot
fig = plt.figure()

ax1 = plt.subplot(2, 1, 1)
ax1.plot(t_ps, S * V, '-', label='Laser Power')
ax1.set_ylabel('Laser Power (W)')
ax1.tick_params(axis='x', labelbottom=False)
ax1.set_ylim(0, None)
ax1.set_xlim(t_ps.min(), t_ps.max())

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax2.plot(t_ps, T_e, label='Electron Temperature')
ax2.set_xlabel('Time (ps)')
ax2.set_ylabel('Temperature (K)')

plt.savefig("figures/model te.pdf")
plt.show()

# Blackbody spectrum plot with reference temperatures
plt.figure()
plt.plot(wavelength, power_spectrum_nW)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Average Radiated Power (W/nm)')
plt.ylim(0, None)

# Calculate effective radiating time for duty cycle
T_hot = T_e - T_room
mean_time = np.sum(t_ps * T_hot) / np.sum(T_hot)  # Weighted mean time
variance_time = np.sum(T_hot * (t_ps - mean_time)**2) / np.sum(T_hot)  # Weighted variance
std_time = np.sqrt(variance_time) * 1e-12  # Convert back to seconds
duty_cycle = std_time * f_rep

# Plot reference temperature curves
reference_temps = np.arange(5000, 7001, 500)  # K
thermal_lines = []

for temp in reference_temps:
    # Calculate Planck spectrum for this temperature
    thermal_spectrum = np.zeros_like(wavelength)
    for j, wl in enumerate(wavelength):
        radiance = planck_spectrum(wl, temp)  # W/(m³·sr)
        # Power from volume into hemisphere, per wavelength bin
        power_per_nm = radiance * A * solid_angle * 1e-9  # W/nm
        # Apply duty cycle to get average power
        thermal_spectrum[j] = power_per_nm * duty_cycle
    
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
