# %%
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
import numpy as np
from math import pi
from scipy.constants import h, c, k

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


# Laser Pulse Parameters
P_avg = 0.1                    # Average power (W)
f_rep = 40e3                   # Repetition rate (Hz)
E_pulse = P_avg / f_rep        # Pulse energy (J)
tau_fwhm = 250e-15             # FWHM pulse duration (s)
sigma = tau_fwhm / 2.355       # Gaussian std dev (s)
# spot_diameter = 50e-6          # Laser spot diameter (m)
spot_diameter = 70e-6          # Laser spot diameter (m)
r = spot_diameter / 2          # Laser spot radius (m)

# Volume and material constants
d = 250e-9                     # Optical absorption depth (m)
V = pi * r**2 * d              # Excited volume (m³)
A = pi * r**2                  # Surface area (m²)
M = 0.012                      # Molar mass (kg/mol)
rho = 2260                     # Density (kg/m³)
Vm = M / rho                   # Molar volume (m³/mol)

# Coupling and environment
g = 0.3e-12                    # Electron-lattice coupling time (s)
T_room = 294                   # Room temperature (K)

def gauss_pulse(t, sigma, t0=0):
    """Gaussian laser pulse"""
    return np.exp(-((t-t0)**2) / (2 * sigma**2))

def c_e(T):
    """Electronic heat capacity"""
    return 1e-6 * 12.8 * T * (1 + 1.16e-3 * T + 2.6e-7 * T**2)

def planck_spectrum(wavelength, T):
    """Planck's law for blackbody radiation [W/(m³·sr)]"""
    wavelength = wavelength * 1e-9
    B = (2 * h * c**2 / wavelength**5) / (np.exp(h * c / (wavelength * k * T)) - 1)
    return B

# Time vector
t = np.linspace(-5*sigma, 3*(sigma+g), 1000)
dt = t[1] - t[0]

# Initialize temperature array
T_e = np.full(len(t), T_room, dtype=np.float64)

# Laser source array (normalized to W/m³)
S = gauss_pulse(t, sigma)
S = S * E_pulse / (V * np.sum(S * dt))  # Now S has units W/m³

# Time evolution using forward Euler
for n in range(len(t) - 1):
    T_e[n+1] = T_e[n]
    # Laser heating term (S is already power density W/m³)
    T_e[n+1] += dt * Vm / c_e(T_e[n]) * S[n]  # Remove the "/ V"
    # Cooling term
    T_e[n+1] += -dt * (T_e[n] - T_room) / g

# Convert time to ps for plotting
t_ps = t / 1e-12

# Calculate integrated blackbody spectrum with real units
wavelength = np.linspace(200, 1000, 500)  # nm

# Calculate average power radiated per unit wavelength [W/nm]
# Assuming emission into 2π steradians (hemisphere)
solid_angle = 2 * pi  # steradians

power_spectrum = np.zeros_like(wavelength)

for i, wl in enumerate(wavelength):
    # Integrate over the hot electron cooling time
    for j, temp in enumerate(T_e):
        # Power radiated = Planck function × Volume × solid angle × repetition rate
        radiance = planck_spectrum(wl, temp)  # W/(m³·sr)
        power_density = radiance * A * solid_angle  # W/m (power per unit wavelength)
        power_spectrum[i] += power_density * dt * f_rep  # Average power W/m

# Convert to more convenient units
power_spectrum_nW = power_spectrum * 1e-9 # Convert to W/nm

# Create plots
fig = plt.figure()

# Temperature dynamics
ax1 = plt.subplot(2, 1, 1)
ax1.plot(t_ps, S * V, '-', label='Laser Power')
ax1.set_ylabel('Laser Power (W)')

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax2.plot(t_ps, T_e, label='Electron Temperature')
ax2.set_xlabel('Time (ps)')
ax2.set_ylabel('Temperature (K)')

plt.savefig("figures/model te.pdf")
plt.show()


# Blackbody spectrum with real units
plt.figure()
plt.plot(wavelength, power_spectrum_nW)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Average Radiated Power (W/nm)')
plt.ylim(0,None)

plt.savefig("figures/model spectrum.pdf")
plt.show()

# Print results
max_temp = np.max(T_e)
max_temp_time = t_ps[np.argmax(T_e)]
fluence = E_pulse / (pi * r**2)
peak_wavelength = wavelength[np.argmax(power_spectrum_nW)]
total_power = np.trapz(power_spectrum, wavelength * 1e-9)

print(f"Pulse energy: {E_pulse*1e6:.1f} μJ")
print(f"Fluence: {fluence:.2f} J/m²")
print(f"Maximum electron temperature: {max_temp:.0f} K")
print(f"Time of maximum temperature: {max_temp_time:.3f} ps")
print(f"Temperature rise: {max_temp - T_room:.0f} K")
print(f"Peak wavelength of thermal radiation: {peak_wavelength:.0f} nm")
print(f"Total average radiated power: {total_power*1e12:.1f} pW")
print(f"Peak spectral power: {np.max(power_spectrum_nW):.2f} nW/nm")