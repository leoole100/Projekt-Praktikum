#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False
from math import pi

# %%
t = np.linspace(0, 3000e-15, 1000)	# s
dt = t[1]-t[0]

# Laser Pulse
E = 12e-6		    # pulse energy (J)
# sigma = 250e-15     # pulse width (s)
sigma = 250e-15/2.33# FWHM (s)
t0 = 5*sigma        # center of pulse (s)
S = np.exp(-((t - t0)**2) / (2 * sigma**2))  # laser source array
S = E*S/np.sum(S*dt)# normalisation
r = 200e-6          # laser spot radius (m)
d = 200e-9          # optical depth (m)
V = pi*r**2*d
M = 0.012           # Molar mass (kg/mol)
rho = 2260          # density in (kg/m³)
Vm = M/rho
c_e = lambda T: 1e-6 * 12.8*T*(1+1.16e-3*T+2.6e-7*T**2) #  specific heat capacity (J/mol/K)

# different sources for hot electron lifetime
g = 300e-15         # coupling to lattice in s

T_room = 300        # K
T_e = np.full(len(t), T_room, dtype=np.float64)
T_l = np.full(len(t), T_room, dtype=np.float64)

# Time evolution
for n in range(len(t) - 1):
    T_e[n+1] = T_e[n]
    T_e[n+1] += dt*Vm/c_e(T_e[n]) * S[n] / V   # Laser Heating
    T_e[n+1] += -dt* (T_e[n] - T_l[n])/g   # cooling to lattice

# plot
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t/1e-15, S / 1e6)
ax[0].set_ylabel("Laser Power / MW")
ax[1].plot(t/1e-15, T_e, label="$T_e$")
ax[1].plot(t/1e-15, T_l, "--", label="$T_l$")
ax[1].legend()
ax[1].set_ylabel("T / K")
ax[1].set_xlabel("t / ps")
plt.tight_layout()
plt.savefig("figures/temperature profile.pdf")
plt.show()
# %%
