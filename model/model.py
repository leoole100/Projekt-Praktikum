#%%
import matplotlib.pyplot as plt
import numpy as np

# %%
t = np.arange(0, 100e-15, 0.1e-15)	# s
dt = t[1]-t[0]

# Laser Pulse
E = 12e-6		    # pulse energy (J)
sigma = 2e-15     	# pulse width (s)
t0 = 5*sigma        # center of pulse (s)
S = np.exp(-((t - t0)**2) / (2 * sigma**2))  # laser source array
S = E*S/np.sum(S*dt)# normalisation
r = 200e-6          # laser spot radius (m)
d = 200e-9          # optical depth (m)
M = 0.012           # Molar mass (kg/mol)
rho = 2260          # density in (kg/m³)
Vm = M/rho
c_e = lambda T: 1e-6 * 12.8*T*(1+1.16e-3*T+2.6e-7*T**2) #  specific heat capacity (J/mol/K)

g = 100e-15         # coupling to lattice in s

T_room = 300        # K
T_e = np.full(len(t), T_room, dtype=np.float64)
T_l = np.full(len(t), T_room, dtype=np.float64)

# Time evolution
for n in range(len(t) - 1):
    T_e[n+1] = T_e[n]
    T_e[n+1] += Vm/c_e(T_e[n]) * S[n]
    T_e[n+1] -= (T_e[n] - T_room)/g
    # T_l[n+1] = T_l[n] + dT_l

# plot
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t/1e-15, S / 1e9)
ax[0].set_ylabel("Laser Power / GW")
ax[1].plot(t/1e-15, T_e, label="$T_e$")
# ax[1].plot(t/1e-15, T_l, label="$T_l$")
ax[1].legend()
ax[1].set_ylabel("T / K")
ax[1].set_xlabel("t / ps")
# for a in ax: a.label_outer()
plt.show()
# %%
