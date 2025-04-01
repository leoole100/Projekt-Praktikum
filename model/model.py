#%%
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("../style.mplstyle")
from math import pi

# %%

class Model():
    t = np.linspace(0, 1500e-15, 1000)	# s
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
    def c_e(self, T):
        return 1e-6 * 12.8*T*(1+1.16e-3*T+2.6e-7*T**2) #  specific heat capacity (J/mol/K)

    # different sources for hot electron lifetime
    g = 300e-15         # coupling to lattice in s

    T_room = 300        # K
    T_e = np.full(len(t), T_room, dtype=np.float64)
    T_l = np.full(len(t), T_room, dtype=np.float64)

    # 2.88ms
    def __call__(self, *args, **kwds):
        # Time evolution
        for n in range(len(self.t) - 1):
            self.T_e[n+1] = self.T_e[n]
            self.T_e[n+1] += self.dt*self.Vm/self.c_e(self.T_e[n]) * self.S[n] / self.V   # Laser Heating
            self.T_e[n+1] += -self.dt* (self.T_e[n] - self.T_l[n])/self.g   # cooling to lattice

if __name__ == "__main__":
    m = Model()
    m()
    # plot
    fig, ax = plt.subplots(2, 1, sharex=True)
    t=(m.t-m.t0)/1e-15
    ax[0].plot(t, m.S / 1e6)
    ax[0].set_ylabel("Laser Power / MW")
    ax[1].plot(t, m.T_e, label="$T_e$")
    ax[1].plot(t, m.T_l, "--", label="$T_l$")
    ax[1].legend()
    ax[1].set_ylabel("T / K")
    ax[1].set_xlabel("t / ps")
    plt.tight_layout()
    plt.savefig("figures/temperature profile.pdf")
    plt.show()
# %%
