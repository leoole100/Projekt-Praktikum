#%%
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("../style.mplstyle")
from math import pi

# %%

class Model():
    # Laser Pulse
    E = 12e-6		    # pulse energy (J)
    sigma = 250e-15/2.33# FWHM (s)
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

    t = None

    # 2.88ms
    def __call__(self, *args, **kwds):
        if self.t is None:
            self.t = np.linspace(-3*self.sigma, 3*(self.sigma+self.g), 1000)      # s
        self.dt = self.t[1]-self.t[0]
        self.T_e = np.full(len(self.t), self.T_room, dtype=np.float64)

        self.S = np.exp(-(self.t**2) / (2 * self.sigma**2)) # laser source array
        self.S = self.E*self.S/np.sum(self.S*self.dt)       # normalisation

    # Time evolution
        for n in range(len(self.t) - 1):
            self.T_e[n+1] = self.T_e[n]
            self.T_e[n+1] += self.dt*self.Vm/self.c_e(self.T_e[n]) * self.S[n] / self.V   # Laser Heating
            self.T_e[n+1] += -self.dt* (self.T_e[n] - self.T_room)/self.g   # cooling

    @property
    def fluence(self):
        return self.E/pi*self.r**2
    
    @fluence.setter
    def fluence(self, value):
        self.E = value*pi*self.r**2
    
    @property
    def fwhm(self):
        return self.sigma*2.33
    
    @fwhm.setter
    def fwhm(self, value): self.sigma = value/2.33

if __name__ == "__main__":
    m = Model()
    m()
    # plot
    fig, ax = plt.subplots(2, 1, sharex=True)
    t=m.t/1e-15
    ax[0].plot(t, m.S / 1e6)
    ax[0].set_ylabel("Laser Power (MW)")
    ax[1].plot(t, m.T_e)
    ax[1].set_ylabel(r"$T_e$ (K)")
    ax[1].set_xlabel("t (ps)")
    plt.tight_layout()
    plt.savefig("figures/temperature profile.pdf")
    plt.show()

    # remake the plot in “Hot electron cooling in graphite”, Stange et al. 2015 (Fig. 4)
    m = Model()
    m.fluence = 14  #J/m²
    m.fwhm = 30e-15
    m()
    plt.plot(m.t/1e-12, m.T_e/1e3)
    plt.xlabel("t (ps)")
    plt.ylabel(r"$T_e$ (10³ K)")
    plt.tight_layout()
    plt.savefig("figures/temperature stange_hot_2015.pdf")
# %%
