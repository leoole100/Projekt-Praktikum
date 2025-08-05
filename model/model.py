#%%
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("../style.mplstyle")
from math import pi

# %%

class Model():
    def __init__(self, source_function=None):
        # Laser Pulse Parameters
        self.E = 12e-6              # Pulse energy (J)
        self.sigma = 250e-15 / 2.33 # Gaussian std dev (s), converts from FWHM
        self.r = 200e-6             # Laser spot radius (m)
        self.source_function = self.gauss_peak

        # Volume and material constants
        self.d = 200e-9             # Optical absorption depth (m)
        self.V = pi * self.r**2 * self.d   # Excited volume (m³)
        self.M = 0.012              # Molar mass (kg/mol)
        self.rho = 2260             # Density (kg/m³)
        self.Vm = self.M / self.rho # Molar volume (m³/mol)

        # Coupling and environment
        self.g = 0.25e-12            # Electron-lattice coupling time (s)
        self.T_room = 50           # Ambient temperature (K)

        # Time vector
        self.t = None

    def gauss_peak(self, t, sigma=None, t0=0):
        if sigma==None: sigma=self.sigma
        return np.exp(-((t-t0)**2) / (2 * sigma**2)) # laser source array

    def c_e(self, T):
        return 1e-6 * 12.8 * T * (1 + 1.16e-3 * T + 2.6e-7 * T**2)

    def __call__(self, *args, **kwds):
        if self.t is None:
            self.t = np.linspace(-3*self.sigma, 3*(self.sigma+self.g), 1000)      # s
        self.dt = self.t[1]-self.t[0]
        self.T_e = np.full(len(self.t), self.T_room, dtype=np.float64)

        self.S = self.source_function(self.t) # laser source array
        self.S = self.E*self.S/np.sum(self.S*self.dt)       # normalisation

        # Time evolution
        for n in range(len(self.t) - 1):
            self.T_e[n+1] = self.T_e[n]
            self.T_e[n+1] += self.dt*self.Vm/self.c_e(self.T_e[n]) * self.S[n] / self.V   # Laser Heating
            self.T_e[n+1] += -self.dt* (self.T_e[n] - self.T_room)/self.g   # cooling
        return self

    #J/m²   
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
    ax[1].plot(t, m.T_e/1e3)
    ax[1].set_ylabel(r"$T_e$ (10³ K)")
    ax[1].set_xlabel("t (ps)")
    plt.savefig("figures/temperature profile.pdf")
    plt.show()

    # remake the plot in “Hot electron cooling in graphite”, Stange et al. 2015 (Fig. 4)
    m = Model()
    m.fluence = 14  #J/m²
    m.fwhm = 30e-15
    m.t = np.linspace(-0.1,1, 1000)*1e-12
    m()
    plt.figure(figsize=(2.5, 1.85))
    plt.plot(m.t/1e-12, m.T_e/1e3)
    plt.xlabel("t (ps)")
    plt.ylabel(r"$T_e$ (10³ K)")
    plt.xlim(-.1, 1)
    plt.savefig("figures/temperature stange_hot_2015.pdf")
# %%
