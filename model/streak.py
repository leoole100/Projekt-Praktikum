#%%
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
plt.style.use("../style.mplstyle")

# %%

# Physical constants
h = 6.626e-34   # Planck constant (J·s)
c = 300e6       # Speed of light (m/s)
kB = 1.381e-23  # Boltzmann constant (J/K)

class Streak:
    def __init__(self, model):
        self.model = model
        self.model.t = np.linspace(-3*self.model.sigma, .7*(self.model.sigma+self.model.g), 1000)
        self.l = np.linspace(100, 2000, 100)*1e-9

    def __call__(self):
        self.model()
        self.t = self.model.t
        self.T_e = self.model.T_e
        self.S =self.model.S
        self.b = self._compute_b(self.l, self.T_e)
        return self

    # W/nm^3
    def _compute_b(self, l, T_e):
        return 2*h*c**2 / l[:, None]**5 / (
            np.exp(h * c / (l[:, None] * kB * T_e[None, :])) - 1
        )

    @property
    def sum(self):
        return self.b.sum(axis=1)

    @property
    def power(self):
        return self.b.sum()

if __name__=="__main__":

    from model import Model
    m = Model()
    m.fwhm = 10e-15
    s = Streak(m)
    s()

    # Plot example
    fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].invert_yaxis()
    ax[1].invert_xaxis()
    ax[0].plot(s.T_e/1e3, s.t/1e-15)
    ax[0].set_xlabel("T (10³ K)")
    ax[1].contourf(s.l/1e-9, s.t/1e-15, s.b.T, cmap='gray_r', vmin=0)
    ax[0].set_ylabel("t (fs)")
    ax[1].set_xlabel("λ (nm)")
    plt.savefig("./figures/streak view.pdf")
    plt.show()

    # plot the expected spectrum
    m = Model()
    m.E = 50e-6
    m.fwhm = 250e-15
    s = Streak(m)
    s()
    s.T_e.max()
    plt.plot(s.l/1e-9, s.b.sum(axis=1)/1e15)
    plt.xlim(300, 2000)
    plt.xlabel(r"$\lambda$ (nm)")
    plt.ylabel(r"$\int B dt$")
    plt.savefig("./figures/spectrum.pdf")

# %%
