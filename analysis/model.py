#%%
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import *
from math import pi

# --------- Utility functions (kept simple) ---------
def pulse(t: np.ndarray, sigma: float, t0: float = 0.0) -> np.ndarray:
    """Unit-area Gaussian centered at t0 with stdev sigma."""
    g = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    return g / np.trapz(g, t)

def electronic_heat_capacity(T: np.ndarray) -> np.ndarray:
    """
    Electronic heat capacity C_e(T) [J·m^-3·K^-1].
    From https://doi.org/10.1103/PhysRevB.68.134305
    """
    M = 0.012                # Molar mass (kg/mol)
    rho = 2260               # Density (kg/m³)
    c = 1e-6 * 13.8 * T * (1 + 1.16e-3 * T + 2.6e-7 * T**2) # J/mol/K
    c *= rho/M
    return c


def planck(wavelength_nm: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Spectral radiance B_λ(T) [W·m^-3·sr^-1] at wavelength(s) in nm and temperature(s) T (in Kelvin).
    If T is an array, the function returns an array with shape (len(T), len(wavelength_nm)).
    """
    lam = wavelength_nm * 1e-9  # Convert wavelengths to meters
    lam = lam.reshape(1, -1)
    T = np.atleast_1d(T).reshape(-1, 1)
    
    B = (2 * h * c**2 / lam**5) / (np.expm1(h * c / (lam * k * T)))  
    return B

# --------- Main simulation class ---------
@dataclass
class HotElectronSim:
    # Practical inputs
    F_exc: float = 600           # Excitation fluence per pulse [J·m^-2]
    tau_fwhm: float = 250*femto  # Pulse FWHM [s]
    d_abs: float = 200*nano      # Effective absorption depth [m]
    tau_eph: float = 0.25*pico   # Electron-lattice equilibration time [s]
    T_room: float = 300          # Initial / lattice temperature [K]

    # Grids
    t_min: float = -0.3e-12
    t_max: float =  1.0e-12
    n_t:   int    = 1000
    wl_min_nm: float = 200.0
    wl_max_nm: float = 1000.0
    n_wl: int = 500

    # ------- Derived convenience properties -------
    @property
    def sigma(self) -> float:
        return self.tau_fwhm / 2.355 / 2

    @property
    def time(self) -> np.ndarray:
        return np.linspace(self.t_min, self.t_max, self.n_t)

    @property
    def wavelength_nm(self) -> np.ndarray:
        return np.linspace(self.wl_min_nm, self.wl_max_nm, self.n_wl)

    # ------- Core model -------
    def temperature(self) -> np.ndarray:
        t = self.time
        dt = t[1] - t[0]

        U = self.F_exc / self.d_abs  # energy density deposited per volume
        S = pulse(t, self.sigma) * U

        T = np.full_like(t, self.T_room, dtype=float)

        for i in range(len(t) - 1):
            Ce = electronic_heat_capacity(T[i])
            heating = (S[i] / Ce) * dt                    # K increment from source
            cooling = -((T[i] - self.T_room) / self.tau_eph) * dt  # K increment from cooling
            T[i + 1] = T[i] + heating + cooling

        return T

    def spectrum(self) -> np.ndarray:
        T_t = self.temperature()
        dt = self.time[1] - self.time[0]
        E_lambda = np.sum(planck(self.wavelength_nm, T_t)*dt, axis=0)
        return E_lambda

# ---------------- Example usage ----------------
if __name__ == "__main__":
    sim = HotElectronSim()
    sim.spectrum()

    # ---- Plots (per pulse, per area) ----
    plt.plot()
    plt.plot(sim.time / pico, sim.temperature())
    plt.xlabel("Time (ps)")
    plt.ylabel(r"$T_e$ (K)")
    plt.show()


    plt.plot(sim.wavelength_nm, sim.spectrum())
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(r"$E_\lambda$ (J/m³/sr)")

    ev_nm = lambda x:1240/x
    plt.gca().secondary_xaxis('top', functions=(ev_nm, ev_nm)).\
    set_xlabel("Photon Energy (eV)")

    plt.show()


# %%