#%%
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import *
from math import pi
import numpy as np
plt.style.use("style.mplstyle")

# --------- Utility functions (kept simple) ---------
def ev_nm(x): return 1240/x

def convert_spectrum_to_ev(wavelength_nm: np.ndarray, spectrum_per_nm: np.ndarray):
        """
        Convert a spectrum expressed per nm to per eV.
        Given E = 1240/λ, we have |dE/dλ| = 1240/λ², so dλ/dE = λ²/1240.
        The spectrum per eV is then: S(E) = S(λ) * |dλ/dE|.
        """
        return 1240/wavelength_nm, spectrum_per_nm * (wavelength_nm**2 / 1240)

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
    F_exc: float = 400           # Excitation fluence per pulse [J·m^-2]
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

    @property
    def dt(self) -> float:
        t = self.time
        return t[1]-t[0]

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
    # Plot electron temperature
    fig, axs = plt.subplots(3, 1, sharex=True)

    # Excitation pulse profile
    axs[0].plot(sim.time / pico, pulse(sim.time, sim.sigma) / tera, label="excitation (TW/m²)")
    axs[0].set_ylabel("Excitation\n(TW/m²)")
    axs[0].legend()
    axs[0].set_ylim(0, None)


    # Electron temperature
    axs[1].plot(sim.time / pico, sim.temperature(), label=r'$T_e$ (K)')
    axs[1].set_ylabel(r"$T_e$ (K)")
    axs[1].legend()
    axs[1].set_ylim(0, None)

    # Radiance
    axs[2].plot(sim.time / pico, sim.temperature()**4 * Stefan_Boltzmann / pi / mega, label="total")

    wl = np.linspace(400, 900, 100)
    T = sim.temperature()
    dwl = wl[1]-wl[0]
    axs[2].plot(
        sim.time / pico,
        np.sum(planck(wl, T)*dwl*nano, axis=1)/mega,
        label=f"{wl.min():g}-{wl.max():g} nm"
    )

    axs[2].set_ylabel(r'$P$ (MW/m²/sr)')
    axs[2].set_ylim(0, None)
    axs[2].legend()

    axs[2].set_xlabel("Time (ps)")
    plt.savefig("figures/model.time_evolution.pdf")
    plt.show()

    ########### Spectrum #############
    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.plot(*convert_spectrum_to_ev(sim.wavelength_nm, sim.spectrum()))
    plt.xlabel("Photon Energy (eV)")
    plt.ylabel(r"Spectrum (J/m²/sr/eV)")
    plt.ylim(0, None)

    plt.subplot(1, 2, 2)
    plt.plot(sim.wavelength_nm, sim.spectrum())
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(r"$E / \epsilon$ (J/m³/sr)")
    plt.ylim(0, None)

    plt.tight_layout()
    plt.savefig("figures/model.spectrum.pdf")
    plt.show()


# %%