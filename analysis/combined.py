# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.constants import *
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import scipy as sp
from model import HotElectronSim

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

plt.style.use("style.mplstyle")

def spectrum_data(filepath, exposure_time=2):
    data = np.loadtxt(filepath)
    wl = data[:, 0]
    # Remove baseline, apply gain and exposure correction
    counts = (data[:, 1] - np.min(data[:, 1])) * 10 / exposure_time
    # Convert to spectral counts (counts/s/nm) while applying filters
    wl_centers = wl[:-1] + np.diff(wl) / 2
    spectral_counts = counts[:-1] * np.diff(wl) / (0.6 * 0.5)
    return wl_centers, spectral_counts

def efficiency_curve(k=1.18):
    camera_data = np.loadtxt("../measurement/2025-04-03/QEcurve.dat")
    wl_cam = camera_data[:, 0]
    qe_cam = camera_data[:, 1] / 100
    # Blaze function: simplified version using np.sinc for illustration
    blaze_eff = (np.sinc(k * (1 - 500 / wl_cam)))**2
    combined_eff = blaze_eff * qe_cam * 0.6
    mirrors = np.loadtxt("mirrors.csv", delimiter=",")
    mirror_func = interp1d(
        mirrors[:, 0]*1000, (mirrors[:, 1]/100)**2, 
        kind="cubic", fill_value="extrapolate"
    )
    # combined_eff *= mirror_func(wl_cam)
    return sp.interpolate.interp1d(wl_cam, combined_eff, bounds_error=False, fill_value=0)

def counts_to_power_density(wavelength, counts):
    energy_per_photon = 1240 / wavelength  # Joules per photon conversion (in W/nm)
    power_density = counts * energy_per_photon
    solid_angle = pi / 4
    f_rep = 40 * kilo
    area = pi * (50 * micro)**2
    emittance = power_density / (solid_angle * f_rep * area) / nano
    return emittance

if __name__ == "__main__":

    # Load and process measurement data
    data_files = sorted(glob("../measurement/2025-05-05/003 thermal*.asc"))
    wavelength_meas, raw_power = spectrum_data(data_files[0])
    raw_power = counts_to_power_density(wavelength_meas, raw_power)
    wl_range = slice(200, 1800)
    wavelength_meas, raw_power = wavelength_meas[wl_range], raw_power[wl_range]
    raw_power /= raw_power.max()

    # Apply efficiency correction to measurement data
    eff_func = efficiency_curve()
    power_corr = raw_power / eff_func(wavelength_meas)

    # Simulation results (uncorrected and corrected)
    sim = HotElectronSim(F_exc=385)
    sim_spec = sim.spectrum()
    sim_corr = sim_spec * eff_func(sim.wavelength_nm)

    # === Model fitting ===
    def model(x):
        F_exc, k, s = x
        sim_data = HotElectronSim(F_exc=F_exc)
        sim_corr = sim_data.spectrum() * efficiency_curve(k=k)(sim_data.wavelength_nm)
        mdl = interp1d(sim_data.wavelength_nm, sim_corr, bounds_error=False, fill_value="extrapolate")(wavelength_meas)
        return mdl * s

    def loss(x):
        return np.mean((model(x) - raw_power)**2) * 1e4

    x0 = [385, 1.18, 1e-3]
    opt = minimize(loss, x0, bounds=([200, 900], [0.1, 2], [0, np.inf]))

    print(f"F exc.:\t{opt.x[0]:.3g} J/m²")
    print(f"k.:\t{opt.x[1]:.3g}")
    print(f"scale:\t{opt.x[2]:.2g}")

    # Plot model fit and residuals

    plt.figure()

    # Plot raw measurement data
    plt.plot(wavelength_meas, raw_power, label="Raw", color="gray")

    # Compute and plot efficiency-corrected measurement data
    corrected = raw_power / efficiency_curve(k=opt.x[1])(wavelength_meas)
    plt.plot(wavelength_meas, corrected, label="Corrected")

    # Plot fitted simulation model
    sim_fit = HotElectronSim(F_exc=opt.x[0], wl_min_nm=350, wl_max_nm=980)
    plt.plot(sim_fit.wavelength_nm, sim_fit.spectrum() * opt.x[2], label="Model")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectral Radiance (J/m³/sr)")

    # Compute and plot residuals
    residual = (raw_power - model(opt.x)) / efficiency_curve(k=opt.x[1])(wavelength_meas)
    print(f"res:\t{np.std(residual):g}")
    # plt.plot(wavelength_meas, residual, label="Residual")
    # plt.axhline(0, color="k")

    secax = plt.gca().secondary_xaxis('top', functions=(lambda wl: 1240/wl, lambda ev: 1240/ev))
    secax.set_xlabel('Photon Energy (eV)')
    plt.ylim(0, None)
    plt.legend()
    plt.savefig("figures/combined.fit.pdf")
    plt.show()


# %%
