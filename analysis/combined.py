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

def solid_angle_from_aperture(aperture_radius, focal_length):
    theta = np.arctan2(aperture_radius, focal_length)
    return 2 * pi * (1 - np.cos(theta))

def spectrum_data(filepath, exposure_time=2):
    data = np.loadtxt(filepath)
    wl = data[:, 0]
    wl_centers_nm = wl[:-1] + np.diff(wl) / 2
    dwl = np.diff(wl)*nano # m/bin
    
    # Remove baseline, apply gain and exposure correction
    baseline = np.min(data[:, 1])
    gain = 10
    photons = (data[:, 1] - baseline) / gain
    photons_per_s = photons / exposure_time
    
    photons_per_s_m = photons_per_s[:-1] / dwl
    return wl_centers_nm, photons_per_s_m

def counts_to_power_density(wavelength_nm, counts_per_s_m):
    energy_per_photon = h*c / (wavelength_nm*nano)  # J per photon
    power_density = counts_per_s_m * energy_per_photon # J/s/m
    solid_angle = solid_angle_from_aperture(1, 2)
    f_rep = 40 * kilo
    area = pi * (25 * micro)**2
    emittance = power_density / solid_angle / area / f_rep
    return emittance # J/m³/sr

def efficiency_curve(k=1.18):
    camera_data = np.loadtxt("../measurement/2025-04-03/QEcurve.dat")
    wl_nm = camera_data[:, 0]
    qe_cam = camera_data[:, 1] / 100
    # Blaze function: simplified version using np.sinc for illustration
    blaze_eff = (np.sinc(k * (1 - 500 / wl_nm)))**2 * 0.7
    combined_eff = blaze_eff * qe_cam * 0.6 # 0.6 is the HR filter
    mirrors = np.loadtxt("mirrors.csv", delimiter=",")
    mirror_func = interp1d(
        mirrors[:, 0]*1000, (mirrors[:, 1]/100)**2, 
        kind="cubic", fill_value="extrapolate"
    )
    combined_eff *= mirror_func(wl_nm)
    return sp.interpolate.interp1d(wl_nm, combined_eff, bounds_error=False, fill_value=0)

if __name__ == "__main__":
    # Load and process measurement data
    data_files = sorted(glob("../measurement/2025-05-05/003 thermal*.asc"))
    wavelength_meas, counts_per_m_s = spectrum_data(data_files[0])
    emittance = counts_to_power_density(wavelength_meas, counts_per_m_s)
    mask = (wavelength_meas >= 375) & (wavelength_meas <= 930)
    wavelength_meas, emittance = wavelength_meas[mask], emittance[mask]


    # === Model fitting ===
    def model(x):
        P_exc, k, s = x
        sim_data = HotElectronSim(P_exc=P_exc*1e9)
        sim_corr = sim_data.spectrum() * efficiency_curve(k=k)(sim_data.wavelength_nm)
        mdl = interp1d(sim_data.wavelength_nm, sim_corr, bounds_error=False, fill_value="extrapolate")(wavelength_meas)
        return mdl * s

    def loss(x):
        return np.mean((model(x) - emittance)**2) * 1e4

    x0 = [15, 1.0, 0.1]
    opt = minimize(loss, x0, bounds=([0, np.inf], [0.1, 2], [0, np.inf]))
    opt.x[0] *= 1e9

    print(f"F exc.:\t{opt.x[0]:.3g} J/m³")
    print(f"k.:\t{opt.x[1]:.3g}")
    print(f"scale:\t{opt.x[2]:.2g}")

    # ############## Plot model fit and residuals ##############
    plt.figure()

    # Plot raw measurement data
    plt.plot(wavelength_meas, emittance, label="Raw", color="gray")

    # Compute and plot efficiency-corrected measurement data
    corrected = emittance / efficiency_curve(k=opt.x[1])(wavelength_meas)
    plt.plot(wavelength_meas, corrected, label="Corrected")

    # Plot fitted simulation model
    sim_fit = HotElectronSim(P_exc=opt.x[0], wl_min_nm=350, wl_max_nm=980)
    plt.plot(sim_fit.wavelength_nm, sim_fit.spectrum() * opt.x[2], label="Model")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectrum (J/m³/sr)")

    # Compute and plot residuals
    residual = (emittance - model(opt.x)) / efficiency_curve(k=opt.x[1])(wavelength_meas)
    print(f"res:\t{np.std(residual):g}")

    # secax = plt.gca().secondary_xaxis('top', functions=(lambda wl: 1240/wl, lambda ev: 1240/ev))
    # secax.set_xlabel('Photon Energy (eV)')
    plt.ylim(0, None)
    plt.xlim(wavelength_meas.min(), wavelength_meas.max())
    plt.legend()
    plt.savefig("figures/combined.fit.pdf")
    plt.show()

    # ################# Plot the efficiency   ##################
    
    # %%

    camera_data = np.loadtxt("../measurement/2025-04-03/QEcurve.dat")
    camera_data = camera_data[camera_data[:, 0].argsort()]  # Sort by wavelength
    wl_nm = camera_data[:, 0]
    qe_cam = camera_data[:, 1]
    plt.plot(wl_nm, qe_cam/100, label="camera")

    mirror = np.loadtxt("mirrors.csv", delimiter=",")
    mask = mirror[:,0]*1000<1000
    plt.plot(mirror[mask,0]*1000, (mirror[mask,1]/100)**2, label="mirrors")

    blaze_eff = (np.sinc(opt.x[1] * (1 - 500 / wl_nm)))**2 * 0.8
    plt.plot(wl_nm, blaze_eff, label="grating")

    plt.plot(wl_nm, efficiency_curve(k=opt.x[1])(wl_nm)/0.6, label="combined", color="k")

    np.save("expected_efficiency.npy", 
        np.array([wl_nm, efficiency_curve(k=opt.x[1])(wl_nm)])
    )

    plt.legend(frameon=True)
    # plt.yscale("log"); plt.ylim(1e-2, 1)
    plt.ylim(0, 1)
    plt.xlim(300, 1100)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Efficiency")
    plt.savefig("figures/combined.efficiency.pdf")
    plt.show()
# %%
