# Sensitivity: $\partial f(p) / \partial p  * p/P(p) = lim s->0: (P(p+sp)/P(p) - 1)/s$
# %%
from curses import noqiflush
import matplotlib.pyplot as plt

plt.style.use("../style.mplstyle")
from networkx import power
import numpy as np
from math import pi
from scipy.constants import h, c, k
from labellines import labelLines
from glob import glob
import scipy as sp
from scipy.interpolate import interp1d
from uncertainties import ufloat
from uncertainties.unumpy import uarray
import uncertainties.unumpy as unp

from model import run_monte_carlo_simulation, LaserParameters, MaterialParameters, SimulationParameters, run_single_simulation

import os

# %%

lparam = LaserParameters(
    P_avg_uncertainty=0,
    tau_uncertainty=0,
    spot_uncertainty=0
)

mparam = MaterialParameters(
    d_uncertainty = 0,
    rho_uncertainty = 0,
    g_uncertainty = 0
)

sparam = SimulationParameters()

params = {
    r"$P_{avg}$":   (lparam, "P_avg"),
    r"$\tau_{fwhm}$": (lparam, "tau_fwhm"),
    r"$d_{spot}$": (lparam, "spot_diameter"),
    r"$d_{abs}$":   (mparam, "d"),
    r"$g$":   (mparam, "g"),
    r"$T_l$":   (mparam, "T_room"),
}

base_line = run_single_simulation(lparam, mparam, sparam)

change = 0.10
results = {}
for k, (obj, attr) in params.items():
    orig = getattr(obj, attr)
    v = setattr(obj, attr, orig * (1+change))
    results[k] = run_single_simulation(lparam, mparam, sparam)
    v =  setattr(obj, attr, orig)  

wavelength = np.linspace(sparam.wavelength_min, sparam.wavelength_max, sparam.n_wavelengths)

# plot how the spectrum changes
# ax = plt.subplot(2, 1, 1)
plt.plot(wavelength, base_line[2], color="black")
for k, v in results.items():
    plt.plot(wavelength, v[2], label=k)
plt.legend()
# plt.xlabel("Wavelength (nm)")
plt.ylabel("Power density\n(W/m)")
# ax.tick_params(labelbottom=False)

# plt.subplot(2, 1, 2, sharex=ax)
# for k, v in results.items():
#     plt.plot(wavelength, (v[2]/base_line[2]-1)/change, label=k)
# plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Sensitivity")
plt.savefig("figures/sensitivity spectrum.pdf")
plt.show()

# get the relative change
realtive_change = {
    k: (np.sqrt(np.mean((v[2]/base_line[2] - 1)**2)))/change 
    for k, v in results.items()
}
plt.bar(realtive_change.keys(), realtive_change.values(), color=["C0", "C1", "C2", "C3", "C4", "C5"])
plt.grid(False)
plt.ylabel("RMS Sensitivity")
plt.savefig("figures/sensitivity bars.pdf")
plt.show()


# %%
