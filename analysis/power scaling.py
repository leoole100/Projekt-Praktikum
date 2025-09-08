# %%
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
from model import HotElectronSim
import numpy as np
from scipy.optimize import curve_fit

# F = np.linspace(0, 1000, 100)
F = np.geomspace(1e-3, 1000, 100)

sims = [HotElectronSim(
    F_exc=f,
    T_room=300
) for f in F]

Tmax = np.array([s.temperature().max() for s in sims])

def model(x, *p): return p[0] + p[1] * x**p[2] 

popt, pcov = curve_fit(model, F, Tmax, p0=[0., 1., 1.])

plt.plot(F, Tmax, label="numerical simulation")
plt.plot(F, model(F, *popt), label=f"P^{popt[2]:.2f}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Excitation (J/m²)")
plt.ylabel("Max Temperature (T)")
plt.legend()
plt.show()