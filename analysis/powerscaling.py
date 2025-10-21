# %%
import matplotlib.pyplot as plt
plt.style.use("style.mplstyle")
from model import HotElectronSim, planck 
import numpy as np
from scipy.constants import *
from scipy.optimize import curve_fit

# F = np.geomspace(1e3, 1e11, 100)
F = np.geomspace(1e7, 1e10, 100)

sims = [HotElectronSim(
    P_exc=f,
    T_room=300
) for f in F]

Tmax = np.array([s.temperature().max() for s in sims])
P = np.array([np.sum(s.temperature()**4)*Stefan_Boltzmann*s.dt/np.pi for s in sims])
wl = np.linspace(400, 900, 100)
def _P_range(s:HotElectronSim):
    T = s.temperature()
    dwl = wl[1]-wl[0]
    return np.sum(planck(wl, T)*s.dt*dwl*nano)

P_range = np.array([_P_range(s) for s in sims])

def model(x, *p): return p[0] * x**p[1 ] 

plt.plot(F, P, label="total")
plt.plot(F, P_range, label=f"{wl.min():g}-{wl.max():g} nm")

mask = F > 1e9
popt, pcov = curve_fit(model, F[mask], P[mask], p0=[1e-8, 1.])
plt.plot(F, model(F, *popt),"--", color="gray", 
    label=r"$\propto U^{" + f"{popt[1]:.2f}" + r"}$"
)
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Radiant Fluence (J/m²/sr)")
plt.xlabel("Absorbed Power Density (J/m³)")
plt.ylim(1e-10, None)
plt.legend()

plt.tight_layout()
plt.savefig("figures/powerscaling.pdf")
plt.show()