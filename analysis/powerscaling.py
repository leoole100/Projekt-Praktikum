# %%
import matplotlib.pyplot as plt
plt.style.use("style.mplstyle")
from model import HotElectronSim, planck 
import numpy as np
from scipy.constants import *
from scipy.optimize import curve_fit

F = np.geomspace(1e7, 2e9, 20)

sims = [HotElectronSim(
    P_exc=f,
    T_room=300,
    t_max=1e-11,
    n_t=10000
) for f in F]

# check if sufficiently cooled down
for s in sims:
    assert s.temperature()[-1] < 330

Tmax = np.array([s.temperature().max() for s in sims])
P = np.array([np.sum(s.temperature()**4)*Stefan_Boltzmann*s.dt/np.pi for s in sims])
wl = np.linspace(400, 900, 100)
def _P_range(s:HotElectronSim):
    T = s.temperature()
    dwl = wl[1]-wl[0]
    return np.sum(planck(wl, T)*s.dt*dwl*nano)

P_range = np.array([_P_range(s) for s in sims])

def model(x, *p): return p[0] * x**p[1 ] 


# fit power laws to both curves and plot
mask = F > 0
popt_tot, pcov_tot = curve_fit(model, F[mask], P[mask], p0=[1e-8, 1.])
popt_rng, pcov_rng = curve_fit(model, F[mask], P_range[mask], p0=[1e-10, 1.])

xscale = 1e9
yscale = 1e-6

plt.plot(F/xscale, P/yscale, label="total")
plt.plot(F/xscale, P_range/yscale, label=f"{wl.min():g}-{wl.max():g} nm")


plt.xlabel(r"$P_V$ (GJ/m³)")
plt.ylabel(r"$H_\lambda$ (uJ/m²/sr)")
plt.xlim(0, None); plt.ylim(0, None)
# plt.xscale("log"); plt.yscale("log")
plt.legend()
plt.savefig("figures/powerscaling.pdf")
plt.show()

print(f"exponents:\t{popt_tot[1]:.3g}, {popt_rng[1]:.3g}")