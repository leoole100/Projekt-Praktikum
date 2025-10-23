# %%
import matplotlib.pyplot as plt
plt.style.use("style.mplstyle")
from model import HotElectronSim, planck 
import numpy as np
from scipy.constants import *
from scipy.optimize import curve_fit

# %%

# F = np.linspace(1e7, 2e9, 100)
F = np.geomspace(1e6, 2e9, 50)

sims = [HotElectronSim(
    P_exc=f,
    T_room=300,
    t_max=1e-11,
    n_t=1000,
    wl_min_nm=400,
    wl_max_nm=900,
    n_wl = 100
) for f in F]

# check if sufficiently cooled down
# for s in sims:
    # assert s.temperature()[-1] < (s.T_room+30)

Tmax = np.array([s.temperature().max() for s in sims])
P = np.array([np.sum(s.temperature()**4)*Stefan_Boltzmann*s.dt/np.pi for s in sims])
# P_range = np.array([np.sum(s.spectrum()*np.diff(s.wavelength_nm)) for s in sims])
P_range = np.array([np.trapz(s.spectrum(), s.wavelength_nm*nano) for s in sims])

def model(x, *p): return p[0] * x**p[1 ] 


# fit power laws to both curves and plot
mask = F > 1e-3*giga
popt_tot, pcov_tot = curve_fit(model, F[mask], P[mask], p0=[1e-8, 1.])
popt_rng, pcov_rng = curve_fit(model, F[mask], P_range[mask], p0=[1e-10, 1.])

xscale = 1e9
yscale = 1e-6

l = plt.plot(F/xscale, P/yscale, label="total")
# plt.plot(F/xscale, model(F, *popt_tot)/yscale, "--", color=l[0].get_color())

l = plt.plot(F/xscale, P_range/yscale, label=f"400-900 nm")
# plt.plot(F/xscale, model(F, *popt_rng)/yscale, "--", color=l[0].get_color())

plt.xlabel(r"$U_\text{abs}$ (GJ/m³)")
plt.ylabel(r"Thermal emission (uJ/m²/sr)")
# plt.xlim(0, None); plt.ylim(0, None)
plt.yscale("log"); plt.xscale("log")
plt.legend()

plt.ylim(1e-5, None)

ax2 = plt.gca().twinx()
ax2.plot(F/xscale, Tmax, "--", color="C2")
ax2.set_ylabel("$T_{max}$ (K)", color="C2")
ax2.tick_params(axis="y", colors="C2")
# plt.ylim(0, None)

plt.savefig("figures/powerscaling.pdf")
# plt.show()

# %%
for s,p in list(zip(sims, F))[::10]:
    plt.plot(s.time, s.temperature(), ".-", label=p)
plt.xlim(None, 0.2e-11)
plt.legend()

#%%
for s,p in list(zip(sims, F))[::10]:
    plt.plot(s.wavelength_nm, s.spectrum(), ".-", label=p)
plt.legend()
