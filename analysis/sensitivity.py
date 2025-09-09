#%%
from dataclasses import replace
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import *

# Local module
import model
from matplotlib.ticker import MaxNLocator

# ---------- Helpers ----------
def pretty_value(name: str, attr: str, value: float) -> str:
    """Return a nice string with proper unit prefix depending on attr."""
    if attr in ("tau_fwhm", "tau_eph"):          # convert s → ps
        return f"{value/femto:.3g} fs"
    elif attr == "d_abs":                        # convert m → nm
        return f"{value/nano:.3g} nm"
    elif attr == "P_exc":
        return f"{value/giga:.3g} GJ/m³"
    elif attr == "T_room":
        return f"{value:.3g} K"
    else:
        return f"{value:.3g}"

# ---------- Sweep setup ----------
base = model.HotElectronSim(
    wl_min_nm=100,
    wl_max_nm=1600
)

# Values are chosen to span a sensible range around the defaults.
# Feel free to tweak to match your experiment.
# TODO: change to reflect names from calculations
sweep_plan = {
    # name: (attr, unit, iterable of absolute values)
    "$U_{abs}$": ("P_exc", "J/m³", [0.5*base.P_exc, base.P_exc, 2.0*base.P_exc]),
    "$t$": ("tau_fwhm", "s", [0.5*base.tau_fwhm, 1*base.tau_fwhm, 2.0*base.tau_fwhm]),
    r"$\tau$": ("tau_eph", "s", [0.5*base.tau_eph, base.tau_eph, 2.0*base.tau_eph]),
    "$T_l$": ("T_room", "L", [0.5*base.T_room, base.T_room, 2.0*base.T_room]),
}

# ---------- Plotting ----------
plt.style.use("style.mplstyle") if os.path.exists("style.mplstyle") else None

nrows = 2
ncols = 2
fig, axes = plt.subplots(nrows, ncols, constrained_layout=True, sharex=True, sharey=True, figsize=(6,3))
axes = axes.ravel()

colors = [plt.get_cmap("viridis")(i) for i in np.linspace(0, 1, 3)]

for ax in axes:
    ax.set_visible(False)

for idx, (title, (attr, unit, values)) in enumerate(sweep_plan.items()):
    ax = axes[idx]
    ax.set_visible(True)

    for v,c,s  in zip(values, colors, ["-", "--", "-"]):
        sim_new = replace(base, **{attr: float(v)})
        ax.plot(
            sim_new.wavelength_nm, sim_new.spectrum(), s, 
            label=pretty_value(title, attr, v),
            color=c
        )

    ax.legend(title=title)
    ax.set_ylim(0, None)
    ax.set_xlim(base.wavelength_nm.min(), base.wavelength_nm.max())

def nm_eV(x): return 1240 / x

# add top axis only for bottom-row subplots so it appears once per column
for i, ax in enumerate(axes):
    if not ax.get_visible():
        continue
    sec = ax.secondary_xaxis('top', functions=(nm_eV, nm_eV))
    if i < ncols:  # top row
        sec.set_xlabel("Energy (eV)")
        sec.set_ticks([1, 2, 3, 4, 5])
    else:  # bottom row
        sec.set_ticks([1, 2, 3, 4, 5])
        sec.tick_params(labeltop=False)
        ax.set_xlabel("Wavelength (nm)")

fig.supylabel("Spectrum (J/m³/sr)", fontsize=10)

plt.savefig("figures/sensitivity.pdf")
plt.show()
