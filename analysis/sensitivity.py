#%%
from dataclasses import replace
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import *

# Local module
import model

# ---------- Helpers ----------
def pretty_value(name: str, attr: str, value: float) -> str:
    """Return a nice string with proper unit prefix depending on attr."""
    if attr in ("tau_fwhm", "tau_eph"):          # convert s → ps
        return f"{value/femto:.3g} fs"
    elif attr == "d_abs":                        # convert m → nm
        return f"{value/nano:.3g} nm"
    elif attr == "F_exc":
        return f"{value:.3g} J/m²"
    else:
        return f"{value:.3g}"

# ---------- Sweep setup ----------
base = model.HotElectronSim()

# Values are chosen to span a sensible range around the defaults.
# Feel free to tweak to match your experiment.
# TODO: change to reflect names from calculations
sweep_plan = {
    # name: (attr, unit, iterable of absolute values)
    "Fluence": ("F_exc", "J/m²", [0.5*base.F_exc, base.F_exc, 2.0*base.F_exc]),
    "Pulse FWHM": ("tau_fwhm", "s", [0.5*base.tau_fwhm, 1*base.tau_fwhm, 2.0*base.tau_fwhm]),
    "Absorption depth": ("d_abs", "m", [0.5*base.d_abs, base.d_abs, 2.0*base.d_abs]),
    "e‑ph equil. time": ("tau_eph", "s", [0.5*base.tau_eph, base.tau_eph, 2.0*base.tau_eph]),
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

    ax.legend(title=title, bbox_to_anchor=(1.0, 1), loc="upper left")


fig.supxlabel("Photon energy (eV)", fontsize=10)
fig.supylabel("Spectrum (J/m²/sr/nm)", fontsize=10)

plt.savefig("figures/sensitivity.pdf")
plt.show()
