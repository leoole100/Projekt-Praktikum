#%%
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("../style.mplstyle")
from model import Model
from streak import plot_streak, B, streak
import xarray as xr

# %%
m = Model()
m.fwhm = 10e-15
m.t = np.linspace(-3*m.sigma, .7*(m.sigma+m.g), 1000)
def double(t, dt=50e-15): return (m.gauss_peak(t) + m.gauss_peak(t, t0=dt))/2
m.source_function = double
m()
plot_streak(m)
plt.gcf().axes[0].plot(m.t/1e-15, m.S/1e9, "--")
plt.gcf().axes[2].plot(m.t[100:]/1e-15, 2.9/m.T_e[100:]/1e-6, "w")
plt.savefig("./figures/double_pump.pdf")
# %%
m()
streak(m).sum(axis=1)