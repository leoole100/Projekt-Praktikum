#%%
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("../style.mplstyle")
from model import Model

# %%
m = Model()
m.fwhm = 10e-15
m.t = np.linspace(-3*m.sigma, 1*(m.sigma+m.g), 1000)

def source(t, dt=50e-15):
	return m.gauss_peak(t) + m.gauss_peak(t, t0=dt)

m.source_function = source
m()
plt.plot(m.t/1e-15, m.S/1e9)
plt.plot(m.t/1e-15, m.T_e/1e3)