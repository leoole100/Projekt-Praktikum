#%%
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("../style.mplstyle")
from model import Model
from streak import Streak

# %%
s = Streak(Model())
s.model.fwhm = 10e-15
def double(t, dt=50e-15): return (s.model.gauss_peak(t) + s.model.gauss_peak(t, t0=dt))/2
s.model.source_function = double
s()

# Plot example
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(s.t/1e-15, s.T_e/1e3)
ax[0].set_ylabel("T (10³ K)")
ax[1].contourf(s.t/1e-15, s.l/1e-9, s.b)
ax[1].set_xlabel("t (fs)")
ax[1].set_ylabel("λ (nm)")
plt.savefig("./figures/double_pump.pdf")
plt.show()
