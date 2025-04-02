#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
plt.style.use("../style.mplstyle")
from streak import Streak

# %%
def double_streak(dt=50e-15, t=None):
	s = Streak()
	s.model.fwhm = 10e-15
	if not t is None: s.model.t = t
	def double(t, dt=dt): return (s.model.gauss_peak(t) + s.model.gauss_peak(t, t0=dt))/2
	s.model.source_function = double
	return s

s = double_streak()()
# Plot example
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(s.t/1e-15, s.T_e/1e3)
ax[0].set_ylabel("T (10³ K)")
ax[1].contourf(s.t/1e-15, s.l/1e-9, s.b)
ax[1].set_xlabel("t (fs)")
ax[1].set_ylabel("λ (nm)")
plt.savefig("./figures/double_pump.pdf")
plt.show()

#%%
dt = np.linspace(10,200, 5)*1e-15
st = [double_streak(d, t=np.arange(-50e-15, 500e-15, 1e-15))() for d in dt]

fig = plt.figure()
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])  
ax1 = fig.add_subplot(gs[0, 0])  # top-left
ax2 = fig.add_subplot(gs[1, 0])  # bottom-left
ax1.sharex(ax1)
ax1.set_xticklabels([])
ax3 = fig.add_subplot(gs[:, 1])  # right side

for s,c in zip(st, mpl.colormaps["viridis"](dt/dt.max())):
	ax1.plot(s.t/1e-15, s.S/1e9, color=c)
	ax2.plot(s.t/1e-15, s.T_e/1e3, color=c)
	ax3.plot(s.l/1e-9, s.sum/1e15, color=c)
ax2.set_xlabel("t (fs)")
ax1.set_ylabel("S (GW)")
ax2.set_ylabel("T (10³ K)")
ax3.set_xlabel(r"$\lambda$ (nm)")
ax3.set_ylabel(r"$\int$B dt")

plt.tight_layout()
plt.savefig("./figures/double pump sweep.pdf")
plt.show()

