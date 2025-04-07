#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
plt.style.use("../style.mplstyle")
from streak import Streak
from model import Model

# %%
def double_streak(dt=50e-15, t=None, fwhm=50e-15):
	m = Model()
	m.fwhm = fwhm
	s = Streak(m)
	if not t is None: s.model.t = t
	def double(t, dt=dt): return (s.model.gauss_peak(t) + s.model.gauss_peak(t, t0=dt))/2
	s.model.source_function = double
	return s

dt = np.linspace(10,500, 7)*1e-15
st = [double_streak(d, t=np.arange(-500e-15, 5000e-15, 1e-15), fwhm=50e-15)() for d in dt]

fig = plt.figure()
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], figure=fig)  
ax1 = fig.add_subplot(gs[0, 0])  # top-left
ax2 = fig.add_subplot(gs[1, 0])  # bottom-left
ax2.sharex(ax1)
ax1.label_outer()
ax2.set_xlim(left=-150, right=700)
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
ax3.axvspan(400,800, color="lightgray")
ax3.set_xlim(200, 1200)

plt.tight_layout()
plt.savefig("./figures/double pump sweep.pdf")
plt.show()


# %%
