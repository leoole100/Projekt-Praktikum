#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
plt.style.use("../style.mplstyle")
from streak import Streak
from model import Model

# %%
def streak(p, t=None, fwhm=250e-15):
	m = Model()
	m.fwhm = fwhm
	m.fluence = p
	s = Streak(m)
	s.l = np.arange(100, 5000, 100)*1e-9
	s.model.t = np.arange(-500e-15, 5000e-15, 1e-15)
	return s

fluence = np.linspace(1,15, 7)
st = [streak(p)() for p in fluence]

fig = plt.figure()
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], figure=fig)  
ax1 = fig.add_subplot(gs[0, 0])  # top-left
ax2 = fig.add_subplot(gs[1, 0])  # bottom-left
ax2.sharex(ax1)
ax1.label_outer()
ax2.set_xlim(left=-300, right=700)
ax3 = fig.add_subplot(gs[:, 1])  # right side

for s,c in zip(st, mpl.colormaps["viridis"](fluence/fluence.max())):
	ax1.plot(s.t/1e-15, s.S/1e6, color=c)
	ax2.plot(s.t/1e-15, s.T_e/1e3, color=c)
	ax3.plot(s.l/1e-9, s.sum/1e15, color=c)
ax2.set_xlabel("t (fs)")
ax1.set_ylabel("S (MW)")
ax2.set_ylabel("T (10³ K)")
ax3.set_xlabel(r"$\lambda$ (nm)")
ax3.set_ylabel(r"$\int$B dt")

plt.tight_layout()
plt.savefig("./figures/power sweep.pdf")
plt.show()