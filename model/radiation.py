#%%
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("../style.mplstyle")
from math import pi
from model import Model

# %%
m = Model()
m()

h = 6.626e-34	# J/Hz
c = 300e6		# m/s
kB= 1.381e-23	# J/K

def B(l, T): 
	return 1/l**5 * 1/(np.exp(h*c/(l*kB*T))-1)


l = np.linspace(400, 2000, 100)*1e-9
b = B(l[:, None], m.T_e[None, :])
t = (m.t-m.t0)*1e15

fig, ax = plt.subplots(2,2, gridspec_kw={
	'height_ratios': [1, 2],
	'width_ratios': [2, 1]
})

ax[0,0].plot(t, m.T_e)
ax[0,0].set_ylabel("$T_e$ (k)")

ax[1,0].contourf((m.t-m.t0)*1e15, l/1e-9, b, vmin=0)
ax[1,0].sharex(ax[0,0])
ax[1,0].set_ylabel("$\lambda$ (nm)")
ax[1,0].set_xlabel("t (fs)")


ax[1,1].plot(b.sum(axis=1), l)
ax[1,1].set_xticks([])
ax[1,1].set_xlabel("summed counts")

ax[0,1].axis("off")
for a in ax.flatten(): a.label_outer()
plt.savefig("figures/steak view.pdf")
plt.show()

# %%
