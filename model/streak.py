#%%
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("../style.mplstyle")
from model import Model

# %%
m = Model()
m.fwhm = 50e-15
m.t = np.linspace(-3*m.sigma, 1*(m.sigma+m.g), 1000)
m()

h = 6.626e-34	# J/Hz
c = 300e6		# m/s
kB= 1.381e-23	# J/K

def B(l, T): 
	return 1/l**5 * 1/(np.exp(h*c/(l*kB*T))-1)


l = np.linspace(200, 2500, 100)*1e-9
b = B(l[:, None], m.T_e[None, :])
t = m.t*1e15

fig, ax = plt.subplots(2,2, gridspec_kw={
	'height_ratios': [1, 3],
	'width_ratios': [3, 1]
})

ax[0,0].plot(t, m.T_e)
ax[0,0].set_ylabel(r"$T_e$ (k)")

ax[1,0].contourf(m.t*1e15, l/1e-9, b, vmin=0)
ax[1,0].sharex(ax[0,0])
ax[1,0].set_ylabel(r"$\lambda$ (nm)")
ax[1,0].set_xlabel("t (fs)")

def norm(a): return a/a.max()

ax[1,1].plot(norm(b.mean(axis=1)), l, label="mean")
# ax[1,1].plot(norm(B(l, m.T_e.max())), l, label="max T")
# ax[1,1].legend()
ax[1,1].set_xlabel("summed")
ax[1,1].set_ylim(l.min(), l.max())


ax[0,1].axis("off")
for a in ax.flatten(): a.label_outer()
plt.savefig("figures/steak view.pdf")
plt.show()

# %%
