#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
import scipy as sp
plt.style.use("../style.mplstyle")
from streak import Streak
from model import Model
from scipy.optimize import curve_fit

# %%
def streak(p, t=None, fwhm=250e-15):
	m = Model()
	m.fwhm = fwhm
	m.fluence = p
	s = Streak(m)
	s.l = np.arange(100, 5000, 100)*1e-9
	s.model.t = np.arange(-500e-15, 5000e-15, 1e-15)
	return s

fluence = np.geomspace(0.1,20, 20)
st = [streak(p)() for p in fluence]
pwr =  np.array([s.power for s in st])/1e15

def fit_function(x, a): return a * x**(4/3)
popt, pcov = curve_fit(fit_function, fluence, pwr)

fig = plt.figure()
plt.plot(fluence, pwr, label="numerical Model")
plt.plot(fluence, fit_function(fluence, *popt), label="^(4/3) Fit", linestyle="--")
plt.xscale("log")
plt.yscale("log")

plt.legend()
plt.xlabel("fluence (J/m²)")
plt.ylabel("Radiated Power")
plt.savefig("./figures/power sweep.pdf")
plt.show()