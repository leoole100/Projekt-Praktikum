#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
plt.style.use("../style.mplstyle")
from streak import Streak
from model import Model
from scipy.optimize import curve_fit
from scipy.special import wofz

# %%
def double_streak(dt=50e-15, t=None, fwhm=50e-15):
	m = Model()
	m.fwhm = fwhm
	s = Streak(m)
	if not t is None: s.model.t = t
	def double(t, dt=dt): return (s.model.gauss_peak(t) + s.model.gauss_peak(t, t0=dt))/2
	s.model.source_function = double
	return s

# dt = np.linspace(10,500, 7)*1e-15
dt = np.linspace(0,700, 70)*1e-15
st = [double_streak(d, t=np.arange(-500e-15, 5000e-15, 1e-15), fwhm=50e-15)() for d in dt]
power = np.array([s.power for s in st])
power /= power[-1]

# %%

plt.figure()
mirrored_dt = -np.flip(dt)
mirrored_power = np.flip(power)
full_dt = np.concatenate((mirrored_dt, dt))
full_power = np.concatenate((mirrored_power, power))

# Define the Voigt profile function
def voigt(x, amplitude, sigma, gamma):
	z = (x + 1j * gamma) / (sigma * np.sqrt(2))
	z = np.real(wofz(z))/ (sigma * np.sqrt(2 * np.pi))
	z /= np.max(z)
	return amplitude *  z + 1

popt = [0.2, st[0].model.sigma/2, st[0].model.g/2]
# popt, pcov = curve_fit(voigt, full_dt / 1e-15, full_power, p0=popt)
fitted_curve = voigt(full_dt, *popt)

plt.plot(full_dt / 1e-15, full_power, label="Numerical")
plt.plot(full_dt / 1e-15, fitted_curve, "--", label="Voigt Fit")
plt.legend()
plt.xlabel("lag (ps)")
plt.ylabel("normalized Power")
plt.savefig("figures/autocorrelation.pdf")
plt.show()

# %%
