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

dt = np.linspace(0,700, 100)*1e-15
st = [double_streak(d, t=np.arange(-500e-15, 5000e-15, 1e-15), fwhm=50e-15)() for d in dt]
power = np.array([s.power for s in st])
power /= power[-1]

power_det = np.array([np.mean(s.sum * s.l) for s in st])
power_det /= power_det[-1]

# %%

plt.figure()
# Define the Voigt profile function
# def voigt(x, amplitude, sigma, gamma):
# 	z = (x + 1j * gamma) / (sigma * np.sqrt(2))
# 	z = np.real(wofz(z))/ (sigma * np.sqrt(2 * np.pi))
# 	z /= np.max(z)
# 	return amplitude *  z + 1

# popt = [0.2, st[0].model.sigma/2, st[0].model.g/2]
# popt, pcov = curve_fit(voigt, full_dt / 1e-15, full_power, p0=popt)
# fitted_curve = voigt(dt, *popt)

plt.plot(dt / 1e-15, power, label=r"$\int B\;dt\; d\lambda$")
# plt.plot(dt / 1e-15, power_det, label=r"$\int B\lambda\;dt\; d\lambda$")
# plt.plot(dt / 1e-15, fitted_curve, "--", label="Voigt Fit")
plt.legend()
plt.xlabel("lag (ps)")
plt.ylabel("normalized Power")
plt.savefig("figures/autocorrelation.pdf")
plt.show()

# %%
# Create a 2D array for the autocorrelation at each wavelength
wavelengths = st[0].l
autocorrelation = np.array([s.sum for s in st])

# add 5% noise
# autocorrelation = np.array([s.sum + s.sum*0.05 * np.random.normal(size=s.sum.shape) for s in st])

# Normalize the autocorrelation for each wavelength
# autocorrelation /= np.max(autocorrelation, axis=0)

# Plot the 2D autocorrelation
plt.figure()
plt.contourf(dt / 1e-15, wavelengths * 1e9, autocorrelation.T, vmin=0, 	levels=17)
plt.xlabel("lag (ps)")
plt.ylabel("Wavelength (nm)")
plt.savefig("figures/autocorrelation spectrum.pdf")
plt.show()