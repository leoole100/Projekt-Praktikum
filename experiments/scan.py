from spectrometer import Spectrometer
import numpy as np
import matplotlib as mpl
import time

s = Spectrometer()

wl = np.arange(300, 1200, 100)

for w in wl:
	s.monochromator.wavelength = w
	print(w)
	time.sleep(1)
	print(f"{w}nm")
	s()
	s.save(f"../measurement/2025-04-25/005 {w}.yaml")