from spectrometer import Spectrometer
import numpy as np
import matplotlib.pyplot as plt
import time

# s = Spectrometer()

positions = np.arange(300, 1200, 100)

wavelengths = []
counts = []

for w in positions:
	s.monochromator.wavelength = w
	time.sleep(0.1)
	print(f"{w} nm")
	spec = s()
	s.save(f"../measurement/2025-04-28/001 {w}nm.yaml")
	wavelengths.append(spec[0])
	counts.append(spec[1])

for p, w, c in zip(positions, wavelengths, counts):
	plt.plot(w, c, label=f"{round(p)} nm")
plt.legend()
plt.show()
