# %%
import os
import matplotlib.pyplot as plt
from spectrometer import Spectrometer
import numpy as np

# Ensure the output directory exists
output_dir = "../measurement/2025-07-23"
os.makedirs(output_dir, exist_ok=True)

print("connecting")

spec = Spectrometer()

spec.lineCamera.exposure = 0.1
spec.lineCamera.temperature = -80
spec.monochromator.wavelength = 350

print("Waiting for cooldown")
spec.lineCamera.wait_for_cooldown()

print('\a')

exposure_times = np.geomspace(0.001, 5, 20)
n = 100
print(f"estimated time: {np.sum(exposure_times)*n/60} min")

for et in exposure_times:
    spectra = []
    for i in range(n):
        spec.lineCamera.exposure = et
        s = spec()
        print(f"taking {i}/{n} {et}s, min: {s[1].min()/2**14:.0%}, max: {s[1].max()/2**14:.0%}")
        spectra.append(s)

    filename = os.path.join(output_dir, f"005 {et}s.npy")
    np.save(filename, np.array(spectra))

print('\a')

# spec.close()
