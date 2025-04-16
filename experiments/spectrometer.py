from lineCamera import LineCamera
from monochromator import Monochromator
import numpy as np
from typing import NamedTuple
from functools import cached_property

class Spectrum(NamedTuple):
    wavelength: np.ndarray
    counts: np.ndarray

class Spectrometer():
    def __init__(self):
        self.lineCamera = LineCamera()
        self.monochromator = Monochromator()

    @cached_property
    def offsets(self):
        pixels = self.lineCamera().shape[0]
        density = self.monochromator.density
        center_px = 282
        scale = 50.78092367906066	# from Andor Solis / calibration
        return (np.arange(pixels)-center_px)*scale/density

    def __call__(self) -> Spectrum:
        return Spectrum(
            wavelength=self.monochromator.wavelength + self.offsets,
            counts = self.lineCamera()
        )
    
    def close(self):
        self.lineCamera.__del__()
        self.monochromator.close()	

    def __del__(self):
        self.close()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    spec = Spectrometer()

    spec.lineCamera.exposure = 0.1
    spec.monochromator.grating = 3
    spec.monochromator.wavelength = 600

    spectrum = spec()

    plt.plot(*spectrum)
    plt.show()