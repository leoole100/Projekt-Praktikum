from lineCamera import LineCamera
from monochromator import Monochromator
import numpy as np
from offset import offset
from time import time
import yaml

class Spectrometer():
    def __init__(self):
        self.lineCamera = LineCamera()
        self.monochromator = Monochromator()

    def __call__(self):
        return (
            self.monochromator.wavelength + offset(self.monochromator.wavelength),
            self.lineCamera()[::-1]
        )
    
    def close(self):
        self.lineCamera.__del__()
        self.monochromator.close()	

    def __del__(self):
        self.close()
    
    @property
    def info(self) -> dict:
        return {
            "camera": self.lineCamera.info,
            "monochromator": self.monochromator.info,
        }
    
    def save(self, path: str):
        spectrum = self()
        data = {
            "time": time(),
            "metadata": self.info,
            "spectrum": {
                "wavelength": spectrum.wavelength.tolist(),
                "counts": spectrum.counts.tolist(),
            }
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    spec = Spectrometer()

    spec.lineCamera.exposure = 0.1
    spec.monochromator.grating = 3
    spec.monochromator.wavelength = 600

    print("Waiting for cooldown")
    spec.lineCamera.wait_for_cooldown()
    
    spectrum = spec()

    spec.close()

    plt.plot(*spectrum)
    plt.show()