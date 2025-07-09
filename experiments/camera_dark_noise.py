# %%
from camera import Camera
import time
import yaml
import numpy as np

cam = Camera()
results = {"temperature_sweep": {}, "exposure_sweep": {}}

# Temperature sweep
cam.exposure = 1
temperatures = np.linspace(0, -80, 8)
print(f"Starting temperature sweep: {temperatures}")
for temp in temperatures:
    print(f"Setting temperature to {temp}°C")
    cam.temperature = temp
    cam.wait_for_cooldown()
    time.sleep(1)  # wait for the temperature to stabilize
    results["temperature_sweep"][temp] = cam()

# Exposure sweep with fixed temperature (use first temperature as example)
cam.temperature = -80
cam.wait_for_cooldown()
exposures = np.geomspace(0.1, 10, 9)  # in seconds
print(f"Starting exposure sweep: {exposures}")
for exposure in exposures:
    print(f"Setting exposure to {exposure} seconds")
    cam.exposure = exposure
    results["exposure_sweep"][exposure] = cam()

# Save results to a file
with open("../measurements/2025-07-09/dark_noise.yaml", "w") as f:
    yaml.dump(results, f)
    