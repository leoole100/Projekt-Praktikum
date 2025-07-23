# %%
from camera import Camera
import time
import yaml
import numpy as np

cam = Camera()
results = {"temperature_sweep": {}, "exposure_sweep": {}}

# Temperature sweep
cam.exposure = 10
temperatures = np.linspace(10, -30, 15)
print(f"Starting temperature sweep: {temperatures}")
for temp in temperatures:
    print(f"Setting temperature to {temp}°C")
    cam.temperature = temp
    cam.wait_for_cooldown()
    time.sleep(10)  # wait for the temperature to stabilize
    print(f"cam.temperature {cam.temperature}")
    frame = cam()
    results["temperature_sweep"][cam.temperature] = frame
    print(f"mean: {frame.mean()}, std: {frame.std()}")

# Exposure sweep with fixed temperature (use first temperature as example)
cam.temperature = -20
cam.wait_for_cooldown()
exposures = np.linspace(0.1, 100, 15)  # in seconds
print(f"Starting exposure sweep: {exposures}")
for exposure in exposures:
    print(f"Setting exposure to {exposure} seconds")
    cam.exposure = exposure
    frame = cam()
    results["exposure_sweep"][cam.exposure] = frame
    print(f"mean: {frame.mean()}, std: {frame.std()}")
    

# Save results to a file
print("saving")
with open("dark_noise.yaml", "w") as f:
    yaml.dump(results, f)
print("saved")
    
