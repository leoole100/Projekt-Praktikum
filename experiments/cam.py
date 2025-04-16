# %%
from pylablib.devices import Andor
import matplotlib.pyplot as plt
# plt.style.use("../style.mplstyle")
import matplotlib as mpl
mpl.use("Qt5Agg")

#%%
cam = Andor.AndorSDK2Camera(temperature=-80, fan_mode="full")

cam.set_temperature(-80)
cam.get_temperature()

cam.set_read_mode("image")
# cam.get_capabilities()


# cam.setup_shutter("closed")
# cam.setup_shutter("open")

cam.set_exposure(0.1)

print("camera ready")

# %%

def frame():
    return cam.grab(frame_timeout=999)[0]

plt.ion()

fig, ax = plt.subplots(1,1)
img = ax.imshow(frame())
plt.show()

def loop():
    img.set_data(frame())
    fig.canvas.draw()
    fig.canvas.flush_events()

while True: loop()


#%%

cam.close()
