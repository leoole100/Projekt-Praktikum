# %%
from pylablib.devices import Andor
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
from IPython import display


#%%
cam = Andor.AndorSDK2Camera(temperature=-80, fan_mode="full")

cam.set_temperature(-80)
cam.get_temperature()

cam.get_capabilities()

cam.set_read_mode("image")

# cam.setup_shutter("closed")
cam.setup_shutter("open")

cam.set_exposure(0)

# %%
plt.imshow(cam.grab(frame_timeout=999)[0])
plt.colorbar()
plt.show()

# %%

# def frame_gen():
# 	while True:
# 		yield cam.grab(frame_timeout=999)[0]

# f = frame_gen()


# fig, ax = plt.subplots(1,1)
# plot = ax.imshow(next(f))

# while True:
# 	plot.set_data(next(f))
# 	fig.canvas.draw()
# 	display.display(fig)
# 	display.clear_output(wait=True)

#%%

cam.close()