from pylablib.devices import Andor

class Camera:
    def __init__(self):
        self.cam = Andor.AndorSDK2Camera(fan_mode="full")
        self.cam.set_read_mode("image")
        self.temperature = -80

    @property
    def temperature(self):
        return self.cam.get_temperature()

    @temperature.setter
    def temperature(self, value):
        self.cam.set_temperature(value, enable_cooler=True)

    @property
    def exposure(self):
        return self.cam.get_exposure()
    
    @exposure.setter
    def exposure(self, value):
        self.cam.set_exposure(value)

    def __call__(self):
        return self.cam.grab(frame_timeout=self.exposure+1)[0]

    def wait_for_cooldown(self):
        while self.temperature > self.cam.get_temperature_setpoint():
            pass

    def __del__(self):
        self.cam.close()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    plt.ion()

    cam = Camera()
    cam.exposure = config["exposure"]


    fig, ax = plt.subplots()
    img = ax.imshow(cam())
    plt.colorbar(img, ax=ax)
    plt.show()

    running = True

    def handle_close(event):
        global running
        running = False

    fig.canvas.mpl_connect("close_event", handle_close)

    try:
        while running:
            frame = cam()
            img.set_data(frame)
            img.set_clim(vmin=frame.min(), vmax=frame.max()) # autoscale
            fig.canvas.draw()
            fig.canvas.flush_events()
    finally:
        del cam
        print("Camera safely closed.")