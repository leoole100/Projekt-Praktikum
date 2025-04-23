from camera import Camera

class LineCamera(Camera):
    def __init__(self):
        super().__init__()
        self.cam.set_read_mode("fvb") # full vertical binning
    
    def __call__(self):
    	return super().__call__()[0]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    plt.ion()

    cam = LineCamera()
    cam.exposure = config["exposure"]


    fig, ax = plt.subplots()
    frame = cam()
    line = plt.plot(range(len(frame)), frame)[0]
    plt.show()

    running = True

    def handle_close(event):
        global running
        running = False

    fig.canvas.mpl_connect("close_event", handle_close)

    try:
        while running:
            frame = cam()
            line.set_data(range(len(frame)), frame)
            ax.relim()             # Recalculate limits based on data
            ax.autoscale_view()    # Autoscale the axes
            fig.canvas.draw()
            fig.canvas.flush_events()
    finally:
        del cam
        print("Camera safely closed.")