import serial

class Monochromator:
    gratings = {
        1: 2400,
        2: 1200,
        3: 150,
    }

    def __init__(self):
        self.ser = serial.Serial("COM3", baudrate=9600)
        self.grating = 3

    def send(self, command: str) -> str:
        self.ser.write((command+"\r").encode())
        line = self.ser.readline().decode()
        rest = line[len(command):].strip() # Remove the echoed command part
        tokens = rest.split()
        if not tokens[-1] == "ok":
            print(tokens[-1])
        
        return tokens[0]

    @property
    def wavelength(self) -> float:
        return float(self.send("?NM"))

    @wavelength.setter
    def wavelength(self, value: float):
        self.send(f"{value:.3f} GOTO")

    @property
    def grating(self) -> int:
        return int(self.send("?GRATING"))

    @grating.setter
    def grating(self, number: int):
        self.send(f"{number} GRATING")

    @property
    def density(self) -> int:
        return self.gratings[self.grating]
    
    @property
    def info(self) -> dict:
        return {
            "center": self.wavelength,
            "grating": self.grating,
            "density": self.density
        }

    def close(self):
        self.ser.close()
