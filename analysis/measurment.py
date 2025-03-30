"""
implements a basic measurement object and function for manipulating measurement data
"""

#%%

import yaml
import numpy as np
from os.path import exists
class Measurement:
    def __init__(self, path: str):
        self.path = path
        self.name = self.path.rsplit("data/", 1)[-1]
        self._array = None  # stores np.ndarray
        self.attrs = {}

        # Load metadata from YAML
        meta_path = self.path + ".yaml"
        if exists(meta_path):
            with open(meta_path, 'r') as stream:
                self.attrs.update(yaml.safe_load(stream))
        else:
            raise FileNotFoundError(f"Metadata file not found for {self.path}")

    def __getattr__(self, name):
        if name in self.attrs:
            return self.attrs[name]
        raise AttributeError(f"no attribute '{name}'")

    def __call__(self):
        return self.data
    
    @property
    def data(self) -> np.ndarray:
        if self._array is None:
            self.read_csv()
        return self._array

    def read_csv(self):
        csv_path = self.path + ".csv"
        if not exists(csv_path):
            raise FileNotFoundError(f"No CSV file found for {self.path}")
        self._array = np.genfromtxt(csv_path, delimiter=",")


    def __repr__(self):
        shape = self.data.shape if self._array is not None else "not loaded"
        return f"<Measurement {self.name} | {self.attrs} | shape={shape}>"

import glob
def load_measurements(data_root: str = "../data") -> list[Measurement]:
    files = glob.glob(f"{data_root}/**/*", recursive=True)
    paths = [f.rsplit('.', 1)[0] for f in files]
    paths = sorted(set(paths))[1:]
    return [Measurement(p) for p in paths]


if __name__ == "__main__":
    data_root: str = "../data"
   
    m = load_measurements()
    m[1].data
    print(m[1])
