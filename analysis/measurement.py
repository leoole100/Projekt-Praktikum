"""
let's the user lazy load the measurement data
"""
# %%
import numpy as np
from notebook import Post
from abc import ABC, abstractmethod

class Measurement(dict, ABC):
    def __init__(self, metadata):
        super().__init__()
        self.update(metadata)
        self.post: Post = metadata["post"]
        self._path = self.post.path.parent / metadata["path"]
        self.name: str = str(self._path).rsplit("measurement/")[1].rsplit(".", 1)[0]
        self._data = None
    
    def __call__(self): return self.data
    
    def __repr__(self):
        status = "loaded" if self._data is not None else "not loaded"
        metadata = {k:v for (k,v) in self.items() if k not in ["path"]}
        return f"<Measurement {self.name} | {status} | "+ str(metadata) + " >"

    @property
    def data(self):
        if self._data is None:
            self._data = self.load()
        return self._data
    
    @abstractmethod
    def load(self) -> np.ndarray: pass


class CSVMeasurement(Measurement):
    def load(self):
        return np.loadtxt(self._path, delimiter=",")

if __name__ == "__main__":
    import notebook
    measurements = notebook.measurements()
    measurements = [m for m in measurements if ".csv" in m["path"].lower()]
    measurements = [CSVMeasurement(m) for m in measurements]
    m = measurements[0]
    assert type(m()) is np.ndarray
    assert type(m.data) is np.ndarray
    print(m)
# %%
