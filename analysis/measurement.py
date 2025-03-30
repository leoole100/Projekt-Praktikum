"""
let's the user lazy load the measurement data
"""
# %%
import numpy as np

class Measurement(dict):
    def __init__(self, metadata: dict):
        super().__init__()
        self.update(metadata)
        self._path = metadata["path"]
        self._data = None
    
    def __call__(self): return self.data

    @property
    def data(self)->np.ndarray:
        if self._data is None:
            self._data = self.load()
        return self._data

    def load(self)->np.ndarray: return np.loadtxt(self._path, delimiter=",")

if __name__ == "__main__":
    import notebook
    m = Measurement(notebook.measurements()[0])
    assert type(m()) is np.ndarray
    assert type(m.data) is np.ndarray
# %%
