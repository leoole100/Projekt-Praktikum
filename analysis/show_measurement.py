# %%
import notebook
from measurement import CSVMeasurement


#%%
meas = [CSVMeasurement(m) for m in notebook.measurements()]
meas[0]
# %%
