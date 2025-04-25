#%%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")
import yaml
from glob import glob

# %%
# import the data
p = glob("../measurement/2025-04-25/004 *nm.yaml")
data = []
for file in p:
	with open(file, 'r') as f:
		data.append(yaml.safe_load(f))
data = sorted(data, key=lambda x: x["metadata"]["monochromator"]["center"])

#%%
for d in data:
	s = d["spectrum"]
	plt.plot(
		s["wavelength"], s["counts"], 
		label=d["metadata"]["monochromator"]["center"],
		alpha=0.7
	)
plt.legend()
plt.xlabel(r"$\lambda$ (nm)")
plt.ylabel("Counts")
plt.savefig("figures/stich.pdf")
plt.show()
