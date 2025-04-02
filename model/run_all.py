#%%
import os
import subprocess
import sys

folder = "."  # or "your_folder"
exclude = "run_all.py"

for filename in sorted(os.listdir(folder)):
    if filename.endswith(".py") and filename != exclude:
        path = os.path.join(folder, filename)
        print(f"Running {filename}")
        subprocess.run([sys.executable, path])

