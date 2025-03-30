"""
This makes the lab-notebook located at "../notebook" computable by extracting referenced notebooks.
This assumes, that the post frontmatter contains a section with a list called measurements, and each measurement has one define path.
"""
#%%
import frontmatter
from glob import glob
from pathlib import Path

class Post(dict):
    def __init__(self, path: str):
        super().__init__()
        self.path = Path(path)
        metadata = frontmatter.load(self.path).metadata
        self.update(metadata)

    @property
    def content(self)->str: return frontmatter.load(self.path).content

def posts(path:str = "../notebook") -> list[Post]:
    paths = glob(path+"/**.md", recursive=True)
    files = [Post(p) for p in paths]
    return files

# get measurements from one entry
def get_measurements(post: Post) -> list[dict]:
    measurements = post.get("measurements", [])
    for m in measurements:
        m.update({"post": post})
    return measurements

# get measurements from all entries
def measurements(p: list[Post]|None = None) -> list[dict]:
    if p is None:
        p = posts()
    return [m for i in p for m in get_measurements(i)]

if __name__ == "__main__":
    measurements = measurements()
    measurements = [m for m in measurements if 1==m["sample"]]
    post = measurements[0]["post"]
    print(len(measurements))
# %%
