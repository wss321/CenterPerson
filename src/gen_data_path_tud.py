import os
from os.path import join

root = "E:/PyProjects/datasets/TUD"
seq = os.listdir(join(root, "images/test"))
with open("./data/tud.test", "w") as f:
    for i in seq:
        imgs = os.listdir(join(root, f"images/test/{i}/img1"))
        for img in imgs:
            f.write(f"TUD/images/test/{i}/img1/{img}\n")
