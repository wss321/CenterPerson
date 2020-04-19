import os
from os.path import join

root = "E:/PyProjects/datasets/PETS2009"
seq = os.listdir(join(root, "images/test"))
with open("../data/pets09.val", "w") as f:
    for i in seq:
        imgs = os.listdir(join(root, f"images/test/{i}/img1"))
        for img in imgs:
            f.write(f"PETS2009/images/test/{i}/img1/{img}\n")
