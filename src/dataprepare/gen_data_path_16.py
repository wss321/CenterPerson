import os
from os.path import join

root = "E:/PyProjects/datasets/MOT16"
seq = os.listdir(join(root, "images/train"))
with open("../data/mot16.train", "w") as f:
    for i in seq:
        imgs = os.listdir(join(root, f"images/train/{i}/img1"))
        for img in imgs:
            f.write(f"MOT16/images/train/{i}/img1/{img}\n")

root = "E:/PyProjects/datasets/MOT16"
seq = os.listdir(join(root, "images/test"))
with open("../data/mot16.test", "w") as f:
    for i in seq:
        imgs = os.listdir(join(root, f"images/test/{i}/img1"))
        for img in imgs:
            f.write(f"MOT16/images/test/{i}/img1/{img}\n")
