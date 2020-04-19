#!/usr/bin/env bash
cd src
python train.py det --exp_id det_dla34 --gpus 0 --num_worker 1 --batch_size 1 --lr 0.0001 --load_model ../models/ctdet_coco_dla_2x.pth
::python train.py mot --exp_id mot_dla34_2 --gpus 0  --num_worker 0 --batch_size 8 --lr 0.0001 --load_model ../models/ctdet_coco_dla_2x.pth
cd ..