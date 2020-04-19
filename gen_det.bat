#!/usr/bin/env bash
cd src
python gen_dets.py det --load_model ../models/all_dla34.pth --data_cfg lib/cfg/data_det.json
cd ../