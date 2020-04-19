from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import json
import os
import cv2
from tqdm import tqdm
import numpy as np

from lib.opts import opts

from lib.detectors.detector_factory import detector_factory


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


@torch.no_grad()
def gen_det(opt):
    data_cfg = opt.data_cfg
    f = open(data_cfg)
    data_cfg_dict = json.load(f)
    f.close()
    dataset_root = data_cfg_dict['root']
    seqs_dir = os.path.join(dataset_root, data_cfg_dict["seqs_dir"])
    seqs = os.listdir(seqs_dir)

    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    for seqs_id, seq in enumerate(seqs):
        # if "MOT16-06" not in seq:
        #     continue
        print(seq)
        seq_root = os.path.join(seqs_dir, seq, "img1")
        files = [os.path.join(seq_root, fn) for fn in os.listdir(seq_root)]
        det_file = os.path.join(seq_root.replace("img1", "det"), "ct_det.txt")
        with open(det_file, "w+") as f1:
            files = tqdm(files)
            files.set_description(f"{seq}:")
            for path in files:
                ret = detector.run(path)
                dets = np.array(ret['results'][1])
                split = os.path.split(path)

                frame_id = int(split[1].strip(".jpg"))
                img1 = cv2.imread(path)
                remain_inds = dets[:, 4] > 0.2
                dets = dets[remain_inds]
                xywh = xyxy2xywh(dets[:, :4])
                xywh = np.asarray(xywh, dtype=np.int)

                for t in range(len(dets)):
                    x1 = int(dets[t, 0])
                    y1 = int(dets[t, 1])
                    x2 = int(dets[t, 2])
                    y2 = int(dets[t, 3])
                    f1.write(f"{frame_id}, {xywh[t, 0]}, {xywh[t, 1]}, {xywh[t, 2]}, {xywh[t, 3]}, {dets[t, 4]}\n")
                    cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.imshow("", img1)
                cv2.waitKey(100)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    gen_det(opt)
