from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import argparse
import torch
import json
import time
import os
import cv2
import os.path as osp
import numpy as np
from torchvision.transforms import transforms as T
from lib.models.model import create_model, load_model
from lib.utils.utils import ct_xywh2xyxy, xyxy2ct_xywh
from lib.datasets.dataset.jde import LoadImagesAndLabels
from lib.opts import opts
from lib.models.decode import mot_decode
from lib.utils.post_process import ctdet_post_process
from copy import copy
from tqdm import tqdm
from collections import OrderedDict


# from tensorboardX import SummaryWriter


def post_process(opt, dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], opt.num_classes)
    for j in range(1, opt.num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    return dets[0]


def merge_outputs(opt, detections):
    results = {}
    for j in range(1, opt.num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, opt.num_classes + 1)])
    if len(scores) > 128:
        kth = len(scores) - 128
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, opt.num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results


class DetDataset(LoadImagesAndLabels):  # for training
    def __init__(self, root, paths, img_size=(1088, 608), augment=False, transforms=None):
        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                try:
                    lb = np.loadtxt(lp)
                except:
                    continue
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]
        imgs, labels, img_path, (h, w) = self.get_data(img_path, label_path)
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        return imgs, img_path, [(h, w)]


@torch.no_grad()
def gen_det(opt,
            batch_size=12,
            img_size=(1088, 608)):
    data_cfg = opt.data_cfg
    f = open(data_cfg)
    data_cfg_dict = json.load(f)
    f.close()
    test_path = data_cfg_dict['test']
    dataset_root = data_cfg_dict['root']
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    # model = torch.nn.DataParallel(model)
    model = model.to(opt.device)
    model.eval()
    # dummy_input = torch.rand(1, 3, 1088, 608).cuda()  # 假设输入13张1*28*28的图片
    # with SummaryWriter(comment='model') as w:
    #     w.add_graph(model, dummy_input)
    # Get dataloader
    transforms = T.Compose([T.ToTensor()])
    dataset = DetDataset(dataset_root, test_path, img_size, augment=False, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=0, drop_last=False)
    seen = 0
    dataloader = tqdm(dataloader)
    for batch_i, (imgs, paths, shapes) in enumerate(dataloader):
        seen += batch_size
        if seen < 3148:
            continue
        path = paths[0]
        split = path.split("/")
        split[0] += "/"
        if "MOT16-03" in path:
            continue
        if "MOT16-01" in path:
            continue
        # if int(split[-1].strip(".jpg")) < 736:
        #     continue
        output = model(imgs.cuda())[-1]
        origin_shape = shapes[0]
        width = origin_shape[1]
        height = origin_shape[0]
        inp_height = img_size[1]
        inp_width = img_size[0]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // opt.down_ratio,
                'out_width': inp_width // opt.down_ratio}
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg'] if opt.reg_offset else None
        opt.K = 200
        detections, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        # Compute average precision for each sample
        for si, _ in enumerate(imgs):
            seen += 1
            # path = paths[si]
            # img0 = cv2.imread(path)

            dets = detections[si]
            dets = dets.unsqueeze(0)
            dets = post_process(opt, dets, meta)
            dets = merge_outputs(opt, [dets])[1]
            if dets is None:
                continue
            path = paths[si]
            split = path.split("/")
            split[0] += "/"
            det_file = os.path.join(*split[:-2], "det", "FairMOT_det.txt")
            with open(det_file, "a+") as f:
                frame_id = int(split[-1].strip(".jpg"))
                img1 = cv2.imread(path)
                remain_inds = dets[:, 4] > 0.4
                dets = dets[remain_inds]
                xywh = xyxy2ct_xywh(dets[:, :4])

                for t in range(len(dets)):
                    x1 = dets[t, 0]
                    y1 = dets[t, 1]
                    x2 = dets[t, 2]
                    y2 = dets[t, 3]
                    f.write(
                        "%d,-1, %.2f, %.2f, %.2f, %.2f, %.2f, -1,-1,-1\n" % (
                            frame_id, xywh[t, 0], xywh[t, 1], xywh[t, 2], xywh[t, 3], dets[t, 4]))
                    cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.imshow("", img1)
                cv2.waitKey(100)
                # cv2.imwrite('pred.jpg', img1)

    return None


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    # opt.task = "mot"
    # opt.load_model = "../models/all_dla34.pth"
    # opt.data_cfg = "lib/cfg/data_det.json"
    with torch.no_grad():
        gen_det(opt, batch_size=1)
# python gen_dets.py mot --load_model ../models/all_dla34.pth --data_cfg lib/cfg/data_det.json
