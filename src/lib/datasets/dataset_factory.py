from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.jde import JointDataset, CtDetDataset


def get_dataset(task):
    if task == 'mot':
        return JointDataset
    if task == 'det':
        return CtDetDataset
    else:
        return None
