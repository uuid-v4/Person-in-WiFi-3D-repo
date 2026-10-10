import os

import numpy as np
import torch
from mmdet.datasets.pipelines import Compose

from .builder import DATASETS
from .wifi_pose import WifiPoseDataset


@DATASETS.register_module()
class WifiPoseCachedDataset(WifiPoseDataset):
    """WifiPoseDataset backed by features precomputed with csi_demo/train/cache_wiception.py.

    Produces the same sample dict as WifiPoseDataset.get_item_single_frame, so training and
    evaluate() are unchanged, without re-running the CSI phase sanitization every epoch.
    """

    def __init__(self, cache_dir, pipeline, **kwargs):
        self.data_root = cache_dir
        self.csi = np.load(os.path.join(cache_dir, 'csi.npy'))
        self.keypoints = np.load(os.path.join(cache_dir, 'keypoints.npy'))
        self.num_person = np.load(os.path.join(cache_dir, 'num_person.npy'))
        self.filename_list = np.load(os.path.join(cache_dir, 'names.npy')).tolist()
        self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def get_item_single_frame(self, index):
        n = int(self.num_person[index])
        keypoint = torch.from_numpy(np.array(self.keypoints[index, :n], dtype=np.float32))
        return dict(
            img=torch.from_numpy(np.array(self.csi[index], dtype=np.float32)),
            gt_keypoints=keypoint,
            gt_labels=np.zeros(n, dtype=np.int64),
            gt_bboxes=torch.tensor([]),
            gt_areas=torch.tensor([]),
            img_name=self.filename_list[index])
