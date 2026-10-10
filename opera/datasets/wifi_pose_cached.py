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

    def __init__(self, cache_dir, pipeline, use_vis=False, link_dropout=0.0, noise_std=0.0, **kwargs):
        """use_vis: append keypoints_vis.npy as an extra channel (joints with vis <= 0.5 get no loss).
        link_dropout / noise_std: training-only augmentation on (rx, tx, packets, features) windows;
        each link is zeroed with probability link_dropout, Gaussian noise is added to all values."""
        self.data_root = cache_dir
        self.csi = np.load(os.path.join(cache_dir, 'csi.npy'))
        self.keypoints = np.load(os.path.join(cache_dir, 'keypoints.npy'))
        self.vis = np.load(os.path.join(cache_dir, 'keypoints_vis.npy')) if use_vis else None
        self.link_dropout, self.noise_std = link_dropout, noise_std
        self.num_person = np.load(os.path.join(cache_dir, 'num_person.npy'))
        self.filename_list = np.load(os.path.join(cache_dir, 'names.npy')).tolist()
        self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def get_item_single_frame(self, index):
        n = int(self.num_person[index])
        kp = np.array(self.keypoints[index, :n], dtype=np.float32)
        if self.vis is not None:
            kp = np.concatenate([kp, self.vis[index, :n, :, None].astype(np.float32)], axis=-1)
        keypoint = torch.from_numpy(kp)
        csi = np.array(self.csi[index], dtype=np.float32)
        if self.link_dropout > 0 and csi.ndim == 4:
            keep = np.random.rand(csi.shape[0], csi.shape[1]) >= self.link_dropout
            csi *= keep[:, :, None, None]
        if self.noise_std > 0:
            csi += np.random.normal(0, self.noise_std, csi.shape).astype(np.float32)
        return dict(
            img=torch.from_numpy(csi),
            gt_keypoints=keypoint,
            gt_labels=np.zeros(n, dtype=np.int64),
            gt_bboxes=torch.tensor([]),
            gt_areas=torch.tensor([]),
            img_name=self.filename_list[index])
