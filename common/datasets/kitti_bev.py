"""
KITTI_5k LiDAR BEV dataset, shared by Complex-YOLO and YOLO3D.

Merged from the legacy KittiDataset + KittiYOLODataset of
Complex_Yolo_Yolov3 and Yolo_3d_Yolov3 (identical except for the target
encoder). Splits are read from datasets/KITTI_5k/ImageSets/{split}.txt.

Output contract
    bev     : FloatTensor [3, 608, 608]
    targets : FloatTensor [N, 8]   complex: [b, cls, x, y, w, l, im, re]
              FloatTensor [N, 10]  yolo3d : [b, cls, x, y, z, h, w, l, im, re]

Ground truth for evaluation (metric LiDAR frame, same for both tasks)
    boxes   : [M, 7] = [x, y, z, h, w, l, yaw]  (z = box bottom)
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from common.kitti import config as cnf
from common.kitti import kitti_aug_utils as aug_utils
from common.kitti import kitti_bev_utils as bev_utils
from common.kitti import kitti_utils

TARGET_BUILDERS = {
    "complex": (bev_utils.build_complex_target, 8, 6),   # builder, columns, im column
    "yolo3d": (bev_utils.build_yolo3d_target, 10, 8),
}


class KittiBEVDataset(Dataset):
    def __init__(self, root, split, task, train=False, hflip_prob=0.0, max_samples=None):
        self.root = Path(root)
        self.task = task
        self.train = train
        self.hflip_prob = hflip_prob if train else 0.0
        self.build_target, self.ncols, self.im_col = TARGET_BUILDERS[task]
        split_file = self.root / "ImageSets" / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(split_file)
        ids = [x.strip() for x in split_file.read_text().split() if x.strip()]
        if max_samples:
            ids = ids[:max_samples]
        d = self.root / "training"
        self.lidar_dir, self.image_dir = d / "velodyne", d / "image_2"
        self.calib_dir, self.label_dir = d / "calib", d / "label_2"
        # Legacy: training only uses frames that contain at least one valid object.
        self.ids = [i for i in ids if self._has_valid_object(i)] if train else ids

    # ---------------------------------------------------------------- raw I/O
    def get_lidar(self, sid):
        return np.fromfile(self.lidar_dir / f"{sid}.bin", dtype=np.float32).reshape(-1, 4)

    def get_calib(self, sid):
        return kitti_utils.Calibration(str(self.calib_dir / f"{sid}.txt"))

    def get_label(self, sid):
        return kitti_utils.read_label(str(self.label_dir / f"{sid}.txt"))

    def image_path(self, sid):
        return self.image_dir / f"{sid}.png"

    def lidar_boxes(self, sid, calib=None):
        """[N, 8] = [cls, x, y, z, h, w, l, yaw] in the LiDAR frame."""
        calib = calib or self.get_calib(sid)
        labels, empty = bev_utils.read_labels_for_bevbox(self.get_label(sid))
        if empty:
            return np.zeros((0, 8), dtype=np.float32)
        labels[:, 1:] = aug_utils.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P)
        return labels

    def _has_valid_object(self, sid):
        bc = cnf.boundary
        for row in self.lidar_boxes(sid):
            x, y, z = row[1:4]
            if (bc["minX"] <= x <= bc["maxX"] and bc["minY"] <= y <= bc["maxY"]
                    and bc["minZ"] <= z <= bc["maxZ"]):
                return True
        return False

    def make_bev(self, lidar):
        b = bev_utils.removePoints(lidar, cnf.boundary)
        return bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)

    # ---------------------------------------------------------------- dataset
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]
        lidar = self.get_lidar(sid)
        labels = self.lidar_boxes(sid)
        if self.train and len(labels):
            lidar, labels[:, 1:] = aug_utils.complex_yolo_pc_augmentation(lidar, labels[:, 1:], True)
        bev = torch.from_numpy(self.make_bev(lidar)).float()
        t = self.build_target(labels) if len(labels) else np.zeros((0, self.ncols - 1), np.float32)
        targets = torch.zeros((len(t), self.ncols))
        if len(t):
            targets[:, 1:] = torch.from_numpy(t)
        if self.hflip_prob and np.random.random() < self.hflip_prob:
            bev = torch.flip(bev, [-1])
            targets[:, 2] = 1 - targets[:, 2]                     # x (column)
            targets[:, self.im_col] = -targets[:, self.im_col]    # yaw imaginary part
        return sid, bev, targets

    def collate_fn(self, batch):
        sids, bevs, targets = zip(*batch)
        out = []
        for b, t in enumerate(targets):
            if t.numel():
                t = t.clone()
                t[:, 0] = b
                out.append(t)
        targets = torch.cat(out, 0) if out else torch.zeros((0, self.ncols))
        return list(sids), torch.stack(bevs, 0), targets

    # ---------------------------------------------------------------- eval GT
    def ground_truth(self, sid):
        bc = cnf.boundary
        rows = [r for r in self.lidar_boxes(sid)
                if bc["minX"] < r[1] < bc["maxX"] and bc["minY"] < r[2] < bc["maxY"]]
        rows = np.array(rows, dtype=np.float32).reshape(-1, 8)
        return {
            "boxes": rows[:, 1:8].astype(np.float64),
            "labels": rows[:, 0].astype(np.int64),
            "ignore": np.zeros(len(rows), dtype=bool),
        }
