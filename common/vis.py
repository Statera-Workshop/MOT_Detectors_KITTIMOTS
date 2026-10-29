"""Common drawing and video I/O for every demo."""

from pathlib import Path

import cv2
import numpy as np

from common.kitti import config as cnf
from common.kitti import kitti_aug_utils as aug_utils
from common.kitti import kitti_bev_utils as bev_utils
from common.kitti import kitti_utils


def class_color(c):
    rng = np.random.RandomState(c * 7 + 3)
    return tuple(int(v) for v in rng.randint(60, 255, 3))


# ------------------------------------------------------------------ 2D image
def draw_2d(img_bgr, det, class_names):
    for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = class_color(int(label))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{class_names[int(label)]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_bgr, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img_bgr


# ------------------------------------------------------------------ BEV
def bev_to_image(bev):
    """[3, S, S] BEV tensor -> uint8 BGR image (legacy channel mapping)."""
    bev = bev.cpu().numpy() if hasattr(bev, "cpu") else bev
    s = bev.shape[1]
    img = np.zeros((s, s, 3))
    img[:, :, 2] = bev[0]
    img[:, :, 1] = bev[1]
    img[:, :, 0] = bev[2]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def gt_to_bev_px(gt_boxes, size):
    """Metric LiDAR boxes -> BEV pixel boxes (x, y, w, l, yaw) for drawing."""
    bc = cnf.boundary
    out = []
    for x, y, z, h, w, l, yaw in gt_boxes:
        col = (y - bc["minY"]) / (bc["maxY"] - bc["minY"]) * size
        row = (x - bc["minX"]) / (bc["maxX"] - bc["minX"]) * size
        wp = w / (bc["maxY"] - bc["minY"]) * size
        lp = l / (bc["maxX"] - bc["minX"]) * size
        out.append((col, row, wp, lp, -yaw))
    return out


def draw_bev(bev_img, det, gt=None):
    """Draw predictions (class colors) and optional GT (white) on the BEV map."""
    size = bev_img.shape[0]
    if gt is not None:
        for (x, y, w, l, yaw) in gt_to_bev_px(gt["boxes"], size):
            bev_utils.drawRotatedBox(bev_img, x, y, w, l, yaw, (255, 255, 255), 1)
    for (x, y, w, l, im, re), label in zip(det["bev_px"], det["labels"]):
        bev_utils.drawRotatedBox(bev_img, x, y, w, l, np.arctan2(im, re), cnf.colors[int(label)])
    # legacy demo orientation: ego vehicle at the bottom, driving upwards
    return cv2.rotate(bev_img, cv2.ROTATE_180)


# ------------------------------------------------------------------ camera 3D
def draw_camera_3d(img_bgr, boxes_lidar, labels, calib, color=None, thickness=2):
    """Project metric LiDAR 3D boxes into the left camera image (legacy path)."""
    if len(boxes_lidar) == 0:
        return img_bgr
    cam = aug_utils.lidar_to_camera_box(np.asarray(boxes_lidar, dtype=np.float64), calib.V2C, calib.R0, calib.P)
    for (x, y, z, h, w, l, ry), label in zip(cam, labels):
        obj = kitti_utils.Object3d("Car -1 -1 0 0 0 0 0 0 0 0 0 0 0 0")
        obj.t = (x, y, z)
        obj.h, obj.w, obj.l = h, w, l
        obj.ry = np.arctan2(np.sin(ry), np.cos(ry))
        pts2d, _ = kitti_utils.compute_box_3d(obj, calib.P)
        if pts2d is not None:
            c = color if color is not None else tuple(cnf.colors[int(label)])
            kitti_utils.draw_projected_box3d(img_bgr, pts2d, c, thickness)
    return img_bgr


# ------------------------------------------------------------------ video
class VideoWriter:
    def __init__(self, path, fps, size):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {path}")

    def write(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self.writer.write(frame)

    def close(self):
        self.writer.release()
