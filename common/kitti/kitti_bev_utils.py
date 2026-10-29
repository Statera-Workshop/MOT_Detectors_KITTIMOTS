"""
KITTI LiDAR -> Bird's-Eye-View utilities shared by Complex-YOLO and YOLO3D.

Origin
    Complex_Yolo_Yolov3/data_process/kitti_bev_utils.py   (BEV map, drawing)
    Yolo_3d_Yolov3/data_process/kitti_bev_utils.py        (3D target builder)

The BEV map and the geometry are identical in both legacy projects, so they
live here once. Only the target encoders differ (task-specific):

    build_complex_target  -> [cls, x_col, y_row, w, l, im, re]
    build_yolo3d_target   -> [cls, x_col, y_row, z, h, w, l, im, re]

All values are normalized to [0, 1] exactly as in the legacy code.
"""

import math

import cv2
import numpy as np

from common.kitti import config as cnf


def removePoints(PointCloud, BoundaryCond):
    minX = BoundaryCond["minX"]
    maxX = BoundaryCond["maxX"]
    minY = BoundaryCond["minY"]
    maxY = BoundaryCond["maxY"]
    minZ = BoundaryCond["minZ"]
    maxZ = BoundaryCond["maxZ"]

    mask = np.where(
        (PointCloud[:, 0] >= minX) & (PointCloud[:, 0] <= maxX)
        & (PointCloud[:, 1] >= minY) & (PointCloud[:, 1] <= maxY)
        & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2] <= maxZ)
    )
    PointCloud = PointCloud[mask]
    PointCloud[:, 2] = PointCloud[:, 2] - minZ
    return PointCloud


def makeBVFeature(PointCloud_, Discretization, bc):
    """3-channel BEV map: [intensity, height, density] (legacy order)."""
    Height = cnf.BEV_HEIGHT + 1
    Width = cnf.BEV_WIDTH + 1

    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / Discretization))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)

    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    heightMap = np.zeros((Height, Width))
    _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]
    max_height = float(np.abs(bc["maxZ"] - bc["minZ"]))
    heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = (
        PointCloud_frac[:, 2] / max_height
    )

    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))
    _, indices, counts = np.unique(
        PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True
    )
    PointCloud_top = PointCloud[indices]
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    RGB_Map[2, :, :] = densityMap[: cnf.BEV_HEIGHT, : cnf.BEV_WIDTH]    # r_map
    RGB_Map[1, :, :] = heightMap[: cnf.BEV_HEIGHT, : cnf.BEV_WIDTH]     # g_map
    RGB_Map[0, :, :] = intensityMap[: cnf.BEV_HEIGHT, : cnf.BEV_WIDTH]  # b_map
    return RGB_Map


def read_labels_for_bevbox(objects):
    """[cls, x, y, z, h, w, l, ry] in rect-camera coordinates."""
    bbox_selected = []
    for obj in objects:
        if obj.cls_id != -1:
            bbox_selected.append(
                [obj.cls_id, obj.t[0], obj.t[1], obj.t[2], obj.h, obj.w, obj.l, obj.ry]
            )
    if len(bbox_selected) == 0:
        return np.zeros((1, 8), dtype=np.float32), True
    return np.array(bbox_selected).astype(np.float32), False


def get_corners(x, y, w, l, yaw):
    """BEV image-coordinate corners (legacy definition)."""
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    bev_corners[0, 0] = x - w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)  # front left
    bev_corners[0, 1] = y - w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)
    bev_corners[1, 0] = x - w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)  # rear left
    bev_corners[1, 1] = y - w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)
    bev_corners[2, 0] = x + w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)  # rear right
    bev_corners[2, 1] = y + w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)
    bev_corners[3, 0] = x + w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)  # front right
    bev_corners[3, 1] = y + w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)
    return bev_corners


def _in_bev(x, y, bc):
    return (bc["minX"] < x < bc["maxX"]) and (bc["minY"] < y < bc["maxY"])


def build_complex_target(labels):
    """Complex-YOLO target (Complex_Yolo_Yolov3 legacy, unchanged math)."""
    bc = cnf.boundary
    target = []
    for i in range(labels.shape[0]):
        cl, x, y, z, h, w, l, yaw = labels[i]
        l = l + cnf.BOX_WL_MARGIN
        w = w + cnf.BOX_WL_MARGIN
        yaw = np.pi * 2 - yaw
        if _in_bev(x, y, bc):
            y1 = (y - bc["minY"]) / (bc["maxY"] - bc["minY"])
            x1 = (x - bc["minX"]) / (bc["maxX"] - bc["minX"])
            w1 = w / (bc["maxY"] - bc["minY"])
            l1 = l / (bc["maxX"] - bc["minX"])
            target.append([cl, y1, x1, w1, l1, math.sin(float(yaw)), math.cos(float(yaw))])
    return np.array(target, dtype=np.float32).reshape(-1, 7)


def build_yolo3d_target(labels):
    """YOLO3D target (Yolo_3d_Yolov3 legacy, unchanged math)."""
    bc = cnf.boundary
    target = []
    for i in range(labels.shape[0]):
        cl, x, y, z, h, w, l, yaw = labels[i]
        l = l + cnf.BOX_WL_MARGIN
        w = w + cnf.BOX_WL_MARGIN
        yaw = -yaw
        if _in_bev(x, y, bc):
            y1 = (y - bc["minY"]) / (bc["maxY"] - bc["minY"])
            x1 = (x - bc["minX"]) / (bc["maxX"] - bc["minX"])
            z1 = (z - bc["minZ"]) / (bc["maxZ"] - bc["minZ"])
            h1 = h / (bc["maxZ"] - bc["minZ"])
            w1 = w / (bc["maxY"] - bc["minY"])
            l1 = l / (bc["maxX"] - bc["minX"])
            target.append([cl, y1, x1, z1, h1, w1, l1, math.sin(float(yaw)), math.cos(float(yaw))])
    return np.array(target, dtype=np.float32).reshape(-1, 9)


def drawRotatedBox(img, x, y, w, l, yaw, color, thickness=2):
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(img, [corners_int], True, color, thickness)
    c = bev_corners.reshape(-1, 2).astype(np.int32)
    # heading edge (front) in yellow, as in the legacy demo
    cv2.line(img, (int(c[0, 0]), int(c[0, 1])), (int(c[3, 0]), int(c[3, 1])), (255, 255, 0), thickness)
