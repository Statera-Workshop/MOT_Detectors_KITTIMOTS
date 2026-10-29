"""
YOLOv3 builders for the three tasks.

Everything here is algorithm-specific:
    - which .cfg (backbone + neck + head layout)
    - how [yolo] anchors are parsed
    - which detection layer (loss / target assignment) is used
    - how many box attributes the head regresses

Everything else (Darknet parsing, training, evaluation, demo) is common.
"""

import math
from pathlib import Path

from common.darknet import Darknet
from detectors.yolov3.complex_yolo_layer import YoloLayer as ComplexYoloLayer
from detectors.yolov3.image_yolo_layer import YoloLayer as ImageYoloLayer
from detectors.yolov3.yolo3d_yolo_layer import YoloLayer as Yolo3dYoloLayer

CFG_DIR = Path(__file__).resolve().parent / "cfg"


def _floats(text):
    return [float(v) for v in text.split(",")]


def _mask(block):
    return [int(v) for v in block["mask"].split(",")]


def image_factory(block, hyperparams):
    """Image YOLOv3: (w, h) anchors, global 9-anchor matching (legacy)."""
    v = _floats(block["anchors"])
    all_anchors = [(v[i], v[i + 1]) for i in range(0, len(v), 2)]
    mask = _mask(block)
    return ImageYoloLayer(
        [all_anchors[i] for i in mask],
        int(block["classes"]),
        int(hyperparams["height"]),
        all_anchors=all_anchors,
        anchor_mask=mask,
    )


def complex_factory(block, hyperparams):
    """Complex-YOLO: (w, l, yaw) anchors -> (w, l, sin, cos) (legacy)."""
    v = _floats(block["anchors"])
    anchors = [(v[i], v[i + 1], math.sin(v[i + 2]), math.cos(v[i + 2])) for i in range(0, len(v), 3)]
    return ComplexYoloLayer([anchors[i] for i in _mask(block)], int(block["classes"]),
                            int(hyperparams["height"]))


def yolo3d_factory(block, hyperparams):
    """YOLO3D: (h, w, l, yaw) anchors -> (h, w, l, sin, cos) (legacy)."""
    v = _floats(block["anchors"])
    anchors = [(v[i], v[i + 1], v[i + 2], math.sin(v[i + 3]), math.cos(v[i + 3])) for i in range(0, len(v), 4)]
    return Yolo3dYoloLayer(anchors=[anchors[i] for i in _mask(block)], num_classes=int(block["classes"]),
                           stride=1, ignore_thresh=0.5)


SPECS = {
    #          cfg file                 factory          box attributes regressed
    "image":   ("image_yolov3.cfg",   image_factory,   5),   # x y w h conf
    "complex": ("complex_yolov3.cfg", complex_factory, 7),   # x y w l im re conf
    "yolo3d":  ("yolo3d_yolov3.cfg",  yolo3d_factory,  8),   # x y z h w l yaw conf  (legacy: 9)
}


def build_model(task, num_classes):
    cfg_name, factory, num_box_attr = SPECS[task]
    return Darknet(CFG_DIR / cfg_name, factory, num_classes, num_box_attr)
