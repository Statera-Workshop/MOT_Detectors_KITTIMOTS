"""
YOLOv13 builder (Image task only, anchor-free).

Paper : Lei et al., *YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual
        Perception*, arXiv:2506.17733
Code  : official iMoonLab/yolov13 YAML (cfg/yolov13.yaml, unchanged) and blocks (modules.py, verbatim),
        built with the iMoonLab parse rules (detectors/ultralytics_core/model.py, extra=modules)
Delta : HyperACE (hypergraph-based adaptive correlation enhancement), FullPAD tunnels (full-pipeline
        aggregation-and-distribution), DSC3k2 / DSConv depthwise-separable blocks, A2C2f area attention;
        Detect head and v8 loss as YOLO11 / YOLO12 (the official code is an Ultralytics 8.3 fork)
Size  : n
Loss  : v8 (TAL + BCE + CIoU + DFL); [Unified] STAL off (introduced later with YOLO26)
"""

from pathlib import Path

from detectors.ultralytics_core.model import UltralyticsDetector
from detectors.yolov13 import modules as V13

CFG = Path(__file__).resolve().parent / "cfg" / "yolov13.yaml"
SCALE = "n"


def build_model(task, num_classes):
    if task != "image":
        raise ValueError("YOLOv13 is implemented for the Image task only (anchor-free, see WORK_CONDITIONS.md).")
    return UltralyticsDetector(CFG, num_classes, SCALE, loss="v8", stal=False, extra=V13)
