"""
YOLO12 builder (Image task only, anchor-free).

Paper : Tian, Ye & Doermann, *YOLOv12: Attention-Centric Real-Time Object Detectors*, arXiv:2502.12524
Code  : official Ultralytics YAML (cfg/yolo12.yaml, copied unchanged) built by detectors/ultralytics_core
        with the official blocks, head and loss (dependency-closed copy of ultralytics/ultralytics).
Delta : A2C2f (R-ELAN with area attention) backbone / neck; head and loss as YOLO11 / v8
Size  : n
Loss  : v8 - [Unified] STAL (introduced with YOLO26) switched off - assignment as in the original release
Inference: the dense predictions of the inference head go through the common NMS ([Unified]).
"""

from pathlib import Path

from detectors.ultralytics_core.model import UltralyticsDetector

CFG = Path(__file__).resolve().parent / "cfg" / "yolo12.yaml"
SCALE = "n"


def build_model(task, num_classes):
    if task != "image":
        raise ValueError("YOLO12 is implemented for the Image task only (anchor-free, see WORK_CONDITIONS.md).")
    return UltralyticsDetector(CFG, num_classes, SCALE, loss="v8", stal=False)
