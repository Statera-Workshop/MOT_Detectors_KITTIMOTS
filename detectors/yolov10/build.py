"""
YOLOv10 builder (Image task only, anchor-free).

Paper : Wang et al., *YOLOv10: Real-Time End-to-End Object Detection*, arXiv:2405.14458
Code  : official Ultralytics YAML (cfg/yolov10n.yaml, copied unchanged) built by detectors/ultralytics_core
        with the official blocks, head and loss (dependency-closed copy of ultralytics/ultralytics).
Delta : SCDown, C2fCIB, PSA blocks; v10Detect dual head (one-to-many + one-to-one) trained with consistent dual assignments
Size  : n
Loss  : e2e_dual - [Unified] STAL (introduced with YOLO26) switched off - assignment as in the original release
Inference: the dense predictions of the inference head go through the common NMS ([Unified]).
"""

from pathlib import Path

from detectors.ultralytics_core.model import UltralyticsDetector

CFG = Path(__file__).resolve().parent / "cfg" / "yolov10n.yaml"
SCALE = "n"


def build_model(task, num_classes):
    if task != "image":
        raise ValueError("YOLOv10 is implemented for the Image task only (anchor-free, see WORK_CONDITIONS.md).")
    return UltralyticsDetector(CFG, num_classes, SCALE, loss="e2e_dual", stal=False)
