"""
YOLOv9 builder (Image task only, anchor-free).

Paper : Wang, Yeh & Liao, *YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information*, arXiv:2402.13616
Code  : official Ultralytics YAML (cfg/yolov9t.yaml, copied unchanged) built by detectors/ultralytics_core
        with the official blocks, head and loss (dependency-closed copy of ultralytics/ultralytics).
Delta : GELAN backbone / neck (RepNCSPELAN4, ADown, SPPELAN); head and loss as YOLOv8
Size  : t
Loss  : v8 - [Unified] STAL (introduced with YOLO26) switched off - assignment as in the original release
Inference: the dense predictions of the inference head go through the common NMS ([Unified]).
"""

from pathlib import Path

from detectors.ultralytics_core.model import UltralyticsDetector

CFG = Path(__file__).resolve().parent / "cfg" / "yolov9t.yaml"
SCALE = "t"


def build_model(task, num_classes):
    if task != "image":
        raise ValueError("YOLOv9 is implemented for the Image task only (anchor-free, see WORK_CONDITIONS.md).")
    return UltralyticsDetector(CFG, num_classes, SCALE, loss="v8", stal=False)
