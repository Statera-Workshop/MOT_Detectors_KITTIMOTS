"""
YOLO26 builder (Image task only, anchor-free).

Paper : Jocher et al., *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models*, arXiv:2606.03748
Code  : official Ultralytics YAML (cfg/yolo26.yaml, copied unchanged) built by detectors/ultralytics_core
        with the official blocks, head and loss (dependency-closed copy of ultralytics/ultralytics).
Delta : end-to-end dual head (NMS-free one-to-one path), DFL removed (reg_max 1), ProgLoss + STAL (E2ELoss)
Size  : n
Loss  : e2e_prog - STAL on (official YOLO26 label assignment)
    MuSGD (YOLO26 training optimiser) is replaced by the common Adam protocol.
Inference: the dense predictions of the inference head go through the common NMS ([Unified]).
"""

from pathlib import Path

from detectors.ultralytics_core.model import UltralyticsDetector

CFG = Path(__file__).resolve().parent / "cfg" / "yolo26.yaml"
SCALE = "n"


def build_model(task, num_classes):
    if task != "image":
        raise ValueError("YOLO26 is implemented for the Image task only (anchor-free, see WORK_CONDITIONS.md).")
    return UltralyticsDetector(CFG, num_classes, SCALE, loss="e2e_prog", stal=True)
