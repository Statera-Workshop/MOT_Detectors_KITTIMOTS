"""
YOLOv8 builder (Image task only).

Reference: YOLOv8 has no paper. The Ultralytics release (2023) is the original; the Legacy
YOLOv8 project (yjh0410 style) follows it module by module and is copied here.

Algorithm (this folder, Legacy modules):
    backbone : C2f ("ELAN-CSP") stages, 3x3 stem          yolov8_backbone.py / yolov8_basic.py
    neck     : SPPF on P5                                   yolov8_neck.py
    PaFPN    : C2f blocks, no 1x1 reduce layers             yolov8_pafpn.py
    head     : decoupled, no objectness branch              yolov8_head.py
               reg width max(16, C3/4, 4*reg_max), cls width max(C3, min(nc, 100))
    pred     : DFL box (reg_max 16 bins per side, ltrb distances) yolov8_pred.py
    assigner : Task-Aligned Assigner (top-k 10, alpha 0.5, beta 6) matcher.py
    loss     : BCE(soft targets) 0.5 + CIoU 7.5 + DFL 1.5  loss.py

Shared with the other anchor-free detectors: detectors/anchor_free (targets, raw output, logging).
Raw objectness = 1 (no objectness branch), so the common score = class probability.

Delta vs v5: C3 -> C2f, no FPN reduce layers, decoupled obj-free head, DFL, TAL, loss.
Not used (common protocol): EMA, Mosaic / MixUp, close-mosaic, warm-up, BN eps 1e-3 / momentum 0.03.
"""

import torch
import torch.nn as nn

from detectors.anchor_free.common import HeadStats, split_targets, to_raw
from detectors.yolov8.loss import Criterion
from detectors.yolov8.yolov8_backbone import build_backbone
from detectors.yolov8.yolov8_config import yolov8_cfg
from detectors.yolov8.yolov8_head import build_det_head
from detectors.yolov8.yolov8_neck import build_neck
from detectors.yolov8.yolov8_pafpn import build_fpn
from detectors.yolov8.yolov8_pred import build_pred_layer

VARIANT = "yolov8_n"     # same scale as v5 / YOLOX (width 0.25, depth 0.34)


class YOLOv8Body(nn.Module):
    """Backbone + SPPF + PaFPN (reused by YOLOX2, whose Legacy modules are identical)."""

    def __init__(self, cfg):
        super().__init__()
        self.backbone, feats_dim = build_backbone(cfg)
        self.neck = build_neck(cfg, feats_dim[-1], feats_dim[-1])
        feats_dim[-1] = self.neck.out_dim
        self.fpn = build_fpn(cfg, feats_dim)
        self.out_dim = self.fpn.out_dim

    def forward(self, x):
        feats = self.backbone(x)
        feats[-1] = self.neck(feats[-1])
        return self.fpn(feats)


class YOLOv8(nn.Module):
    def __init__(self, num_classes, variant=VARIANT):
        super().__init__()
        cfg = yolov8_cfg[variant]
        self.num_classes = num_classes
        self.reg_max = cfg["reg_max"]
        self.body = YOLOv8Body(cfg)
        self.det_heads = build_det_head(cfg, self.body.out_dim, len(cfg["stride"]), num_classes, self.reg_max)
        self.pred_layers = build_pred_layer(cls_dim=self.det_heads.cls_head_dim,
                                            reg_dim=self.det_heads.reg_head_dim,
                                            strides=cfg["stride"], num_classes=num_classes,
                                            num_coords=4, num_levels=len(cfg["stride"]),
                                            reg_max=self.reg_max)
        self.criterion = Criterion(cfg, None, num_classes)
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def forward(self, x, targets=None):
        cls_feats, reg_feats = self.det_heads(self.body(x))
        out = self.pred_layers(cls_feats, reg_feats)
        for k in ("anchors", "stride_tensor"):
            out[k] = [a.to(x.device) for a in out[k]]
        cls = torch.cat(out["pred_cls"], 1)
        raw = to_raw(torch.cat(out["pred_box"], 1), cls.sigmoid())
        if targets is None:
            return raw
        loss_dict, fg_mask, fg_labels = self.criterion(out, split_targets(targets, x.shape[0], x.shape[2]))
        flat_cls = cls.reshape(-1, self.num_classes)
        self.stats.update(loss_dict, flat_cls.sigmoid().max(-1).values, flat_cls, fg_mask, fg_labels)
        return loss_dict["losses"], raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("YOLOv8 has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    if task != "image":
        raise ValueError("YOLOv8 is implemented for the Image task only (see WORK_CONDITIONS.md).")
    return YOLOv8(num_classes)
