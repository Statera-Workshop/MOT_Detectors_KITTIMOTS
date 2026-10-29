"""
YOLOX2 builder (Image task only).

Reference: YOLOX2 has no paper. It is the Legacy author's (yjh0410) own design, so the
Legacy YOLOX2 project is the definition. Its modules were compared file by file:

    backbone / SPPF / PaFPN : identical to Legacy YOLOv8   -> detectors.yolov8.build.YOLOv8Body
    1x1 output projection   : PaFPN outputs -> 256 * width (as the YOLOX stem)
    head                    : v8 SingleLevelHead, cls / reg width = 256 * width (YOLOX-like)
    pred                    : 1x1 cls | 1x1 reg (4), YOLOX box: t_xy * s + p, exp(t_wh) * s
                              (no DFL, no objectness); bias init as the Legacy pred layer
    assigner                : Aligned SimOTA (RTMDet dynamic soft-label assigner)  matcher.py
    loss                    : Quality Focal Loss (IoU target) 1.0 + GIoU 2.0 + L1 aux   loss.py

Delta vs v8 : output projection, equal-width head, exp box instead of DFL, assigner, loss.
Delta vs YOLOX: v8 body instead of v5 body, no objectness, assigner, cls loss (QFL).
"""

import math

import torch
import torch.nn as nn

from detectors.anchor_free.common import HeadStats, anchor_points, decode_exp_boxes, split_targets, to_raw
from detectors.yolov8.build import VARIANT, YOLOv8Body
from detectors.yolov8.yolov8_basic import Conv
from detectors.yolov8.yolov8_config import yolov8_cfg
from detectors.yolov8.yolov8_head import SingleLevelHead
from detectors.yolox2.loss import Criterion

# Legacy yolox2_config.py values that differ from the v8 config (same for every size)
YOLOX2_CFG = {"loss_cls_weight": 1.0, "loss_box_weight": 2.0,
              "matcher_hpy": {"soft_center_radius": 3.0, "topk_candidates": 13}}


class YOLOX2(nn.Module):
    def __init__(self, num_classes, variant=VARIANT):
        super().__init__()
        cfg = {**yolov8_cfg[variant], **YOLOX2_CFG}
        self.num_classes = num_classes
        self.strides = cfg["stride"]
        head_dim = round(256 * cfg["width"])
        self.body = YOLOv8Body(cfg)
        self.out_layers = nn.ModuleList([Conv(d, head_dim, k=1, act_type=cfg["fpn_act"], norm_type=cfg["fpn_norm"])
                                         for d in self.body.out_dim])
        self.heads = nn.ModuleList([SingleLevelHead(head_dim, head_dim, head_dim, cfg["num_cls_head"],
                                                    cfg["num_reg_head"], cfg["head_act"], cfg["head_norm"],
                                                    cfg["head_depthwise"]) for _ in self.strides])
        self.cls_preds = nn.ModuleList([nn.Conv2d(head_dim, num_classes, 1) for _ in self.strides])
        self.reg_preds = nn.ModuleList([nn.Conv2d(head_dim, 4, 1) for _ in self.strides])
        for s, c, r in zip(self.strides, self.cls_preds, self.reg_preds):      # Legacy init_bias
            nn.init.constant_(c.bias, math.log(5 / num_classes / (640. / s) ** 2))
            nn.init.constant_(r.bias, 1.0)
        self.criterion = Criterion(cfg, num_classes)
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def forward(self, x, targets=None):
        feats = [proj(f) for proj, f in zip(self.out_layers, self.body(x))]
        cls_l, reg_l, sizes = [], [], []
        for level, feat in enumerate(feats):
            cls_feat, reg_feat = self.heads[level](feat)
            cls = self.cls_preds[level](cls_feat)
            reg = self.reg_preds[level](reg_feat)
            sizes.append(cls.shape[-2:])
            cls_l.append(cls.flatten(2).permute(0, 2, 1))
            reg_l.append(reg.flatten(2).permute(0, 2, 1))
        anchors, stride_t = anchor_points(sizes, self.strides, x.device, x.dtype)
        box_l = [decode_exp_boxes(reg, a, s) for reg, a, s in zip(reg_l, anchors, stride_t)]
        cls = torch.cat(cls_l, 1)
        raw = to_raw(torch.cat(box_l, 1), cls.sigmoid())
        if targets is None:
            return raw
        outputs = {"pred_cls": cls_l, "pred_box": box_l, "pred_reg": reg_l, "anchors": anchors,
                   "stride_tensors": stride_t, "strides": self.strides}
        loss_dict, fg_mask, fg_labels = self.criterion(outputs, split_targets(targets, x.shape[0], x.shape[2]))
        flat_cls = cls.reshape(-1, self.num_classes)
        self.stats.update(loss_dict, flat_cls.sigmoid().max(-1).values, flat_cls, fg_mask, fg_labels)
        return loss_dict["losses"], raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("YOLOX2 has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    if task != "image":
        raise ValueError("YOLOX2 is implemented for the Image task only (see WORK_CONDITIONS.md).")
    return YOLOX2(num_classes)
