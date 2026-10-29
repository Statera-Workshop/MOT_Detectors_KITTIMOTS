"""
YOLOX builder (Image task only).

Paper: Ge et al., "YOLOX: Exceeding YOLO Series in 2021" (arXiv 2107.08430).
Code : Legacy YOLOX project (yjh0410 style), checked against Megvii-BaseDetection/YOLOX.

Delta vs v5 (detectors/yolov5), exactly the paper's changes:
    backbone + PaFPN  : identical -> imported from detectors/yolov5 (the paper's YOLOX-S/M/L/X
                        use the YOLOv5 backbone; the Legacy yolox_* modules equal ours)
    1x1 stem          : PaFPN out_dim projection to 256 * width (paper Fig. 2)
    decoupled head    : cls branch (2 x 3x3) / reg branch (2 x 3x3) -> cls | box, obj
    anchor-free       : one prediction per point, box = ((t_xy * s + p), exp(t_wh) * s)
    SimOTA            : dynamic-k label assignment (simota.py, Legacy matcher)
    loss              : obj BCE + IoU-aware cls BCE + 5 x IoU loss + L1 (loss.py)

Size: VARIANT follows detectors/yolov5 (yolov5_n = Legacy yolox_n: width 0.25, depth 0.34,
no depthwise), so v5 and YOLOX differ in head / assignment / loss only.
"""

import torch
import torch.nn as nn

from detectors.anchor_free.common import (HeadStats, anchor_points, decode_exp_boxes, init_prior_bias,
                                          split_targets, to_raw)
from detectors.yolov5.build import VARIANT
from detectors.yolov5.yolov5_backbone import build_backbone
from detectors.yolov5.yolov5_config import yolov5_cfg
from detectors.yolov5.yolov5_pafpn import build_fpn
from detectors.yolox.loss import Criterion
from detectors.yolox.yolox_head import build_head

# Legacy yolox_config.py head / matcher settings (identical for every YOLOX size)
YOLOX_HEAD = {
    "head_act": "silu", "head_norm": "BN", "head_depthwise": False,
    "num_cls_head": 2, "num_reg_head": 2,
    "center_sampling_radius": 2.5, "topk_candidate": 10,
}


class YOLOX(nn.Module):
    def __init__(self, num_classes, variant=VARIANT):
        super().__init__()
        cfg = {**yolov5_cfg[variant], **YOLOX_HEAD}
        self.num_classes = num_classes
        self.strides = cfg["stride"]
        self.backbone, feat_dims = build_backbone(cfg)
        self.fpn = build_fpn(cfg, feat_dims, out_dim=round(256 * cfg["width"]))
        self.heads = nn.ModuleList([build_head(cfg, d, d, num_classes) for d in self.fpn.out_dim])
        self.obj_preds = nn.ModuleList([nn.Conv2d(h.reg_out_dim, 1, 1) for h in self.heads])
        self.cls_preds = nn.ModuleList([nn.Conv2d(h.cls_out_dim, num_classes, 1) for h in self.heads])
        self.reg_preds = nn.ModuleList([nn.Conv2d(h.reg_out_dim, 4, 1) for h in self.heads])
        for conv in list(self.obj_preds) + list(self.cls_preds):
            init_prior_bias(conv, 0.01)        # Legacy build.py / official initialize_biases
        self.criterion = Criterion(num_classes, cfg["center_sampling_radius"], cfg["topk_candidate"])
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def forward(self, x, targets=None):
        feats = self.fpn(self.backbone(x))
        obj_l, cls_l, reg_l, sizes = [], [], [], []
        for level, feat in enumerate(feats):
            cls_feat, reg_feat = self.heads[level](feat)
            obj = self.obj_preds[level](reg_feat)
            cls = self.cls_preds[level](cls_feat)
            reg = self.reg_preds[level](reg_feat)
            sizes.append(obj.shape[-2:])
            obj_l.append(obj.flatten(2).permute(0, 2, 1))     # [B, HW, 1]
            cls_l.append(cls.flatten(2).permute(0, 2, 1))     # [B, HW, C]
            reg_l.append(reg.flatten(2).permute(0, 2, 1))     # [B, HW, 4]
        anchors, stride_t = anchor_points(sizes, self.strides, x.device, x.dtype)
        box_l = [decode_exp_boxes(reg, a, s) for reg, a, s in zip(reg_l, anchors, stride_t)]

        raw = to_raw(torch.cat(box_l, 1), torch.cat(cls_l, 1).sigmoid(), torch.cat(obj_l, 1).sigmoid())
        if targets is None:
            return raw

        outputs = {"pred_obj": obj_l, "pred_cls": cls_l, "pred_box": box_l, "pred_reg": reg_l,
                   "anchors": anchors, "stride_tensors": stride_t, "strides": self.strides}
        loss_dict, fg_mask, fg_labels = self.criterion(outputs, split_targets(targets, x.shape[0], x.shape[2]))
        self.stats.update(loss_dict, torch.cat(obj_l, 1).sigmoid().view(-1),
                          torch.cat(cls_l, 1).reshape(-1, self.num_classes), fg_mask, fg_labels)
        return loss_dict["losses"], raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("YOLOX has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    if task != "image":
        raise ValueError("YOLOX is implemented for the Image task only (see WORK_CONDITIONS.md).")
    return YOLOX(num_classes)
