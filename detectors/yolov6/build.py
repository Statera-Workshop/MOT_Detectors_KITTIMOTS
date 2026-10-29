"""
YOLOv6 builder (Image task only).

Paper : Li et al., "YOLOv6: A Single-Stage Object Detection Framework for Industrial
        Applications" (arXiv 2209.02976, v1.0).
Code  : official meituan/YOLOv6, tag 0.2.0 (the release of the paper), copied into this folder.
        There is no Legacy YOLOv6 project.

Algorithm (YOLOv6-N, configs/yolov6n.py of 0.2.0):
    backbone : EfficientRep  (RepVGG blocks, SimSPPF)          efficientrep.py / common_layers.py
               trained with the multi-branch RepVGG block (3x3 + 1x1 + identity), i.e. the
               structural re-parameterisation of the paper (branches are not fused here)
    neck     : Rep-PAN                                         reppan.py
    head     : Efficient Decoupled Head (hybrid channels: 1 stem, 1 cls conv, 1 reg conv)
               anchor-free point prediction, ltrb distances, no DFL (N/S: use_dfl=False)
                                                               effidehead.py
    assigner : ATSS (top-9) for the first 4 epochs, then TAL (top-13, alpha 1, beta 6)
                                                               atss_assigner.py / tal_assigner.py
    loss     : VariFocal loss 1.0 + SIoU loss 2.5              loss.py / figure_iou.py

Shared: common protocol (Adam, 608, flip), detectors/anchor_free HeadStats (logging).
The official eval output [cx, cy, w, h, 1, class probs] is exactly the common raw contract.

Not used (common protocol): SGD + cosine, warm-up, Mosaic / HSV aug, EMA, self-distillation,
    zero-initialised prediction weights (the common normal(0, 0.02) init applies; the prior
    biases of initialize_biases() are kept).
"""

import torch
import torch.nn as nn

from detectors.anchor_free.common import HeadStats
from detectors.yolov6.anchor_generator import generate_anchors
from detectors.yolov6.common_layers import RepVGGBlock
from detectors.yolov6.efficientrep import EfficientRep
from detectors.yolov6.effidehead import Detect, build_effidehead_layer
from detectors.yolov6.general import dist2bbox
from detectors.yolov6.loss import ComputeLoss
from detectors.yolov6.reppan import RepPANNeck

# configs/yolov6n.py (0.2.0)
YOLOV6N = dict(depth_multiple=0.33, width_multiple=0.25,
               backbone_repeats=[1, 6, 12, 18, 6], backbone_channels=[64, 128, 256, 512, 1024],
               neck_repeats=[12, 12, 12, 12], neck_channels=[256, 128, 128, 256, 256, 512],
               strides=[8, 16, 32], iou_type="siou", use_dfl=False, reg_max=0, warmup_epoch=4)


def make_divisible(x, divisor):          # official yolov6/models/yolo.py
    import math
    return math.ceil(x / divisor) * divisor


class YOLOv6(nn.Module):
    def __init__(self, num_classes, cfg=YOLOV6N):
        super().__init__()
        self.num_classes = num_classes
        self.strides = cfg["strides"]
        # official build_network()
        num_repeat = [(max(round(i * cfg["depth_multiple"]), 1) if i > 1 else i)
                      for i in (cfg["backbone_repeats"] + cfg["neck_repeats"])]
        channels_list = [make_divisible(i * cfg["width_multiple"], 8)
                         for i in (cfg["backbone_channels"] + cfg["neck_channels"])]
        self.backbone = EfficientRep(in_channels=3, channels_list=channels_list,
                                     num_repeats=num_repeat, block=RepVGGBlock)
        self.neck = RepPANNeck(channels_list=channels_list, num_repeats=num_repeat, block=RepVGGBlock)
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, cfg["reg_max"])
        self.detect = Detect(num_classes, 1, len(self.strides), head_layers=head_layers, use_dfl=cfg["use_dfl"])
        self.detect.initialize_biases()
        self.criterion = ComputeLoss(fpn_strides=self.strides, num_classes=num_classes,
                                     warmup_epoch=cfg["warmup_epoch"], use_dfl=cfg["use_dfl"],
                                     reg_max=cfg["reg_max"], iou_type=cfg["iou_type"])
        self.epoch = 0
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def set_epoch(self, epoch):
        """0-based epoch from common.trainer (ATSS warm-up for epochs 0..3)."""
        self.epoch = epoch

    def forward(self, x, targets=None):
        feats = self.neck(self.backbone(x))
        if targets is None:
            return self.detect(feats)                    # eval branch: [B, N, 5 + C]
        feats, cls_scores, reg_distri = self.detect(feats)
        self.criterion.ori_img_size = x.shape[2]         # [Unified] targets are normalised to the input
        loss, _ = self.criterion((feats, cls_scores, reg_distri), targets, self.epoch)
        # raw (same decode as the official eval branch)
        points, stride_t = generate_anchors(feats, self.detect.stride, self.detect.grid_cell_size,
                                            self.detect.grid_cell_offset, device=x.device, is_eval=True)
        boxes = dist2bbox(reg_distri, points, box_format="xywh") * stride_t
        raw = torch.cat([boxes, torch.ones_like(boxes[..., :1]), cls_scores], -1)
        fg_mask, labels = self.criterion.last_assign
        fg = fg_mask.reshape(-1).bool()
        probs = cls_scores.reshape(-1, self.num_classes)
        self.stats.update({"losses": loss.detach()}, probs.max(-1).values, probs, fg, labels.reshape(-1)[fg])
        return loss, raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("YOLOv6 has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    if task != "image":
        raise ValueError("YOLOv6 is implemented for the Image task only (see WORK_CONDITIONS.md).")
    return YOLOv6(num_classes)
